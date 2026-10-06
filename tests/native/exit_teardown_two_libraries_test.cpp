// Exit teardown next to libllamadart in one process.
//
// Usage: exit_teardown_two_libraries_test <libstable-diffusion> <libllamadart>
//            sd|llama exit|sd|llama <model> <backend>|default
//
// libllamadart (leehack/llamadart-native) embeds its own ggml and carries the
// same registry and the same hidden __cxa_atexit for it. The third argument
// says which library is loaded first. Each library then tracks an object, and
// this one also a real context. With `exit` the process returns from main and
// both teardowns have to run by themselves; with `sd` or `llama` that
// library's teardown is called first and must leave the other registry alone.
// Every free is printed, so the caller sees each object freed once.

#include "sd_dart_wrapper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

template <typename Function>
Function symbol(void* library, const char* name) {
    void* address = dlsym(library, name);
    if (address == nullptr) {
        std::fprintf(stderr, "missing %s: %s\n", name, dlerror());
        std::_Exit(1);
    }
    return reinterpret_cast<Function>(address);
}

void report(const char* line) {
    CHECK(write(STDOUT_FILENO, line, std::strlen(line)) > 0);
}

void free_sd_object(void*) {
    report("freed by libstable-diffusion\n");
}

void free_llama_object(void*) {
    report("freed by libllamadart\n");
}

void quiet_log(enum sd_log_level_t, const char*, void*) {}
void quiet_progress(int, int, float, void*) {}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s <libstable-diffusion> <libllamadart> sd|llama exit|sd|llama <model> <backend>\n",
                     argv[0]);
        return 2;
    }
    const bool sd_first    = std::strcmp(argv[3], "sd") == 0;
    const std::string mode = argv[4];
    void* first            = dlopen(argv[sd_first ? 1 : 2], RTLD_NOW | RTLD_LOCAL);
    void* second           = dlopen(argv[sd_first ? 2 : 1], RTLD_NOW | RTLD_LOCAL);
    CHECK(first != nullptr && second != nullptr);
    void* sd    = sd_first ? first : second;
    void* llama = sd_first ? second : first;

    using Track    = bool (*)(void*, void (*)(void*), int32_t);
    using Count    = int32_t (*)();
    using Teardown = void (*)();
    const auto sd_track       = symbol<Track>(sd, "sd_dart_exit_track");
    const auto sd_count       = symbol<Count>(sd, "sd_dart_exit_tracked_count");
    const auto sd_teardown    = symbol<Teardown>(sd, "sd_dart_exit_teardown");
    const auto llama_track    = symbol<Track>(llama, "llama_dart_exit_track");
    const auto llama_count    = symbol<Count>(llama, "llama_dart_exit_tracked_count");
    const auto llama_teardown = symbol<Teardown>(llama, "llama_dart_exit_teardown");

    symbol<void (*)(sd_log_cb_t, void*)>(sd, "sd_set_log_callback")(quiet_log, nullptr);
    symbol<void (*)(sd_progress_cb_t, void*)>(sd, "sd_set_progress_callback")(quiet_progress, nullptr);
    sd_ctx_params_t params;
    symbol<void (*)(sd_ctx_params_t*)>(sd, "sd_ctx_params_init")(&params);
    params.model_path = argv[5];
    params.taesd_path = argv[5];
    params.backend    = std::strcmp(argv[6], "default") == 0 ? nullptr : argv[6];
    params.eager_load = true;

    static char sd_object[]    = "sd";
    static char llama_object[] = "llama";
    // The library that was loaded first tracks last, so that the two orders
    // also differ in which teardown is registered first.
    if (sd_first) {
        CHECK(llama_track(llama_object, free_llama_object, 0));
    }
    CHECK(sd_track(sd_object, free_sd_object, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(symbol<sd_ctx_t* (*)(const sd_ctx_params_t*)>(sd, "sd_dart_new_sd_ctx")(&params) != nullptr);
    if (!sd_first) {
        CHECK(llama_track(llama_object, free_llama_object, 0));
    }
    CHECK(sd_count() == 2 && llama_count() == 1);

    if (mode == "sd") {
        sd_teardown();
        CHECK(sd_count() == 0 && llama_count() == 1);
    } else if (mode == "llama") {
        llama_teardown();
        CHECK(sd_count() == 2 && llama_count() == 0);
    }
    return 0;
}

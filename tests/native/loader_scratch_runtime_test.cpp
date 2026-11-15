// Loads a model, generates once and reports the largest `operator new` request
// a thread other than the caller's made meanwhile.
//
// Usage: loader_scratch_runtime_test devices
//        loader_scratch_runtime_test run <model> <backend> <output.rgb>
//
// upstream's loader reads and converts tensors on its own threads. This
// program replaces the process's `operator new`, which the library's vectors
// use too, so a loader that takes its tensor-sized scratch from the heap shows
// up as `WORKER_NEW <bytes>`.

#include "sd_dart_wrapper.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <pthread.h>
#include <vector>

static std::atomic<size_t> largest_worker_new{0};

// Metal's own threads allocate too; only the library's requests count.
static bool from_library(void* caller) {
    Dl_info info;
    return dladdr(caller, &info) != 0 && info.dli_fname != nullptr &&
           std::strstr(info.dli_fname, "libstable-diffusion") != nullptr;
}

extern "C" void sd_test_vm_report();

void* operator new(size_t size) {
    size_t seen = largest_worker_new.load();
    if (size > seen && pthread_main_np() == 0 && from_library(__builtin_return_address(0))) {
        while (size > seen && !largest_worker_new.compare_exchange_weak(seen, size)) {
        }
    }
    void* memory = std::malloc(size == 0 ? 1 : size);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

namespace {

int list_devices() {
    std::vector<char> devices(sd_list_devices(nullptr, 0) + 1);
    sd_list_devices(devices.data(), devices.size());
    std::printf("%s", devices.data());
    return 0;
}

int fail(const char* what) {
    std::vector<char> error(4096);
    sd_dart_last_error(error.data(), error.size());
    std::fprintf(stderr, "%s: %s\n", what, error.data());
    return 3;
}

int run(const char* model, const char* backend, const char* output) {
    sd_dart_log_enable();
    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    context_params.model_path = model;
    context_params.taesd_path = model;
    context_params.backend    = backend;
    context_params.n_threads  = 1;  // deterministic VM-region observations
    sd_ctx_t* context         = sd_dart_new_sd_ctx(&context_params);
    if (context == nullptr) {
        return fail("load failed");
    }
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt                         = "a lighthouse";
    params.negative_prompt                = "";
    params.width                          = 64;
    params.height                         = 64;
    params.seed                           = 7;
    params.sample_params.sample_steps     = 1;
    params.sample_params.guidance.txt_cfg = 1.0f;
    sd_image_t* images                    = nullptr;
    int generated                         = 0;
    const bool ok                         = sd_dart_generate_image(context, &params, &images, &generated);
    if (ok) {
        if (generated != 1) return fail("unexpected image count");
        FILE* file = std::fopen(output, "wb");
        const size_t bytes = static_cast<size_t>(images[0].width) * images[0].height * images[0].channel;
        if (file == nullptr || std::fwrite(images[0].data, 1, bytes, file) != bytes || std::fclose(file) != 0) {
            return fail("cannot write image");
        }
        free_sd_images(images, generated);
    }
    sd_dart_exit_free(context);
    if (!ok) {
        return fail("generation failed");
    }
    std::printf("WORKER_NEW %zu\n", largest_worker_new.load());
    sd_test_vm_report();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "devices") == 0) {
        return list_devices();
    }
    if (argc == 5 && std::strcmp(argv[1], "run") == 0) {
        return run(argv[2], argv[3], argv[4]);
    }
    std::fprintf(stderr, "usage: %s devices | run <model> <backend> <output.rgb>\n", argv[0]);
    return 2;
}

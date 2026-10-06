// Exit teardown against a built runtime and a real sd_ctx_t.
//
// Usage: exit_teardown_runtime_test make-model <model>
//        exit_teardown_runtime_test <scenario> <model> <backend>|default [<size>]
//
// `make-model` writes a PixArt model with one small transformer block, a
// one-block T5 text encoder and a TAESD decoder: 12 MB that the runtime loads
// and samples like any other model, on Metal as well as on the CPU. Nothing
// is downloaded. A model that carries no TAESD decoder, as real ones do, is
// loaded as it is; give the image size it needs.
//
// The other scenarios each run in their own process, as teardown runs once
// per process, and most of them leave by returning from main: on Apple
// platforms teardown then runs during exit(). Elsewhere it does not, and only
// `idle-untracked` applies: a context that is alive at exit must not abort.
//
// `generate` only loads, generates and frees, to tell whether the backend can
// compute on this machine at all. Where it cannot, SD_EXIT_TEARDOWN_LOAD_ONLY
// makes the scenarios that can do without a generation on that backend skip
// it: a loaded context already holds the backend's buffers.

#include "sd_dart_wrapper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SD_TEST_ADDRESS_SANITIZER 1
#endif
#endif

#if defined(SD_TEST_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int64_t elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - since).count();
}

// Whether the heap block that began at `object` has been freed.
// AddressSanitizer poisons a freed block, and the macOS allocator reports no
// size for one until it hands the block out again. No scenario that asks
// runs anywhere else.
bool is_freed(const void* object) {
#if defined(SD_TEST_ADDRESS_SANITIZER)
    return __asan_address_is_poisoned(object) != 0;
#elif defined(__APPLE__)
    return malloc_size(object) == 0;
#else
    (void)object;
    return false;
#endif
}

// The object that owns the weights, runners and backends of a context: the
// only member of upstream's sd_ctx_t. Unlike the 8-byte sd_ctx_t itself it is
// large enough for every allocator to tell whether it is freed.
const void* engine_of(const sd_ctx_t* context) {
    const void* engine = *reinterpret_cast<const void* const*>(context);
    CHECK(engine != nullptr && !is_freed(engine));
    return engine;
}

std::mutex log_mutex;
std::string log_text;
std::atomic<void (*)()> on_log{nullptr};

void collect_log(enum sd_log_level_t level, const char* text, void*) {
    if (void (*hook)() = on_log.load()) {
        hook();
    }
    std::lock_guard<std::mutex> lock(log_mutex);
    if (level >= SD_LOG_WARN) {
        log_text += text;
    }
}

std::atomic<void (*)(int)> on_progress{nullptr};

void collect_progress(int step, int, float, void*) {
    if (void (*hook)(int) = on_progress.load()) {
        hook(step);
    }
}

int image_size = 64;
bool load_only = false;

// The model `make-model` writes holds its TAESD decoder itself.
bool holds_taesd(const char* model) {
    std::string header(1 << 16, '\0');
    FILE* file = std::fopen(model, "rb");
    if (file == nullptr) {
        return false;
    }
    header.resize(std::fread(&header[0], 1, header.size(), file));
    std::fclose(file);
    return header.find("\"tae.decoder.") != std::string::npos;
}

sd_ctx_params_t context_params(const char* model, const char* backend, bool taesd) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = model;
    params.taesd_path = taesd ? model : nullptr;
    params.backend    = std::strcmp(backend, "default") == 0 ? nullptr : backend;
    params.eager_load = true;
    return params;
}

sd_ctx_t* load(const char* model, const char* backend, bool tracked = true) {
    const sd_ctx_params_t params = context_params(model, backend, holds_taesd(model));
    sd_ctx_t* context            = tracked ? sd_dart_new_sd_ctx(&params) : new_sd_ctx(&params);
    if (context == nullptr) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::fprintf(stderr, "%s", log_text.c_str());
    }
    CHECK(context != nullptr);
    return context;
}

// Returns the number of images, or -1 when the generation failed.
int generate(sd_ctx_t* context, int steps, bool guarded = true) {
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt                         = "a lighthouse";
    params.negative_prompt                = "";
    params.width                          = image_size;
    params.height                         = image_size;
    params.seed                           = 7;
    params.sample_params.sample_steps     = steps;
    params.sample_params.guidance.txt_cfg = 1.0f;
    sd_image_t* images                    = nullptr;
    int count                             = 0;
    const bool ok                         = guarded ? sd_dart_generate_image(context, &params, &images, &count)
                                                    : generate_image(context, &params, &images, &count);
    if (images != nullptr) {
        free_sd_images(images, count);
    }
    return ok ? count : -1;
}

using Tensors = std::map<std::string, std::vector<int64_t>>;

bool write_model(const char* path, const Tensors& tensors) {
    std::string header = "{";
    uint64_t offset    = 0;
    for (const auto& [name, shape] : tensors) {
        uint64_t elements = 1;
        std::string dims;
        for (int64_t dim : shape) {
            elements *= static_cast<uint64_t>(dim);
            dims += (dims.empty() ? "" : ",") + std::to_string(dim);
        }
        header += (offset == 0 ? "\"" : ",\"") + name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
                  "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + 4 * elements) + "]}";
        offset += 4 * elements;
    }
    header += "}";
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        return false;
    }
    const uint64_t header_size = header.size();
    std::fwrite(&header_size, sizeof(header_size), 1, file);
    std::fwrite(header.data(), 1, header.size(), file);
    const std::vector<float> weights(16384, 0.02f);
    for (uint64_t written = 0; written < offset;) {
        const uint64_t bytes = std::min<uint64_t>(offset - written, weights.size() * sizeof(float));
        std::fwrite(weights.data(), 1, bytes, file);
        written += bytes;
    }
    return std::fclose(file) == 0;
}

// The tensors the runtime reads the model's dimensions from are given here.
// The rest are whatever its own validation asks for: a load that fails names
// each missing tensor and the shape it wants, so the model follows the
// pinned upstream revision without a tensor list to maintain.
int make_model(const char* path) {
    const int64_t hidden = 32, caption = 16, vocabulary = 32128;
    const std::string dit = "model.diffusion_model.";
    const std::string t5  = "text_encoders.t5xxl.transformer.";
    Tensors tensors       = {
        {dit + "t_block.1.weight", {6 * hidden, hidden}},
        {dit + "x_embedder.proj.weight", {hidden, 4, 2, 2}},
        {dit + "y_embedder.y_proj.fc1.weight", {hidden, caption}},
        {dit + "blocks.0.mlp.fc1.weight", {2 * hidden, hidden}},
        {dit + "blocks.0.cross_attn.kv_linear.weight", {2 * hidden, hidden}},
        {dit + "final_layer.linear.weight", {32, hidden}},
        {t5 + "shared.weight", {vocabulary, caption}},
        {t5 + "encoder.block.0.layer.0.SelfAttention.q.weight", {64, caption}},
        {t5 + "encoder.block.0.layer.1.DenseReluDense.wi_0.weight", {2 * caption, caption}},
    };
    const std::regex missing("tensor '([^']+)' not in model metadata");
    const std::regex wrong("tensor '([^']+)' has wrong shape in model metadata: got \\[[^\\]]*\\], "
                           "expected \\[(\\d+), (\\d+), (\\d+), (\\d+)\\]");
    for (int attempt = 0; attempt < 8; ++attempt) {
        CHECK(write_model(path, tensors));
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            log_text.clear();
        }
        // With a TAESD decoder the model needs no VAE, which would be 320 MB.
        const sd_ctx_params_t params = context_params(path, "cpu", true);
        if (sd_ctx_t* context = new_sd_ctx(&params)) {
            free_sd_ctx(context);
            CHECK(holds_taesd(path));
            return 0;
        }
        std::lock_guard<std::mutex> lock(log_mutex);
        bool changed = false;
        for (std::sregex_iterator it(log_text.begin(), log_text.end(), missing), end; it != end; ++it) {
            changed |= tensors.insert({(*it)[1], {1}}).second;
        }
        for (std::sregex_iterator it(log_text.begin(), log_text.end(), wrong), end; it != end; ++it) {
            // ggml lists dimensions innermost first; safetensors outermost.
            std::vector<int64_t> shape;
            for (int dim = 5; dim >= 2; --dim) {
                const int64_t size = std::stoll((*it)[dim]);
                if (size != 1 || !shape.empty() || dim == 2) {
                    shape.push_back(size);
                }
            }
            changed |= tensors[(*it)[1]] != shape;
            tensors[(*it)[1]] = shape;
        }
        if (!changed) {
            std::fprintf(stderr, "the runtime rejected the model without naming a tensor:\n%s", log_text.c_str());
            return 1;
        }
    }
    std::fprintf(stderr, "the model did not settle\n");
    return 1;
}

// Teardown blocks a thread that reaches the registry outside a call in
// flight. This one keeps reaching it, so when its count stops after the
// scenario asked for teardown, teardown has begun.
std::atomic<int64_t> heartbeat{0};
std::atomic<bool> teardown_requested{false};

void start_heartbeat() {
    std::thread([] {
        for (;;) {
            sd_dart_exit_untrack(nullptr);
            heartbeat.fetch_add(1);
            sleep_ms(1);
        }
    }).detach();
    while (heartbeat.load() == 0) {
        sleep_ms(1);
    }
}

bool wait_for_teardown() {
    while (!teardown_requested.load()) {
        sleep_ms(1);
    }
    const auto started = std::chrono::steady_clock::now();
    int64_t seen       = heartbeat.load();
    auto changed       = std::chrono::steady_clock::now();
    while (elapsed_ms(started) < 60000) {
        sleep_ms(5);
        const int64_t now = heartbeat.load();
        if (now != seen) {
            seen    = now;
            changed = std::chrono::steady_clock::now();
        } else if (elapsed_ms(changed) > 400) {
            return true;
        }
    }
    return false;
}

// What the two objects tracked around the contexts saw during teardown.
struct Witness {
    std::atomic<const void*> engine{nullptr};
    std::atomic<int> expected_contexts{1};
    std::atomic<bool> before_ran{false};
    std::atomic<bool> after_ran{false};
    // What else the scenario expects once teardown is over; null if nothing.
    std::atomic<const char* (*)()> failure{nullptr};
};
Witness witness;
char witness_before[] = "before";
char witness_after[]  = "after";

// Freed before any context: every context is still there.
void free_before_contexts(void*) {
    CHECK(sd_dart_exit_tracked_count() == witness.expected_contexts.load() + 1);
    const void* engine = witness.engine.load();
    CHECK(engine == nullptr || !is_freed(engine));
    witness.before_ran.store(true);
}

// Freed after every context, with nothing allocated in between.
void free_after_contexts(void*) {
    CHECK(sd_dart_exit_tracked_count() == 0);
    const void* engine = witness.engine.load();
    CHECK(engine == nullptr || is_freed(engine));
    witness.after_ran.store(true);
}

void expect_teardown_ran() {
    const char* (*failure)() = witness.failure.load();
    const char* message      = failure != nullptr ? failure() : nullptr;
    if (!witness.before_ran.load() || !witness.after_ran.load()) {
        message = "exit teardown did not free the tracked objects";
    }
    if (message != nullptr) {
        std::fprintf(stderr, "%s\n", message);
        std::_Exit(1);
    }
}

// Call before anything is tracked: the check then runs after the teardown
// that tracking registers.
void expect_teardown_at_exit(int contexts) {
    witness.expected_contexts.store(contexts);
    CHECK(atexit(expect_teardown_ran) == 0);
    CHECK(sd_dart_exit_track(witness_before, free_before_contexts, SD_DART_EXIT_STAGE_CONTEXT_USER));
}

// Call once the contexts exist, where the scenario can. Within a stage the
// latest tracked object goes first, so this witness would also be freed
// before a context that was tracked in its stage.
void expect_contexts_freed_first() {
    CHECK(sd_dart_exit_track(witness_after, free_after_contexts, SD_DART_EXIT_STAGE_RESOURCE));
}

// Exits with a loaded context that nothing frees.
int test_idle(const char* model, const char* backend, bool tracked) {
    if (tracked) {
        expect_teardown_at_exit(1);
    }
    sd_ctx_t* context = load(model, backend, tracked);
    CHECK(load_only || generate(context, 2, tracked) == 1);
    CHECK(sd_dart_exit_tracked_count() == (tracked ? 2 : 0));
    if (tracked) {
        witness.engine.store(engine_of(context));
        expect_contexts_freed_first();
    }
    // Tells the caller that whatever follows happened during exit().
    std::printf("%s on %s\n", load_only ? "loaded" : "generated", backend);
    std::fflush(stdout);
    return 0;
}

int test_generate(const char* model, const char* backend) {
    sd_ctx_t* context = load(model, backend, false);
    CHECK(generate(context, 1, false) == 1);
    free_sd_ctx(context);
    return 0;
}

int test_dispose(const char* model, const char* backend) {
    for (int i = 0; i < 3; ++i) {
        sd_ctx_t* context  = load(model, backend);
        const void* engine = engine_of(context);
        if (i != 1 && !load_only) {
            CHECK(generate(context, 1) == 1);
        }
        CHECK(sd_ctx_supports_image_generation(context));
        CHECK(sd_get_model_version_name(context) != nullptr);
        CHECK(sd_dart_exit_tracked_count() == 1);
        sd_dart_exit_free(context);
        CHECK(is_freed(engine));
        CHECK(sd_dart_exit_tracked_count() == 0);
        sd_dart_exit_free(context);
        sd_dart_cancel_generation(context, SD_CANCEL_ALL);
    }
    return 0;
}

std::atomic<int> steps_seen{0};
std::atomic<bool> cancel_requested{false};

// Cancelling through the wrapper ends a generation on a tracked context.
int test_cancel(const char* model, const char* backend) {
    static sd_ctx_t* context = load(model, backend);
    on_progress.store([](int) {
        steps_seen.fetch_add(1);
        if (!cancel_requested.exchange(true)) {
            std::thread([] { sd_dart_cancel_generation(context, SD_CANCEL_ALL); }).join();
        }
    });
    generate(context, 40);
    CHECK(steps_seen.load() > 0 && steps_seen.load() < 10);
    sd_dart_exit_free(context);
    return 0;
}

std::atomic<bool> held{false};
std::atomic<bool> released{false};
std::atomic<int> steps_after_release{0};

// A generation in flight when the process exits: teardown cancels it, waits
// for it to end and then frees the context it ran on.
int test_generate_wait(const char* model, const char* backend) {
    witness.failure.store([]() -> const char* {
        if (!released.load()) {
            return "teardown did not wait for the generation";
        }
        return steps_after_release.load() > 3 ? "teardown did not cancel the generation" : nullptr;
    });
    expect_teardown_at_exit(1);
    start_heartbeat();
    static sd_ctx_t* context = load(model, backend);
    witness.engine.store(engine_of(context));
    expect_contexts_freed_first();
    on_progress.store([](int) {
        if (released.load()) {
            steps_after_release.fetch_add(1);
            return;
        }
        if (held.exchange(true)) {
            return;
        }
        CHECK(wait_for_teardown());
        const void* engine = witness.engine.load();
        CHECK(!is_freed(engine));
        sleep_ms(100);
        CHECK(!is_freed(engine));
        CHECK(!witness.before_ran.load());
        released.store(true);
    });
    std::thread([] {
        generate(context, 40);
        std::fprintf(stderr, "a call in flight returned after teardown\n");
        std::_Exit(1);
    }).detach();
    while (!held.load()) {
        sleep_ms(1);
    }
    teardown_requested.store(true);
    return 0;
}

// A load in flight when the process exits: teardown waits for it, and the
// context it creates is tracked and freed although teardown has begun.
int test_load_wait(const char* model, const char* backend) {
    witness.failure.store([]() -> const char* {
        return released.load() ? nullptr : "teardown did not wait for the load";
    });
    expect_teardown_at_exit(1);
    expect_contexts_freed_first();
    start_heartbeat();
    static const char* load_model   = model;
    static const char* load_backend = backend;
    on_log.store([] {
        if (held.exchange(true)) {
            return;
        }
        CHECK(wait_for_teardown());
        CHECK(sd_dart_exit_tracked_count() == 2);
        released.store(true);
    });
    std::thread([] {
        load(load_model, load_backend);
        std::fprintf(stderr, "a call in flight returned after teardown\n");
        std::_Exit(1);
    }).detach();
    while (!held.load()) {
        sleep_ms(1);
    }
    teardown_requested.store(true);
    return 0;
}

// A context on another device, loaded after the first one was tracked, makes
// the runtime create statics that did not exist when teardown was registered.
// They are destroyed after teardown all the same.
int test_late_load(const char* model, const char* backend) {
    expect_teardown_at_exit(2);
    sd_ctx_t* first = load(model, "cpu");
    CHECK(generate(first, 1) == 1);
    sd_ctx_t* second = load(model, backend);
    CHECK(load_only || generate(second, 1) == 1);
    // The first context is freed last, so nothing reuses its block before the
    // witness looks at it.
    witness.engine.store(engine_of(first));
    expect_contexts_freed_first();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "";
    sd_set_log_callback(collect_log, nullptr);
    sd_set_progress_callback(collect_progress, nullptr);
    // The scenarios are about what teardown does once the calls in flight
    // have ended. A busy machine must not turn that into the timeout.
    sd_dart_exit_set_wait_ms(120000);
    if (argc == 3 && scenario == "make-model") {
        return make_model(argv[2]);
    }
    if (argc == 4 || argc == 5) {
        const char* model   = argv[2];
        const char* backend = argv[3];
        image_size          = argc == 5 ? std::atoi(argv[4]) : image_size;
        load_only           = std::getenv("SD_EXIT_TEARDOWN_LOAD_ONLY") != nullptr;
        if (scenario == "generate") {
            return test_generate(model, backend);
        }
        if (scenario == "idle") {
            return test_idle(model, backend, true);
        }
        if (scenario == "idle-untracked") {
            return test_idle(model, backend, false);
        }
        if (scenario == "dispose") {
            return test_dispose(model, backend);
        }
        if (scenario == "cancel") {
            return test_cancel(model, backend);
        }
        if (scenario == "generate-wait") {
            return test_generate_wait(model, backend);
        }
        if (scenario == "load-wait") {
            return test_load_wait(model, backend);
        }
        if (scenario == "late-load") {
            return test_late_load(model, backend);
        }
    }
    std::fprintf(stderr, "usage: %s make-model <model> | <scenario> <model> <backend>|default [<size>]\n", argv[0]);
    return 2;
}

// Exercises src/sd_dart_exit.cpp against stand-ins for the upstream functions
// it wraps and for ggml's device registry, so it needs no model and runs
// under sanitizers.
//
// Each scenario runs in its own process: teardown runs once per process and
// leaves the registry unusable from other threads.

#include "sd_dart_wrapper.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SD_TEST_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(SD_TEST_ADDRESS_SANITIZER)
#define SD_TEST_ADDRESS_SANITIZER 1
#endif

#if defined(SD_TEST_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

// Where exit() runs the wait of teardown and nothing else, as in
// src/sd_dart_exit.cpp.
#if defined(__linux__) && !defined(__ANDROID__)
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#define SD_TEST_EXIT_ON_LINUX 1
#else
#define SD_TEST_EXIT_ON_LINUX 0
#endif

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

// The stand-in for upstream: a context is a heap block that knows whether it
// is being generated on and whether a cancellation is pending.
struct sd_ctx_t {
    std::atomic<int> cancel{SD_CANCEL_RESET};
    std::atomic<int> cancel_requests{0};
    std::atomic<bool> generating{false};
};

namespace {

std::mutex log_mutex;
std::vector<std::string> log_entries;

void record(const std::string& event) {
    std::lock_guard<std::mutex> lock(log_mutex);
    log_entries.push_back(event);
}

std::vector<std::string> recorded() {
    std::lock_guard<std::mutex> lock(log_mutex);
    return log_entries;
}

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int64_t elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - since).count();
}

// Whether the heap block that began at `object` has been freed, for a build
// that can tell: AddressSanitizer poisons a freed block, and the macOS
// allocator reports no size for one until it hands the block out again.
#if defined(SD_TEST_ADDRESS_SANITIZER)
const bool kSeesFreedBlocks = true;
bool is_freed(const void* object) {
    return __asan_address_is_poisoned(object) != 0;
}
#elif defined(__APPLE__)
const bool kSeesFreedBlocks = true;
bool is_freed(const void* object) {
    return malloc_size(object) == 0;
}
#else
const bool kSeesFreedBlocks = false;
bool is_freed(const void*) {
    return false;
}
#endif

// What exit() does with the registry: teardown on Apple platforms, which
// frees. On Linux it frees nothing, destroys no static and blocks no thread:
// with no call in flight it only refuses the calls that arrive later, and
// with one in flight it ends the process.
#if defined(__APPLE__)
const bool kExitRuns  = true;
const bool kExitFrees = true;
#elif SD_TEST_EXIT_ON_LINUX
const bool kExitRuns  = true;
const bool kExitFrees = false;
#else
const bool kExitRuns  = false;
const bool kExitFrees = false;
#endif

// Runs `handler` during exit(), after whatever is registered later. This
// program is one image with the registry, whose __cxa_atexit stands in for the
// C library's. On Linux that definition registers the wait in the place of
// what it is given, and glibc's atexit() is linked into the calling image and
// reaches it, so there the handler goes to the C library itself.
bool at_exit(void (*handler)()) {
#if SD_TEST_EXIT_ON_LINUX
    // As atexit() itself registers a handler; ThreadSanitizer's __cxa_atexit
    // passes no argument on.
    using Register             = int (*)(void (*)(void*), void*, void*);
    const auto system_register = reinterpret_cast<Register>(dlsym(RTLD_DEFAULT, "__cxa_atexit"));
    return system_register != nullptr &&
           system_register(reinterpret_cast<void (*)(void*)>(handler), nullptr, nullptr) == 0;
#else
    return atexit(handler) == 0;
#endif
}

// What a scenario returns when it leaves its verdict to a check that runs
// during exit(). On Linux, where exit() can end the process before the
// host's handlers, the check ends it with 0, and this status says that the
// exit never reached the check.
#if SD_TEST_EXIT_ON_LINUX
const int kNotChecked = 3;
#else
const int kNotChecked = 0;
#endif
void (*exit_check)() = nullptr;

bool check_at_exit(void (*check)()) {
    exit_check = check;
    return at_exit([] {
        exit_check();
        if (kNotChecked != 0) {
            std::_Exit(0);
        }
    });
}

// Whether teardown or exit() has begun, for a thread that it does not block:
// the one that ran teardown, and any thread where exit() frees nothing.
bool teardown_refuses_this_thread() {
    static char probe[] = "probe";
    if (sd_dart_exit_track(probe, [](void*) {}, SD_DART_EXIT_STAGE_RESOURCE)) {
        sd_dart_exit_untrack(probe);
        return false;
    }
    return true;
}

// The stand-in for upstream's log: nothing without a callback, and a text
// that is gone when the callback returns.
std::atomic<sd_log_cb_t> log_callback{nullptr};

void upstream_log(enum sd_log_level_t level, const std::string& text) {
    if (sd_log_cb_t callback = log_callback.load()) {
        std::string passed = text + "\n";
        callback(level, passed.c_str(), nullptr);
        std::fill(passed.begin(), passed.end(), '#');
    }
}

std::string last_error() {
    char text[1024];
    const size_t length = sd_dart_last_error(text, sizeof(text));
    CHECK(length < sizeof(text) && std::strlen(text) == length);
    return text;
}

// What the stand-ins do, set by the scenario before it calls a wrapper.
std::atomic<bool> fail_load{false};
std::atomic<bool (*)()> hold_load{nullptr};
std::atomic<bool (*)(sd_ctx_t*)> run_generation{nullptr};
std::atomic<void (*)()> hold_free{nullptr};
std::atomic<int> contexts_freed{0};
std::atomic<sd_ctx_t*> last_freed{nullptr};
// Upstream calls that began after teardown had returned.
std::atomic<bool> teardown_returned{false};
std::atomic<int> loads_begun{0};
std::atomic<int> calls_after_teardown{0};
// The arguments the stand-ins were last called with.
std::atomic<const sd_ctx_params_t*> last_load_params{nullptr};
std::atomic<const sd_img_gen_params_t*> last_request{nullptr};
std::atomic<sd_image_t**> last_images_out{nullptr};

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
    while (elapsed_ms(started) < 20000) {
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

void free_named(void* object) {
    record(static_cast<const char*>(object));
}

char* name(const char* value) {
    return const_cast<char*>(value);
}

char kUserLate[] = "user-late";

const std::vector<std::string> kStageOrder = {
    "user-late",
    "user-early",
    "context-late",
    "context-early",
    "resource-late",
    "resource-early",
};

void track_out_of_stage_order() {
    const auto track = [](const char* value, int32_t stage) {
        CHECK(sd_dart_exit_track(name(value), free_named, stage));
    };
    track("resource-early", SD_DART_EXIT_STAGE_RESOURCE);
    track("context-early", SD_DART_EXIT_STAGE_CONTEXT);
    track("user-early", SD_DART_EXIT_STAGE_CONTEXT_USER);
    track("resource-late", SD_DART_EXIT_STAGE_RESOURCE);
    track("context-late", SD_DART_EXIT_STAGE_CONTEXT);
    track(kUserLate, SD_DART_EXIT_STAGE_CONTEXT_USER);
}

int test_dispose() {
    CHECK(!sd_dart_exit_track(nullptr, free_named, 0));
    CHECK(!sd_dart_exit_track(name("a"), nullptr, 0));
    CHECK(!sd_dart_exit_track(name("a"), free_named, -1));
    CHECK(!sd_dart_exit_track(name("a"), free_named, SD_DART_EXIT_STAGE_RESOURCE + 1));
    CHECK(sd_dart_exit_tracked_count() == 0);

    char* resource = name("resource");
    char* context  = name("context");
    char* kept     = name("kept");
    CHECK(sd_dart_exit_track(resource, free_named, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(sd_dart_exit_track(context, free_named, SD_DART_EXIT_STAGE_CONTEXT));
    CHECK(sd_dart_exit_track(kept, free_named, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(sd_dart_exit_tracked_count() == 3);

    sd_dart_exit_free(context);
    sd_dart_exit_free(resource);
    sd_dart_exit_free(resource);
    sd_dart_exit_free(nullptr);
    sd_dart_exit_free(name("never-tracked"));
    CHECK(recorded() == std::vector<std::string>({"context", "resource"}));

    CHECK(sd_dart_exit_untrack(kept));
    CHECK(!sd_dart_exit_untrack(kept));
    sd_dart_exit_free(kept);
    CHECK(sd_dart_exit_tracked_count() == 0);

    sd_dart_exit_teardown();
    CHECK(recorded() == std::vector<std::string>({"context", "resource"}));
    return 0;
}

int test_order() {
    track_out_of_stage_order();
    CHECK(sd_dart_exit_tracked_count() == static_cast<int32_t>(kStageOrder.size()));

    sd_dart_exit_teardown();
    CHECK(recorded() == kStageOrder);
    CHECK(sd_dart_exit_tracked_count() == 0);

    // The teardown thread is not blocked, and nothing is freed a second time.
    sd_dart_exit_free(name("context-early"));
    CHECK(!sd_dart_exit_untrack(name("context-early")));
    CHECK(!sd_dart_exit_track(name("late"), free_named, SD_DART_EXIT_STAGE_CONTEXT));
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(context != nullptr);
    CHECK(sd_dart_exit_tracked_count() == 0);
    sd_dart_exit_free(context);
    CHECK(contexts_freed.load() == 0);
    sd_dart_exit_teardown();
    CHECK(recorded() == kStageOrder);
    return 0;
}

int test_repeat() {
    char* object = name("object");
    sd_ctx_params_t params{};
    for (int i = 0; i < 2000; ++i) {
        CHECK(sd_dart_exit_track(object, free_named, SD_DART_EXIT_STAGE_CONTEXT_USER));
        sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
        CHECK(sd_dart_exit_tracked_count() == 2);
        sd_dart_exit_free(context);
        sd_dart_exit_free(object);
        CHECK(sd_dart_exit_tracked_count() == 0);
    }
    CHECK(recorded().size() == 2000);
    CHECK(contexts_freed.load() == 2000);

    // A tracked address that is tracked again holds a new object.
    CHECK(sd_dart_exit_track(object, free_named, SD_DART_EXIT_STAGE_CONTEXT));
    CHECK(sd_dart_exit_track(object, free_named, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(sd_dart_exit_tracked_count() == 1);
    sd_dart_exit_teardown();
    CHECK(recorded().size() == 2001);
    return 0;
}

int test_in_flight() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);

    static std::atomic<bool> in_call{false};
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        in_call.store(true);
        sleep_ms(400);
        record("call-finished");
        sd_dart_exit_call_end();
        returned.store(true);
    }).detach();
    while (!in_call.load()) {
        sleep_ms(1);
    }

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 20000);

    std::vector<std::string> expected = {"call-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);

    // The thread that was in the call never gets back to its caller.
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// Teardown waits for the outermost call of a thread, which stays free to make
// nested calls and to use the registry until that call ends.
int test_nested() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    static char inner[] = "inner";

    static std::atomic<bool> in_call{false};
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        in_call.store(true);
        CHECK(wait_for_teardown());
        sd_dart_exit_call_begin();
        CHECK(sd_dart_exit_track(inner, free_named, SD_DART_EXIT_STAGE_CONTEXT_USER));
        sd_ctx_params_t params{};
        sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
        CHECK(sd_dart_exit_tracked_count() == static_cast<int32_t>(kStageOrder.size()) + 2);
        sd_img_gen_params_t request{};
        CHECK(sd_dart_generate_image(context, &request, nullptr, nullptr));
        sd_dart_exit_free(context);
        sd_dart_exit_free(kUserLate);
        sd_dart_exit_call_end();
        record("call-finished");
        sd_dart_exit_call_end();
        returned.store(true);
    }).detach();
    while (!in_call.load()) {
        sleep_ms(1);
    }

    teardown_requested.store(true);
    sd_dart_exit_teardown();
    std::vector<std::string> expected = {"user-late", "call-finished", "inner"};
    expected.insert(expected.end(), kStageOrder.begin() + 1, kStageOrder.end());
    CHECK(recorded() == expected);
    CHECK(contexts_freed.load() == 1);
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// A thread that left a call in flight gets time to finish what follows it.
int test_settle() {
    track_out_of_stage_order();

    static std::atomic<bool> left_call{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        sd_dart_exit_call_end();
        left_call.store(true);
        sleep_ms(50);
        record("tail-finished");
    }).detach();
    while (!left_call.load()) {
        sleep_ms(1);
    }

    sd_dart_exit_teardown();
    std::vector<std::string> expected = {"tail-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);
    return 0;
}

// A generation that never ends: teardown gives up after the bound for loads
// and generations and frees nothing, also not the objects no call is using.
int test_timeout() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(200, 1000);
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    static sd_ctx_t* idle    = sd_dart_new_sd_ctx(&params);
    const int32_t tracked    = static_cast<int32_t>(kStageOrder.size()) + 2;
    CHECK(sd_dart_exit_tracked_count() == tracked);

    run_generation.store([](sd_ctx_t*) {
        for (;;) {
            sleep_ms(1000);
        }
        return true;
    });
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
    }).detach();
    while (!context->generating.load()) {
        sleep_ms(1);
    }

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    const int64_t waited = elapsed_ms(started);
    CHECK(waited >= 900);
    // Well short of the default for loads and generations, which a setter
    // that dropped `work_wait_ms` would leave in place.
    CHECK(waited < 5000);
    CHECK(recorded().empty());
    CHECK(contexts_freed.load() == 0);
    CHECK(sd_dart_exit_tracked_count() == tracked);
    CHECK(context->cancel_requests.load() > 0);
    CHECK(idle->cancel_requests.load() > 0);

    // Teardown runs once: a second run neither waits nor frees.
    const auto restarted = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(restarted) < 500);
    CHECK(recorded().empty());
    return 0;
}

// A call that is neither a load nor a generation gets the shorter bound.
int test_timeout_plain() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(1000, 30000);
    static std::atomic<bool> in_call{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        in_call.store(true);
        sleep_ms(600000);
    }).detach();
    while (!in_call.load()) {
        sleep_ms(1);
    }

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    const int64_t waited = elapsed_ms(started);
    CHECK(waited >= 900);
    CHECK(waited < 10000);
    CHECK(recorded().empty());
    return 0;
}

// A generation that cannot be cancelled and a load, each outlasting the
// shorter bound: teardown waits for them with the longer one and then frees
// everything.
int test_long_work() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(300, 30000);
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    static std::atomic<int> working{0};
    run_generation.store([](sd_ctx_t*) {
        working.fetch_add(1);
        sleep_ms(1200);
        record("generation-finished");
        return true;
    });
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
    }).detach();
    while (working.load() < 1) {
        sleep_ms(1);
    }
    hold_load.store([] {
        working.fetch_add(1);
        sleep_ms(1500);
        record("load-finished");
        return true;
    });
    std::thread([] {
        sd_ctx_params_t late_params{};
        sd_dart_new_sd_ctx(&late_params);
    }).detach();
    while (working.load() < 2) {
        sleep_ms(1);
    }

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    const int64_t waited = elapsed_ms(started);
    CHECK(waited >= 1000);
    CHECK(waited < 20000);
    std::vector<std::string> expected = {"generation-finished", "load-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);
    CHECK(contexts_freed.load() == 2);
    CHECK(sd_dart_exit_tracked_count() == 0);
    return 0;
}

// The default bound for loads and generations, which this scenario does not
// set: a generation that cannot be cancelled and ends 2.5 s after teardown
// began, later than the default for other calls allows, is waited for, and
// everything is freed.
int test_default_work_wait() {
    track_out_of_stage_order();
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    run_generation.store([](sd_ctx_t*) {
        while (!teardown_requested.load()) {
            sleep_ms(1);
        }
        sleep_ms(2500);
        record("generation-finished");
        return true;
    });
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
    }).detach();
    while (!context->generating.load()) {
        sleep_ms(1);
    }

    teardown_requested.store(true);
    sd_dart_exit_teardown();
    std::vector<std::string> expected = {"generation-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);
    CHECK(contexts_freed.load() == 1);
    CHECK(sd_dart_exit_tracked_count() == 0);
    return 0;
}

// The default bound for other calls, which this scenario does not set: with
// a call that never ends, teardown gives up after 2 s and frees nothing.
int test_default_plain_timeout() {
    track_out_of_stage_order();
    static std::atomic<bool> in_call{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        in_call.store(true);
        sleep_ms(600000);
    }).detach();
    while (!in_call.load()) {
        sleep_ms(1);
    }

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    const int64_t waited = elapsed_ms(started);
    CHECK(waited >= 1800);
    // Far from the default for loads and generations, with room for a runner
    // that stalls.
    CHECK(waited < 6000);
    CHECK(recorded().empty());
    CHECK(sd_dart_exit_tracked_count() == static_cast<int32_t>(kStageOrder.size()));
    return 0;
}

// With contexts tracked and nothing in flight, teardown frees them without
// waiting, however long the bounds are.
int test_idle_no_wait() {
    sd_dart_exit_set_wait_ms(30000, 60000);
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);
    sd_img_gen_params_t request{};
    CHECK(sd_dart_generate_image(context, &request, nullptr, nullptr));
    // Past the time a thread gets to finish what follows its last call.
    sleep_ms(400);

    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 100);
    CHECK(contexts_freed.load() == 2);
    return 0;
}

// A call that never ends does not hold up an exit that has nothing to free:
// what a caller that tracks nothing gets.
int test_idle_wait() {
    std::thread([] { sd_dart_exit_call_begin(); }).join();
    sd_ctx_params_t params{};
    sd_ctx_t* untracked = new_sd_ctx(&params);
    sd_dart_exit_set_wait_ms(30000, 30000);
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 5000);
    CHECK(contexts_freed.load() == 0);
    CHECK(untracked->cancel_requests.load() == 0);
    return 0;
}

// The same once everything that was tracked has been freed: a call that
// created or freed a tracked object counts as one only until it returns.
int test_idle_wait_after_free() {
    sd_ctx_params_t params{};
    sd_dart_exit_free(sd_dart_new_sd_ctx(&params));
    char* object = name("object");
    CHECK(sd_dart_exit_track(object, free_named, SD_DART_EXIT_STAGE_RESOURCE));
    sd_dart_exit_free(object);
    CHECK(contexts_freed.load() == 1);
    CHECK(recorded() == std::vector<std::string>({"object"}));
    CHECK(sd_dart_exit_tracked_count() == 0);

    std::thread([] { sd_dart_exit_call_begin(); }).join();
    sd_dart_exit_set_wait_ms(10000, 10000);
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 5000);
    return 0;
}

// Teardown that runs inside a call in flight, as when a callback of that call
// exits the process, does not wait for the call it is in.
int test_own_call() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);
    sd_dart_exit_call_begin();
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 5000);
    CHECK(recorded() == kStageOrder);
    sd_dart_exit_call_end();
    return 0;
}

// A generation that exits the process from one of its own callbacks is not a
// reason for the longer bound: teardown cannot wait for the call it is in.
int test_own_work_call() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(300, 30000);
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    static std::atomic<bool> in_call{false};
    std::thread([] {
        sd_dart_exit_call_begin();
        in_call.store(true);
        sleep_ms(600000);
    }).detach();
    while (!in_call.load()) {
        sleep_ms(1);
    }
    static std::atomic<int64_t> waited{-1};
    run_generation.store([](sd_ctx_t*) {
        const auto started = std::chrono::steady_clock::now();
        sd_dart_exit_teardown();
        waited.store(elapsed_ms(started));
        return true;
    });
    sd_img_gen_params_t request{};
    CHECK(sd_dart_generate_image(context, &request, nullptr, nullptr));
    CHECK(waited.load() >= 250);
    CHECK(waited.load() < 10000);
    CHECK(recorded().empty());
    return 0;
}

int test_blocked() {
    char* object = name("object");
    CHECK(sd_dart_exit_track(object, free_named, SD_DART_EXIT_STAGE_CONTEXT));
    sd_ctx_params_t params{};
    static sd_ctx_t* raw = new_sd_ctx(&params);
    sd_dart_exit_teardown();
    CHECK(recorded() == std::vector<std::string>({"object"}));

    static std::atomic<int> started{0};
    static std::atomic<int> returned{0};
    int launched    = 0;
    const auto late = [&launched](void (*call)(void*), void* argument) {
        ++launched;
        std::thread([call, argument] {
            started.fetch_add(1);
            call(argument);
            returned.fetch_add(1);
        }).detach();
    };
    late([](void* argument) { sd_dart_exit_free(argument); }, object);
    late([](void* argument) { sd_dart_exit_untrack(argument); }, object);
    late([](void* argument) { sd_dart_exit_track(argument, free_named, SD_DART_EXIT_STAGE_CONTEXT); }, object);
    late([](void*) { sd_dart_exit_call_begin(); }, nullptr);
    late([](void*) {
        sd_ctx_params_t late_params{};
        sd_dart_new_sd_ctx(&late_params);
    },
         nullptr);
    late([](void*) {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(raw, &request, nullptr, nullptr);
    },
         nullptr);

    while (started.load() < launched) {
        sleep_ms(1);
    }
    sleep_ms(300);
    CHECK(returned.load() == 0);
    CHECK(recorded() == std::vector<std::string>({"object"}));
    CHECK(!raw->generating.load());

    // Cancelling is the one call that still returns, without touching anything.
    std::thread([] { sd_dart_cancel_generation(raw, SD_CANCEL_ALL); }).join();
    CHECK(raw->cancel_requests.load() == 0);
    return 0;
}

void expect_freed_at_exit() {
    const std::vector<std::string> expected = kExitFrees ? kStageOrder : std::vector<std::string>();
    if (recorded() != expected) {
        std::fprintf(stderr, "exit teardown freed %zu objects, expected %zu\n", recorded().size(), expected.size());
        std::_Exit(1);
    }
    if (teardown_refuses_this_thread() != kExitRuns) {
        std::fprintf(stderr, "exit %s teardown\n", kExitRuns ? "did not run" : "ran");
        std::_Exit(1);
    }
    const size_t tracked = kExitFrees ? 0 : kStageOrder.size();
    if (sd_dart_exit_tracked_count() != static_cast<int32_t>(tracked)) {
        std::fprintf(stderr, "%d objects are tracked after the exit, expected %zu\n", sd_dart_exit_tracked_count(),
                     tracked);
        std::_Exit(1);
    }
}

int test_exit() {
    // Registered first, so it runs after the teardown registered by tracking.
    CHECK(check_at_exit(expect_freed_at_exit));
    track_out_of_stage_order();
    return kNotChecked;
}

struct LateStatic {
    std::string text = std::string(256, 's');
    ~LateStatic() { alive.store(false); }
    static std::atomic<bool> alive;
};
std::atomic<bool> LateStatic::alive{true};

const LateStatic& late_static() {
    static LateStatic value;
    return value;
}

std::atomic<bool> late_static_read{false};

void free_and_read_late_static(void*) {
    if (!LateStatic::alive.load() || late_static().text != std::string(256, 's')) {
        std::fprintf(stderr, "a static was destroyed before teardown\n");
        std::_Exit(1);
    }
    late_static_read.store(true);
}

void expect_late_static_read() {
    if (late_static_read.load() != kExitFrees) {
        std::fprintf(stderr, "exit teardown did not run before the statics were destroyed\n");
        std::_Exit(1);
    }
    if (kExitRuns && !kExitFrees && !LateStatic::alive.load()) {
        std::fprintf(stderr, "exit destroyed a static of the library\n");
        std::_Exit(1);
    }
}

// A static that is created after the first object was tracked is also
// destroyed only after teardown, and where exit() frees nothing, not at all.
int test_late_static() {
    static char reader[] = "reader";
    CHECK(check_at_exit(expect_late_static_read));
    CHECK(sd_dart_exit_track(reader, free_and_read_late_static, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(late_static().text.size() == 256);
    return kNotChecked;
}

// The context is tracked by the time the creating call returns, freed once
// whoever frees it, and a failed load tracks nothing.
int test_context() {
    sd_ctx_params_t params{};
    fail_load.store(true);
    CHECK(sd_dart_new_sd_ctx(&params) == nullptr);
    CHECK(sd_dart_exit_tracked_count() == 0);
    fail_load.store(false);

    sd_ctx_t* first = sd_dart_new_sd_ctx(&params);
    CHECK(last_load_params.load() == &params);
    sd_ctx_t* second = sd_dart_new_sd_ctx(&params);
    sd_ctx_t* raw    = new_sd_ctx(&params);
    CHECK(first != nullptr && second != nullptr);
    CHECK(sd_dart_exit_tracked_count() == 2);

    sd_img_gen_params_t request{};
    sd_image_t* images = nullptr;
    int count          = 0;
    CHECK(sd_dart_generate_image(first, &request, &images, &count));
    CHECK(last_request.load() == &request && last_images_out.load() == &images);
    CHECK(count == 7);
    CHECK(!first->generating.load());
    run_generation.store([](sd_ctx_t*) { return false; });
    CHECK(!sd_dart_generate_image(first, &request, &images, &count));

    sd_dart_exit_free(first);
    CHECK(contexts_freed.load() == 1);
    CHECK(last_freed.load() == first);
    CHECK(!kSeesFreedBlocks || is_freed(first));
    sd_dart_exit_free(first);
    sd_dart_exit_free(raw);
    CHECK(contexts_freed.load() == 1);
    CHECK(sd_dart_exit_tracked_count() == 1);

    sd_dart_exit_teardown();
    CHECK(contexts_freed.load() == 2);
    CHECK(last_freed.load() == second);
    CHECK(!kSeesFreedBlocks || is_freed(second));
    CHECK(!kSeesFreedBlocks || !is_freed(raw));
    CHECK(sd_dart_exit_tracked_count() == 0);
    return 0;
}

// Cancelling reaches a tracked context and nothing else.
int test_cancel() {
    sd_ctx_params_t params{};
    sd_ctx_t* tracked = sd_dart_new_sd_ctx(&params);
    sd_ctx_t* raw     = new_sd_ctx(&params);
    static sd_ctx_t generic;
    CHECK(sd_dart_exit_track(&generic, [](void*) {}, SD_DART_EXIT_STAGE_CONTEXT));

    sd_dart_cancel_generation(tracked, SD_CANCEL_NEW_LATENTS);
    CHECK(tracked->cancel_requests.load() == 1);
    CHECK(tracked->cancel.load() == SD_CANCEL_NEW_LATENTS);
    sd_dart_cancel_generation(raw, SD_CANCEL_ALL);
    sd_dart_cancel_generation(&generic, SD_CANCEL_ALL);
    sd_dart_cancel_generation(nullptr, SD_CANCEL_ALL);
    CHECK(raw->cancel_requests.load() == 0);
    CHECK(generic.cancel_requests.load() == 0);

    // Not after the context is freed either: the stand-in would write to a
    // freed block, which a sanitizer reports.
    sd_dart_exit_free(tracked);
    sd_dart_cancel_generation(tracked, SD_CANCEL_ALL);
    CHECK(contexts_freed.load() == 1);
    return 0;
}

std::vector<int32_t> tracked_at_free;

void free_and_count_tracked(void*) {
    tracked_at_free.push_back(sd_dart_exit_tracked_count());
}

// A context is freed in the CONTEXT stage: what is tracked in the stage before
// still sees it, what is tracked in the stage after does not.
int test_context_stage() {
    static char user[]     = "user";
    static char resource[] = "resource";
    // Within a stage the latest tracked object goes first, so each of the two
    // would be freed after a context that was tracked in its stage.
    CHECK(sd_dart_exit_track(user, free_and_count_tracked, SD_DART_EXIT_STAGE_CONTEXT_USER));
    sd_ctx_params_t params{};
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);
    CHECK(sd_dart_exit_track(resource, free_and_count_tracked, SD_DART_EXIT_STAGE_RESOURCE));
    sd_dart_exit_teardown();
    CHECK(tracked_at_free == std::vector<int32_t>({2, 0}));
    CHECK(contexts_freed.load() == 1);
    return 0;
}

std::atomic<bool> free_started{false};

// Freeing a context is a call in flight: teardown waits for it, frees nothing
// else meanwhile, and the freeing thread never returns.
int test_free_in_flight() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    hold_free.store([] {
        free_started.store(true);
        CHECK(wait_for_teardown());
        sleep_ms(50);
        CHECK(recorded().empty());
        record("free-finished");
    });
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_dart_exit_free(context);
        returned.store(true);
    }).detach();
    while (!free_started.load()) {
        sleep_ms(1);
    }

    teardown_requested.store(true);
    sd_dart_exit_teardown();
    std::vector<std::string> expected = {"free-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);
    CHECK(contexts_freed.load() == 1);
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// The context that is being freed is the only tracked object, so nothing is
// tracked when teardown begins. Teardown still waits for the free: it needs
// the statics that are destroyed once teardown returns.
int test_last_free_in_flight() {
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    hold_free.store([] {
        free_started.store(true);
        CHECK(wait_for_teardown());
        sleep_ms(50);
        record("free-finished");
    });
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_dart_exit_free(context);
        returned.store(true);
    }).detach();
    while (!free_started.load()) {
        sleep_ms(1);
    }
    CHECK(sd_dart_exit_tracked_count() == 0);

    teardown_requested.store(true);
    sd_dart_exit_teardown();
    CHECK(recorded() == std::vector<std::string>({"free-finished"}));
    CHECK(contexts_freed.load() == 1);
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// A thread loads and frees in a loop while teardown runs at a moment of its
// own. Whichever instruction the thread is at, no load and no free begins
// once teardown has returned, and nothing stays tracked. One run tries one
// moment, so this scenario is run many times.
int test_load_race() {
    std::thread([] {
        sd_ctx_params_t params{};
        for (;;) {
            sd_dart_exit_free(sd_dart_new_sd_ctx(&params));
        }
    }).detach();
    while (loads_begun.load() < 1000) {
        std::this_thread::yield();
    }
    const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    for (volatile int i = 0, spins = static_cast<int>(seed % 20000); i < spins; i = i + 1) {
    }
    sd_dart_exit_teardown();
    teardown_returned.store(true);
    sleep_ms(30);
    CHECK(calls_after_teardown.load() == 0);
    CHECK(sd_dart_exit_tracked_count() == 0);
    return 0;
}

std::atomic<bool> load_started{false};

// A load in flight when teardown begins, with nothing tracked yet: teardown
// waits for it, the context is tracked although teardown has begun, and it is
// freed. The loading thread never returns.
int test_load_in_flight() {
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    hold_load.store([] {
        load_started.store(true);
        CHECK(wait_for_teardown());
        record("load-finished");
        return true;
    });
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_ctx_params_t params{};
        sd_dart_new_sd_ctx(&params);
        returned.store(true);
    }).detach();
    while (!load_started.load()) {
        sleep_ms(1);
    }

    teardown_requested.store(true);
    sd_dart_exit_teardown();
    CHECK(recorded() == std::vector<std::string>({"load-finished"}));
    CHECK(contexts_freed.load() == 1);
    CHECK(sd_dart_exit_tracked_count() == 0);
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// A generation in flight when teardown begins: teardown cancels it, waits for
// it to end and only then frees its context. The stand-in clears the first
// cancellation the way generate_image() clears one that precedes its start.
int test_generate_in_flight() {
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);

    run_generation.store([](sd_ctx_t* generating) {
        CHECK(wait_for_teardown());
        while (generating->cancel.load() != SD_CANCEL_ALL) {
            sleep_ms(1);
        }
        generating->cancel.store(SD_CANCEL_RESET);
        while (generating->cancel.load() != SD_CANCEL_ALL) {
            sleep_ms(1);
        }
        sleep_ms(50);
        CHECK(recorded().empty());
        CHECK(contexts_freed.load() == 0);
        CHECK(!kSeesFreedBlocks || !is_freed(generating));
        record("generation-finished");
        return false;
    });
    static std::atomic<bool> returned{false};
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
        returned.store(true);
    }).detach();
    while (!context->generating.load()) {
        sleep_ms(1);
    }

    teardown_requested.store(true);
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 20000);
    std::vector<std::string> expected = {"generation-finished"};
    expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
    CHECK(recorded() == expected);
    CHECK(contexts_freed.load() == 2);
    CHECK(sd_dart_exit_tracked_count() == 0);
    sleep_ms(200);
    CHECK(!returned.load());
    return 0;
}

// The stand-in for ggml's device registry: a static that exit() destroys,
// as ggml's is, and that a device query reads.
struct DeviceRegistry {
    ggml_backend_device* device = nullptr;
    ~DeviceRegistry() {
        destroyed.store(true);
        if (queries.load() > 0) {
            std::fprintf(stderr, "the device registry was destroyed under a device query\n");
            std::_Exit(1);
        }
    }
    static std::atomic<bool> destroyed;
    // The queries that are reading the registry.
    static std::atomic<int> queries;
};
std::atomic<bool> DeviceRegistry::destroyed{false};
std::atomic<int> DeviceRegistry::queries{0};

DeviceRegistry& device_registry() {
    static DeviceRegistry registry;
    return registry;
}

// What a device query does while it reads the registry; null for nothing.
std::atomic<void (*)()> hold_query{nullptr};
std::atomic<bool> in_query{false};
std::atomic<bool> query_returned{false};

void hold_query_through_teardown() {
    in_query.store(true);
    CHECK(wait_for_teardown());
    CHECK(!DeviceRegistry::destroyed.load());
    sleep_ms(100);
    CHECK(!DeviceRegistry::destroyed.load());
    record("query-finished");
}

// Teardown waits for a device query although nothing is tracked: the query
// reads statics that are destroyed once teardown is over. The process then
// exits, so that the registry's destructor has its say.
int query_in_flight(void (*query)()) {
    // Registered before the registry exists, so it runs after the registry's
    // destructor, and with that after teardown.
    CHECK(check_at_exit([] {
        if (recorded() != std::vector<std::string>({"query-finished"}) || query_returned.load()) {
            std::fprintf(stderr, "exit teardown did not wait for the device query\n");
            std::_Exit(1);
        }
    }));
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    hold_query.store(hold_query_through_teardown);
    static void (*run_query)() = query;
    std::thread([] {
        run_query();
        query_returned.store(true);
    }).detach();
    while (!in_query.load()) {
        sleep_ms(1);
    }
    CHECK(sd_dart_exit_tracked_count() == 0);
    teardown_requested.store(true);
    if (!kExitFrees) {
        sd_dart_exit_teardown();
        CHECK(recorded() == std::vector<std::string>({"query-finished"}));
    }
    return kNotChecked;
}

int test_device_memory_in_flight() {
    return query_in_flight([] {
        sd_dart_gpu_device_memory_t memory;
        sd_dart_gpu_device_memory(SD_DART_GPU_DEFAULT_DEVICE, &memory);
    });
}

int test_device_count_in_flight() {
    return query_in_flight([] { sd_dart_gpu_device_count(); });
}

// A device query that outlasts the bound of a load costs that wait and no
// more: the bound for other calls, far longer here, is not the one applied.
int test_device_query_timeout() {
    sd_dart_exit_set_wait_ms(600000, 300);
    hold_query.store([] {
        in_query.store(true);
        sleep_ms(600000);
    });
    std::thread([] { sd_dart_gpu_device_count(); }).detach();
    while (!in_query.load()) {
        sleep_ms(1);
    }
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    const int64_t waited = elapsed_ms(started);
    CHECK(waited >= 250 && waited < 20000);
    // The exit goes on under the query, as it did before the registry.
    std::_Exit(0);
}

// A device query after a query that returned does not hold up the exit, and
// one that begins after teardown never reaches the registry.
int test_device_query_idle() {
    sd_dart_gpu_device_memory_t memory;
    CHECK(sd_dart_gpu_device_count() == 1);
    CHECK(sd_dart_gpu_device_memory(0, &memory) == SD_DART_GPU_OK);
    sd_dart_exit_set_wait_ms(30000, 30000);
    const auto started = std::chrono::steady_clock::now();
    sd_dart_exit_teardown();
    CHECK(elapsed_ms(started) < 5000);
    static std::atomic<bool> late_query_returned{false};
    std::thread([] {
        sd_dart_gpu_device_count();
        late_query_returned.store(true);
    }).detach();
    sleep_ms(200);
    CHECK(!late_query_returned.load() && DeviceRegistry::queries.load() == 0);
    return 0;
}

// A call that fails leaves the reason upstream logged for it with the thread
// that made the call, once the recorder is enabled.
int test_last_error() {
    sd_ctx_params_t params{};
    sd_img_gen_params_t request{};
    fail_load.store(true);
    CHECK(sd_dart_new_sd_ctx(&params) == nullptr);
    CHECK(last_error().empty());

    sd_dart_log_enable();
    CHECK(sd_dart_new_sd_ctx(&params) == nullptr);
    const std::string reason = "model_loader.cpp:1 - tensor 'x' not in model metadata\n"
                               "diffusion_engine.cpp:2 - new_sd_ctx_t failed";
    CHECK(last_error() == reason);
    CHECK(sd_dart_exit_tracked_count() == 0);
    // Another thread's call has its own.
    std::thread([&] {
        CHECK(last_error().empty());
        fail_load.store(false);
        sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
        CHECK(context != nullptr && last_error().empty());
        sd_dart_exit_free(context);
    }).join();
    CHECK(last_error() == reason);

    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(context != nullptr && last_error().empty());
    run_generation.store([](sd_ctx_t*) { return false; });
    CHECK(!sd_dart_generate_image(context, &request, nullptr, nullptr));
    CHECK(last_error() == "diffusion_engine.cpp:3 - generate_image failed");
    run_generation.store(nullptr);
    CHECK(sd_dart_generate_image(context, &request, nullptr, nullptr));
    CHECK(last_error().empty());
    // A call that is not a load or a generation leaves it alone.
    run_generation.store([](sd_ctx_t*) { return false; });
    CHECK(!sd_dart_generate_image(context, &request, nullptr, nullptr));
    sd_dart_exit_free(context);
    CHECK(last_error() == "diffusion_engine.cpp:3 - generate_image failed");

    // The messages themselves are there to read, the free's among them.
    std::vector<std::string> messages;
    char text[SD_DART_LOG_TEXT_SIZE];
    int32_t level = -1;
    for (uint64_t after = 0; (after = sd_dart_log_read(after, text, sizeof(text), &level, nullptr)) != 0;) {
        CHECK(messages.size() + 1 == after);
        messages.push_back(std::to_string(level) + " " + text);
    }
    CHECK(messages.size() == 6);
    CHECK(messages[0] == "4 model_loader.cpp:1 - tensor 'x' not in model metadata");
    CHECK(messages[2] == "2 diffusion_engine.cpp:4 - free_sd_ctx");
    CHECK(messages[5] == "2 diffusion_engine.cpp:4 - free_sd_ctx");
    sd_dart_exit_teardown();
    return 0;
}

std::atomic<bool> log_thread_started{false};

void expect_teardown_logged() {
    // After teardown, on the thread that ran it: reading still works, and
    // what the free of the tracked context logged during teardown is there.
    bool freed = false;
    char text[SD_DART_LOG_TEXT_SIZE];
    uint64_t after = 0;
    for (uint64_t next = 0; (next = sd_dart_log_read(after, text, sizeof(text), nullptr, nullptr)) != 0;) {
        after = next;
        freed = freed || std::strcmp(text, "diffusion_engine.cpp:4 - free_sd_ctx") == 0;
    }
    if (after == 0 || freed != kExitFrees || (contexts_freed.load() == 1) != kExitFrees) {
        std::fprintf(stderr, "read %llu messages after teardown, the free's %s among them\n",
                     static_cast<unsigned long long>(after), freed ? "is" : "is not");
        std::_Exit(1);
    }
    last_error();
    sd_dart_log_dropped();
}

// The process exits with the recorder enabled, a tracked context that logs
// when teardown frees it, and threads that go on logging and reading through
// teardown and whatever exit() does after it.
int test_log_at_exit() {
    // Registered first, so it runs after the teardown registered by tracking.
    CHECK(check_at_exit(expect_teardown_logged));
    sd_dart_log_enable();
    sd_dart_log_set_level(SD_LOG_DEBUG);
    for (int i = 0; i < 2; ++i) {
        std::thread([] {
            for (uint64_t n = 0;; ++n) {
                // Short, so that the free's message is still in the buffer
                // when the check reads it.
                upstream_log(n % 5 == 0 ? SD_LOG_ERROR : SD_LOG_DEBUG, "w");
                log_thread_started.store(true);
                if (n % 8 == 0) {
                    sleep_ms(1);
                }
            }
        }).detach();
    }
    std::thread([] {
        char text[SD_DART_LOG_TEXT_SIZE];
        for (uint64_t after = 0;;) {
            const uint64_t next = sd_dart_log_read(after, text, sizeof(text), nullptr, nullptr);
            after               = next != 0 ? next : after;
            last_error();
        }
    }).detach();
    while (!log_thread_started.load()) {
        sleep_ms(1);
    }
    sd_ctx_params_t params{};
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);
    return kNotChecked;
}

#if defined(__APPLE__)
std::atomic<bool> exit_call_returned{false};

// A generation in flight when main returns: exit() cancels it and waits for
// it, as teardown called by hand does, and frees the tracked objects.
int test_exit_in_flight() {
    CHECK(check_at_exit([] {
        std::vector<std::string> expected = {"generation-finished"};
        expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
        if (recorded() != expected || contexts_freed.load() != 1 || exit_call_returned.load()) {
            std::fprintf(stderr, "exit did not wait for the generation, or did not free\n");
            std::_Exit(1);
        }
    }));
    track_out_of_stage_order();
    sd_dart_exit_set_wait_ms(30000, 30000);
    start_heartbeat();
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    run_generation.store([](sd_ctx_t* generating) {
        CHECK(wait_for_teardown());
        while (generating->cancel.load() != SD_CANCEL_ALL) {
            sleep_ms(1);
        }
        sleep_ms(50);
        CHECK(recorded().empty() && contexts_freed.load() == 0);
        record("generation-finished");
        return false;
    });
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
        exit_call_returned.store(true);
    }).detach();
    while (!context->generating.load()) {
        sleep_ms(1);
    }
    teardown_requested.store(true);
    return kNotChecked;
}
#endif

#if SD_TEST_EXIT_ON_LINUX
// The host's own exit handler, registered before the library has anything to
// register, so it would run after it. It must not run at all once the
// process exits with a call in flight.
void host_handler_under_call() {
    std::fprintf(stderr, "exit went on to the host's handlers under a call in flight\n");
    std::_Exit(1);
}

const int kExitStatus = 37;

// Runs `begin_call`, which leaves a thread inside a call in flight, in a
// child process and exits there with kExitStatus and text in stdio's buffer.
// The child has to end with that status and that text written, without
// reaching its own exit handler: with a call in flight exit() ends the
// process where the library's handler runs.
int exit_ends_process(void (*begin_call)()) {
    int out[2];
    CHECK(pipe(out) == 0);
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        close(out[0]);
        CHECK(dup2(out[1], STDOUT_FILENO) >= 0);
        CHECK(at_exit(host_handler_under_call));
        begin_call();
        std::printf("buffered");
        std::exit(kExitStatus);
    }
    close(out[1]);
    char text[32] = {};
    size_t length = 0;
    for (ssize_t count = 0; (count = read(out[0], text + length, sizeof(text) - 1 - length)) > 0;) {
        length += static_cast<size_t>(count);
    }
    int status = -1;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == kExitStatus);
    CHECK(std::string(text) == "buffered");
    return 0;
}

std::atomic<bool> call_in_flight{false};

bool stay_in_call() {
    call_in_flight.store(true);
    for (;;) {
        sleep_ms(1000);
    }
}

void wait_for_call_in_flight() {
    while (!call_in_flight.load()) {
        sleep_ms(1);
    }
}

// A generation in flight when the process exits.
int test_exit_ends_process_in_generation() {
    return exit_ends_process([] {
        sd_ctx_params_t params{};
        static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
        run_generation.store([](sd_ctx_t*) { return stay_in_call(); });
        std::thread([] {
            sd_img_gen_params_t request{};
            sd_dart_generate_image(context, &request, nullptr, nullptr);
        }).detach();
        wait_for_call_in_flight();
    });
}

// The first load of the process in flight, with nothing tracked.
int test_exit_ends_process_in_load() {
    return exit_ends_process([] {
        hold_load.store(stay_in_call);
        std::thread([] {
            sd_ctx_params_t params{};
            sd_dart_new_sd_ctx(&params);
        }).detach();
        wait_for_call_in_flight();
    });
}

// The first device query in flight, with nothing tracked.
int test_exit_ends_process_in_query() {
    return exit_ends_process([] {
        hold_query.store([] { stay_in_call(); });
        std::thread([] { sd_dart_gpu_device_count(); }).detach();
        wait_for_call_in_flight();
    });
}

// A call that a native caller marked itself, on an object it tracks.
int test_exit_ends_process_in_marked_call() {
    return exit_ends_process([] {
        CHECK(sd_dart_exit_track(name("object"), free_named, SD_DART_EXIT_STAGE_CONTEXT));
        std::thread([] {
            sd_dart_exit_call_begin();
            stay_in_call();
        }).detach();
        wait_for_call_in_flight();
    });
}

// exit() called from inside a call in flight, as from a callback of that
// call, with no other call in flight: exit() goes on to the host's handlers.
int test_exit_from_own_call() {
    CHECK(at_exit([] { std::_Exit(0); }));
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    run_generation.store([](sd_ctx_t*) -> bool { std::exit(kExitStatus); });
    sd_img_gen_params_t request{};
    sd_dart_generate_image(context, &request, nullptr, nullptr);
    return 1;
}

std::atomic<int> driver_handlers_run{0};

// The stand-in for what a GPU driver registers when a call first opens it.
void driver_exit_handler() {
    if (!teardown_refuses_this_thread()) {
        std::fprintf(stderr, "a driver's exit handler ran before the wait of teardown\n");
        std::_Exit(1);
    }
    driver_handlers_run.fetch_add(1);
}

void expect_driver_handlers(int count) {
    static int expected = count;
    CHECK(check_at_exit([] {
        if (driver_handlers_run.load() != expected) {
            std::fprintf(stderr, "%d of %d driver exit handlers ran\n", driver_handlers_run.load(), expected);
            std::_Exit(1);
        }
    }));
}

// A load that is not the first one opens a driver. exit() runs its handlers
// latest first, and the wait still runs before the driver's.
int test_exit_after_driver_load() {
    expect_driver_handlers(1);
    sd_ctx_params_t params{};
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);
    hold_load.store([] {
        CHECK(at_exit(driver_exit_handler));
        return true;
    });
    CHECK(sd_dart_new_sd_ctx(&params) != nullptr);
    return kNotChecked;
}

// The same for the first device query, with nothing tracked.
int test_exit_after_driver_query() {
    expect_driver_handlers(1);
    hold_query.store([] { CHECK(at_exit(driver_exit_handler)); });
    CHECK(sd_dart_gpu_device_count() == 1);
    CHECK(sd_dart_exit_tracked_count() == 0);
    return kNotChecked;
}

std::chrono::steady_clock::time_point main_returned;

// An exit with a context idle takes no time, also right after a call: with
// nothing freed there is nothing to give a thread time for. It goes on to the
// host's handlers, where this is checked.
int test_exit_idle_after_call() {
    CHECK(check_at_exit([] {
        const int64_t waited = elapsed_ms(main_returned);
        if (waited >= 100 || !teardown_refuses_this_thread() || contexts_freed.load() != 0) {
            std::fprintf(stderr, "an exit with nothing in flight took %lld ms\n", static_cast<long long>(waited));
            std::_Exit(1);
        }
    }));
    sd_ctx_params_t params{};
    sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    sd_img_gen_params_t request{};
    CHECK(sd_dart_generate_image(context, &request, nullptr, nullptr));
    main_returned = std::chrono::steady_clock::now();
    return kNotChecked;
}

std::atomic<bool> host_exiting{false};
std::thread* host_worker = nullptr;

// What a host's own exit handler does when it stops and joins its worker.
// Registered first, so it runs after the wait.
void join_host_worker() {
    host_exiting.store(true);
    host_worker->join();
    if (contexts_freed.load() != 0 || sd_dart_exit_tracked_count() != 1) {
        std::fprintf(stderr, "a call that arrived after the wait of exit() freed or untracked something\n");
        std::_Exit(1);
    }
}

// A worker that the host joins at exit frees its context and makes other
// calls then. Each returns at once and does nothing: the worker is not
// blocked, and nothing reaches upstream or a driver after the wait.
int test_exit_refuses_late_calls() {
    alarm(30);
    CHECK(check_at_exit(join_host_worker));
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    CHECK(sd_dart_gpu_device_count() == 1);
    static char other[] = "other";
    host_worker = new std::thread([] {
        while (!host_exiting.load()) {
            sleep_ms(1);
        }
        const int loads = loads_begun.load();
        sd_ctx_params_t late_params{};
        sd_img_gen_params_t request{};
        sd_dart_gpu_device_memory_t memory;
        CHECK(sd_dart_new_sd_ctx(&late_params) == nullptr && loads_begun.load() == loads);
        CHECK(!sd_dart_generate_image(context, &request, nullptr, nullptr) && !context->generating.load());
        CHECK(last_request.load() == nullptr);
        CHECK(sd_dart_gpu_device_count() == SD_DART_GPU_UNAVAILABLE);
        CHECK(sd_dart_gpu_device_memory(0, &memory) == SD_DART_GPU_UNAVAILABLE);
        CHECK(!sd_dart_exit_track(other, free_named, SD_DART_EXIT_STAGE_RESOURCE));
        CHECK(!sd_dart_exit_untrack(context));
        sd_dart_exit_call_begin();
        sd_dart_exit_call_end();
        sd_dart_exit_free(context);
        sd_dart_cancel_generation(context, SD_CANCEL_ALL);
    });
    return kNotChecked;
}

// A child of fork() that calls exit() while the parent generates exits as any
// process does, its own handlers included: the thread that generates is not
// in the child.
int test_exit_in_fork_child() {
    // The child's own exit handler, older than the library's: registered here
    // and copied by fork(). It runs when the child's exit() goes on.
    CHECK(at_exit([] { std::_Exit(kExitStatus); }));
    sd_ctx_params_t params{};
    static sd_ctx_t* context = sd_dart_new_sd_ctx(&params);
    run_generation.store([](sd_ctx_t*) -> bool {
        for (;;) {
            sleep_ms(1000);
        }
    });
    std::thread([] {
        sd_img_gen_params_t request{};
        sd_dart_generate_image(context, &request, nullptr, nullptr);
    }).detach();
    while (!context->generating.load()) {
        sleep_ms(1);
    }
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        std::exit(0);
    }
    int status = -1;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == kExitStatus);
    std::_Exit(0);
}
#endif

}  // namespace

struct ggml_backend_device {
    const char* name = "MTL0";
};

// Each registry function reads the registry, and none may find it destroyed.
struct RegistryRead {
    RegistryRead() {
        CHECK(!DeviceRegistry::destroyed.load());
        DeviceRegistry::queries.fetch_add(1);
    }
    ~RegistryRead() { DeviceRegistry::queries.fetch_sub(1); }
};

size_t sd_list_devices(char*, size_t) {
    RegistryRead read;
    static ggml_backend_device device;
    device_registry().device = &device;
    if (void (*hold)() = hold_query.load()) {
        hold();
    }
    return 0;
}

size_t ggml_backend_dev_count(void) {
    RegistryRead read;
    return 1;
}

ggml_backend_dev_t ggml_backend_dev_get(size_t) {
    RegistryRead read;
    return device_registry().device;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    RegistryRead read;
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ? device_registry().device : nullptr;
}

const char* ggml_backend_dev_name(ggml_backend_dev_t device) {
    return device->name;
}

const char* ggml_backend_dev_description(ggml_backend_dev_t) {
    return "a stand-in";
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void ggml_backend_dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    RegistryRead read;
    *free  = 1;
    *total = 2;
}

void sd_set_log_callback(sd_log_cb_t callback, void*) {
    log_callback.store(callback);
}

void ggml_log_set(ggml_log_callback, void*) {
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    loads_begun.fetch_add(1);
    calls_after_teardown.fetch_add(teardown_returned.load() ? 1 : 0);
    last_load_params.store(sd_ctx_params);
    if (bool (*hold)() = hold_load.load()) {
        hold();
    }
    if (fail_load.load()) {
        // Upstream's loader logs from its own threads.
        std::thread([] { upstream_log(SD_LOG_ERROR, "model_loader.cpp:1 - tensor 'x' not in model metadata"); }).join();
        upstream_log(SD_LOG_ERROR, "diffusion_engine.cpp:2 - new_sd_ctx_t failed");
        return nullptr;
    }
    return new sd_ctx_t();
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    CHECK(!sd_ctx->generating.load());
    calls_after_teardown.fetch_add(teardown_returned.load() ? 1 : 0);
    if (void (*hold)() = hold_free.load()) {
        hold();
    }
    contexts_freed.fetch_add(1);
    last_freed.store(sd_ctx);
    upstream_log(SD_LOG_INFO, "diffusion_engine.cpp:4 - free_sd_ctx");
    delete sd_ctx;
}

bool generate_image(sd_ctx_t* sd_ctx,
                    const sd_img_gen_params_t* sd_img_gen_params,
                    sd_image_t** images_out,
                    int* num_images_out) {
    last_request.store(sd_img_gen_params);
    last_images_out.store(images_out);
    sd_ctx->cancel.store(SD_CANCEL_RESET);
    sd_ctx->generating.store(true);
    bool ok = true;
    if (bool (*run)(sd_ctx_t*) = run_generation.load()) {
        ok = run(sd_ctx);
    }
    if (!ok) {
        upstream_log(SD_LOG_ERROR, "diffusion_engine.cpp:3 - generate_image failed");
    }
    if (num_images_out != nullptr) {
        *num_images_out = 7;
    }
    sd_ctx->generating.store(false);
    return ok;
}

void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    sd_ctx->cancel.store(mode);
    sd_ctx->cancel_requests.fetch_add(1);
}

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "";
    const struct {
        const char* name;
        int (*run)();
    } scenarios[] = {
        {"dispose", test_dispose},
        {"order", test_order},
        {"repeat", test_repeat},
        {"in-flight", test_in_flight},
        {"nested", test_nested},
        {"settle", test_settle},
        {"timeout", test_timeout},
        {"timeout-plain", test_timeout_plain},
        {"long-work", test_long_work},
        {"default-work-wait", test_default_work_wait},
        {"default-plain-timeout", test_default_plain_timeout},
        {"idle-no-wait", test_idle_no_wait},
        {"idle-wait", test_idle_wait},
        {"idle-wait-after-free", test_idle_wait_after_free},
        {"own-call", test_own_call},
        {"own-work-call", test_own_work_call},
        {"blocked", test_blocked},
        {"exit", test_exit},
        {"late-static", test_late_static},
        {"context", test_context},
        {"context-stage", test_context_stage},
        {"free-in-flight", test_free_in_flight},
        {"last-free-in-flight", test_last_free_in_flight},
        {"load-race", test_load_race},
        {"cancel", test_cancel},
        {"load-in-flight", test_load_in_flight},
        {"generate-in-flight", test_generate_in_flight},
        {"device-memory-in-flight", test_device_memory_in_flight},
        {"device-count-in-flight", test_device_count_in_flight},
        {"device-query-timeout", test_device_query_timeout},
        {"device-query-idle", test_device_query_idle},
        {"last-error", test_last_error},
        {"log-at-exit", test_log_at_exit},
#if defined(__APPLE__)
        {"exit-in-flight", test_exit_in_flight},
#endif
#if SD_TEST_EXIT_ON_LINUX
        {"exit-ends-process-in-generation", test_exit_ends_process_in_generation},
        {"exit-ends-process-in-load", test_exit_ends_process_in_load},
        {"exit-ends-process-in-query", test_exit_ends_process_in_query},
        {"exit-ends-process-in-marked-call", test_exit_ends_process_in_marked_call},
        {"exit-from-own-call", test_exit_from_own_call},
        {"exit-after-driver-load", test_exit_after_driver_load},
        {"exit-after-driver-query", test_exit_after_driver_query},
        {"exit-in-fork-child", test_exit_in_fork_child},
        {"exit-idle-after-call", test_exit_idle_after_call},
        {"exit-refuses-late-calls", test_exit_refuses_late_calls},
#endif
    };
    for (const auto& candidate : scenarios) {
        if (scenario == candidate.name) {
            return candidate.run();
        }
    }
    if (scenario == "list") {
        for (const auto& candidate : scenarios) {
            std::printf("%s\n", candidate.name);
        }
        return 0;
    }
    std::fprintf(stderr, "usage: %s <scenario>|list\n", argv[0]);
    return 2;
}

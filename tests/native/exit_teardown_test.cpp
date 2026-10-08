// Exercises src/sd_dart_exit.cpp against stand-ins for the upstream functions
// it wraps, so it needs no model and runs under sanitizers.
//
// Each scenario runs in its own process: teardown runs once per process and
// leaves the registry unusable from other threads.

#include "sd_dart_wrapper.h"

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

#if defined(__APPLE__)
const bool kTeardownRunsAtExit = true;
#else
const bool kTeardownRunsAtExit = false;
#endif

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
    const std::vector<std::string> expected = kTeardownRunsAtExit ? kStageOrder : std::vector<std::string>();
    if (recorded() != expected) {
        std::fprintf(stderr, "exit teardown freed %zu objects, expected %zu\n", recorded().size(), expected.size());
        std::_Exit(1);
    }
}

int test_exit() {
    // Registered first, so it runs after the teardown registered by tracking.
    CHECK(atexit(expect_freed_at_exit) == 0);
    track_out_of_stage_order();
    return 0;
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
    if (late_static_read.load() != kTeardownRunsAtExit) {
        std::fprintf(stderr, "exit teardown did not run before the statics were destroyed\n");
        std::_Exit(1);
    }
}

// A static that is created after the first object was tracked is also
// destroyed only after teardown.
int test_late_static() {
    static char reader[] = "reader";
    CHECK(atexit(expect_late_static_read) == 0);
    CHECK(sd_dart_exit_track(reader, free_and_read_late_static, SD_DART_EXIT_STAGE_RESOURCE));
    CHECK(late_static().text.size() == 256);
    return 0;
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
    if (after == 0 || freed != kTeardownRunsAtExit || (contexts_freed.load() == 1) != kTeardownRunsAtExit) {
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
    CHECK(atexit(expect_teardown_logged) == 0);
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
    return 0;
}

}  // namespace

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
        {"last-error", test_last_error},
        {"log-at-exit", test_log_at_exit},
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

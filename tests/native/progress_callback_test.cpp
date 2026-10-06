// Exercises src/sd_dart_wrapper.cpp against a stand-in for upstream's
// sd_set_progress_callback(), so it needs no model and runs under sanitizers.

#include "sd_dart_wrapper.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace {

sd_progress_cb_t upstream_callback = nullptr;
void* upstream_data                = nullptr;
int upstream_registrations         = 0;

// What upstream does for each progress step.
void report(int step, int steps, float time) {
    CHECK(upstream_callback != nullptr);
    upstream_callback(step, steps, time, upstream_data);
}

struct Recorder {
    int calls  = 0;
    int step   = 0;
    int steps  = 0;
    float time = 0;
};

void record(int step, int steps, float time, void* data) {
    auto* recorder = static_cast<Recorder*>(data);
    recorder->calls++;
    recorder->step  = step;
    recorder->steps = steps;
    recorder->time  = time;
}

// A second callback with its own address and body, so no linker folds the two.
void record_other(int step, int steps, float time, void* data) {
    record(step, steps, -time, data);
}

void* token(sd_progress_cb_t callback) {
    return reinterpret_cast<void*>(callback);
}

void test_clear_before_any_callback_is_set() {
    sd_dart_clear_progress_callback(nullptr);
    sd_dart_clear_progress_callback(token(record));
    CHECK(upstream_registrations == 0);
}

void test_forwards_to_the_current_callback() {
    Recorder first;
    sd_dart_set_progress_callback(record, &first);
    CHECK(upstream_registrations == 1);
    report(3, 20, 0.5f);
    CHECK(first.calls == 1 && first.step == 3 && first.steps == 20 && first.time == 0.5f);

    Recorder second;
    sd_dart_set_progress_callback(record_other, &second);
    report(4, 20, 0.25f);
    CHECK(first.calls == 1);
    CHECK(second.calls == 1 && second.step == 4);
    CHECK(upstream_registrations == 1);
}

void test_null_callback_discards_progress_without_unregistering() {
    Recorder recorder;
    sd_dart_set_progress_callback(record, &recorder);
    sd_dart_set_progress_callback(nullptr, nullptr);
    report(1, 2, 0.0f);
    CHECK(recorder.calls == 0);
    // Upstream prints progress bars only while its callback is null.
    CHECK(upstream_callback != nullptr);
    CHECK(upstream_registrations == 1);
}

void test_clear_matches_the_callback_address() {
    Recorder recorder;
    sd_dart_set_progress_callback(record, &recorder);

    sd_dart_clear_progress_callback(token(record_other));
    report(1, 2, 0.0f);
    CHECK(recorder.calls == 1);

    sd_dart_clear_progress_callback(token(record));
    sd_dart_clear_progress_callback(token(record));
    report(2, 2, 0.0f);
    CHECK(recorder.calls == 1);

    sd_dart_set_progress_callback(record, &recorder);
    sd_dart_clear_progress_callback(nullptr);
    sd_dart_clear_progress_callback(nullptr);
    report(2, 2, 0.0f);
    CHECK(recorder.calls == 1);
    CHECK(upstream_callback != nullptr);
}

void clear_itself(int, int, float, void* data) {
    sd_dart_clear_progress_callback(token(clear_itself));
    ++*static_cast<int*>(data);
}

void test_callback_can_clear_itself() {
    int calls = 0;
    sd_dart_set_progress_callback(clear_itself, &calls);
    report(1, 2, 0.0f);
    report(2, 2, 0.0f);
    CHECK(calls == 1);
}

// State a callback's owner tears down once the clear has returned. `calls` is
// deliberately unsynchronized: a callback that is still running, or that
// starts, after the clear returns races with the owner's write, which
// ThreadSanitizer reports.
struct Owner {
    std::atomic<bool> revoked{false};
    int calls = 0;
};

std::atomic<long> forwarded{0};
std::atomic<long> calls_after_revocation{0};

void count(int, int, float, void* data) {
    auto* owner = static_cast<Owner*>(data);
    if (owner->revoked.load()) {
        calls_after_revocation++;
    }
    owner->calls++;
    forwarded++;
    for (volatile int spin = 0; spin < 64; spin = spin + 1) {
    }
    if (owner->revoked.load()) {
        calls_after_revocation++;
    }
}

void test_no_call_runs_after_a_concurrent_clear_returns() {
    const int rounds = 20000;
    sd_dart_set_progress_callback(nullptr, nullptr);
    forwarded              = 0;
    calls_after_revocation = 0;

    std::atomic<bool> stop{false};
    std::thread reporter([&stop] {
        while (!stop.load()) {
            report(1, 2, 0.0f);
            // Without a gap, a fair mutex such as macOS's turns every round
            // into kernel handoffs and the test takes a minute.
            std::this_thread::yield();
        }
    });

    std::vector<std::unique_ptr<Owner>> owners;
    owners.reserve(rounds);
    for (int round = 0; round < rounds; round++) {
        owners.push_back(std::make_unique<Owner>());
        Owner* owner = owners.back().get();
        sd_dart_set_progress_callback(count, owner);
        if (round % 8 == 0) {
            const long seen = forwarded.load();
            while (forwarded.load() == seen) {
                std::this_thread::yield();
            }
        }
        switch (round % 3) {
            case 0:
                sd_dart_clear_progress_callback(token(count));
                break;
            case 1:
                sd_dart_clear_progress_callback(nullptr);
                break;
            default:
                sd_dart_set_progress_callback(nullptr, nullptr);
                break;
        }
        owner->revoked.store(true);
        owner->calls = -1;
    }

    stop.store(true);
    reporter.join();
    CHECK(forwarded.load() >= rounds / 8);
    CHECK(calls_after_revocation.load() == 0);
}

}  // namespace

extern "C" void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    upstream_callback = cb;
    upstream_data     = data;
    upstream_registrations++;
}

int main() {
    test_clear_before_any_callback_is_set();
    test_forwards_to_the_current_callback();
    test_null_callback_discards_progress_without_unregistering();
    test_clear_matches_the_callback_address();
    test_callback_can_clear_itself();
    test_no_call_runs_after_a_concurrent_clear_returns();
    std::puts("progress callback tests passed");
    return 0;
}

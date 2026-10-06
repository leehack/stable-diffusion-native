#include "sd_dart_wrapper.h"

#include <atomic>
#include <thread>
#include <type_traits>

namespace {

struct Report {
    std::atomic<int32_t> step{0};
    std::atomic<int32_t> steps{0};
    std::atomic<float> time{0.0f};
};

// Report number n is written to reports[n & 1] and then published by storing
// n in `sequence`, so the published report is never the one being written. A
// reader that sees the same `sequence` before and after copying a report got
// all of it, and retries only when a newer one was published meanwhile.
struct Recorder {
    std::atomic<uint64_t> sequence{0};
    Report reports[2];
    std::atomic_flag writing = ATOMIC_FLAG_INIT;
    std::atomic<bool> registered{false};
};

// Never destroyed: a thread can still report progress after exit() has run
// static destructors.
static_assert(std::is_trivially_destructible<Recorder>::value, "Recorder must outlive exit()");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "reading must not take a lock");
Recorder recorder;

void record_progress(int step, int steps, float time, void*) {
    // Reporting threads exclude each other; readers never take this flag.
    while (recorder.writing.test_and_set()) {
        std::this_thread::yield();
    }
    const uint64_t next = recorder.sequence.load() + 1;
    Report& report      = recorder.reports[next & 1];
    report.step.store(step);
    report.steps.store(steps);
    report.time.store(time);
    recorder.sequence.store(next);
    recorder.writing.clear();
}

}  // namespace

void sd_dart_progress_enable(void) {
    // Upstream keeps its callback in an unsynchronized global, so it is
    // written once rather than on every call.
    if (!recorder.registered.exchange(true)) {
        sd_set_progress_callback(record_progress, nullptr);
    }
}

void sd_dart_progress_read(sd_dart_progress_t* progress) {
    for (;;) {
        const uint64_t sequence = recorder.sequence.load();
        const Report& report    = recorder.reports[sequence & 1];
        progress->step          = report.step.load();
        progress->steps         = report.steps.load();
        progress->time          = report.time.load();
        if (recorder.sequence.load() == sequence) {
            progress->sequence = sequence;
            return;
        }
    }
}

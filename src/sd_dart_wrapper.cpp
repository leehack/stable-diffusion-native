#include "sd_dart_wrapper.h"

#include <atomic>
#include <thread>
#include <type_traits>

namespace {

// A power of two. Report n lives in reports[n % kSlots] until report
// n + kSlots replaces it.
constexpr uint64_t kSlots = 4096;
// Report n is safe to copy while the newest report is at most n + kKept - 1:
// the reporter starts on report n + kSlots only after publishing the one
// before it.
constexpr uint64_t kKept = kSlots - 1;
// How often one read starts over because the reporter overtook it.
constexpr int kRestarts = 4;

enum Registration : int { kUnregistered, kRegistering, kRegistered };

struct Report {
    std::atomic<int32_t> step{0};
    std::atomic<int32_t> steps{0};
    std::atomic<float> time{0.0f};
};

struct Recorder {
    // The newest report whose slot is completely written.
    std::atomic<uint64_t> sequence{0};
    Report reports[kSlots];
    std::atomic_flag writing = ATOMIC_FLAG_INIT;
    std::atomic<int> registration{kUnregistered};
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
    Report& report      = recorder.reports[next % kSlots];
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
    int expected = kUnregistered;
    if (recorder.registration.compare_exchange_strong(expected, kRegistering)) {
        sd_set_progress_callback(record_progress, nullptr);
        recorder.registration.store(kRegistered);
        return;
    }
    while (recorder.registration.load() != kRegistered) {
        std::this_thread::yield();
    }
}

size_t sd_dart_progress_read(uint64_t after,
                             sd_dart_progress_t* reports,
                             size_t capacity,
                             uint64_t* latest) {
    uint64_t newest = recorder.sequence.load();
    uint64_t next   = after + 1;
    size_t count    = 0;
    int restarts    = 0;
    while (after < newest && count < capacity && next <= newest && restarts < kRestarts) {
        if (count == 0) {
            const uint64_t oldest = newest < kKept ? 1 : newest - kKept + 1;
            if (next < oldest) {
                next = oldest;
            }
        }
        sd_dart_progress_t& copy = reports[count];
        const Report& report     = recorder.reports[next % kSlots];
        copy.sequence            = next;
        copy.step                = report.step.load();
        copy.steps               = report.steps.load();
        copy.time                = report.time.load();
        newest                   = recorder.sequence.load();
        if (newest - next >= kKept) {
            // The slot may have been rewritten during the copy. Reports
            // already copied stay; a read that has none starts over.
            if (count > 0) {
                break;
            }
            restarts++;
            continue;
        }
        count++;
        next++;
    }
    if (latest != nullptr) {
        *latest = newest;
    }
    return count;
}

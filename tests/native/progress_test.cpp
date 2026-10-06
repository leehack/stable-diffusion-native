// Exercises src/sd_dart_wrapper.cpp against a stand-in for upstream's
// sd_set_progress_callback(), so it needs no model and runs under sanitizers.
//
// With the argument `exit` it only checks that reporting and reading stay
// safe while exit() runs.

#include "sd_dart_wrapper.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

// The number of recent reports the header promises to keep.
const uint64_t kKept = 4095;

std::atomic<sd_progress_cb_t> upstream_callback{nullptr};
std::atomic<int> upstream_registrations{0};

// What upstream does for each progress step.
void report(int step, int steps, float time) {
    upstream_callback.load()(step, steps, time, nullptr);
}

// A report whose three arguments all follow from `n`, so a reader can tell a
// whole report from a mix of two.
void report_numbered(uint64_t n) {
    const int step = static_cast<int>(n & 0x3fffffff);
    report(step, step ^ 0x2aaaaaaa, static_cast<float>(step & 0xffff));
}

bool is_numbered(const sd_dart_progress_t& progress) {
    return progress.steps == (progress.step ^ 0x2aaaaaaa) &&
           progress.time == static_cast<float>(progress.step & 0xffff);
}

bool is_number(const sd_dart_progress_t& progress, uint64_t n) {
    return is_numbered(progress) && progress.step == static_cast<int>(n & 0x3fffffff);
}

uint64_t latest_sequence() {
    uint64_t latest = ~0ull;
    CHECK(sd_dart_progress_read(0, nullptr, 0, &latest) == 0);
    return latest;
}

// One read into a buffer that starts out as garbage, with the checks every
// read must pass. Returns the reports and sets `latest`.
std::vector<sd_dart_progress_t> read(uint64_t after, size_t capacity, uint64_t* latest) {
    std::vector<sd_dart_progress_t> reports(capacity);
    if (capacity > 0) {
        std::memset(reports.data(), 0xff, capacity * sizeof(sd_dart_progress_t));
    }
    const size_t count = sd_dart_progress_read(after, reports.data(), capacity, latest);
    CHECK(count <= capacity);
    reports.resize(count);
    for (size_t i = 0; i < count; i++) {
        CHECK(reports[i].sequence == reports[0].sequence + i);
        CHECK(reports[i].sequence > after && reports[i].sequence <= *latest);
    }
    // Reports are skipped only when more than the library keeps were recorded.
    CHECK(count == 0 || reports[0].sequence == after + 1 || *latest - after > kKept);
    return reports;
}

std::vector<sd_dart_progress_t> read_all(uint64_t after) {
    std::vector<sd_dart_progress_t> all;
    uint64_t latest = 0;
    int empty_reads = 0;
    do {
        const std::vector<sd_dart_progress_t> reports = read(after, 100, &latest);
        for (const sd_dart_progress_t& progress : reports) {
            all.push_back(progress);
            after = progress.sequence;
        }
        // An empty read with reports left means the reporter overtook the
        // reader, which no caller of this function lets happen repeatedly.
        empty_reads = reports.empty() ? empty_reads + 1 : 0;
        CHECK(empty_reads < 100);
    } while (after < latest);
    return all;
}

void test_reads_nothing_before_it_is_enabled() {
    uint64_t latest = 7;
    CHECK(read(0, 8, &latest).empty() && latest == 0);
    CHECK(upstream_registrations == 0);
}

void test_enable_returns_registered_and_registers_once() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++) {
        threads.emplace_back([] {
            sd_dart_progress_enable();
            // Upstream prints progress bars only while its callback is null.
            CHECK(upstream_callback.load() != nullptr);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    sd_dart_progress_enable();
    CHECK(upstream_registrations == 1);
    CHECK(latest_sequence() == 0);
}

void test_reads_reports_in_order() {
    report(3, 20, 0.5f);
    report(4, 20, 0.25f);
    report(5, 21, 0.125f);

    uint64_t latest                              = 0;
    const std::vector<sd_dart_progress_t> all = read(0, 8, &latest);
    CHECK(latest == 3 && all.size() == 3);
    CHECK(all[0].sequence == 1 && all[0].step == 3 && all[0].steps == 20 && all[0].time == 0.5f);
    CHECK(all[1].sequence == 2 && all[1].step == 4 && all[1].steps == 20 && all[1].time == 0.25f);
    CHECK(all[2].sequence == 3 && all[2].step == 5 && all[2].steps == 21 && all[2].time == 0.125f);

    const std::vector<sd_dart_progress_t> first_two = read(0, 2, &latest);
    CHECK(latest == 3 && first_two.size() == 2 && first_two[1].sequence == 2);
    const std::vector<sd_dart_progress_t> last = read(2, 8, &latest);
    CHECK(last.size() == 1 && last[0].sequence == 3 && last[0].step == 5);

    CHECK(read(3, 8, &latest).empty() && latest == 3);
    CHECK(read(99, 8, &latest).empty() && latest == 3);
    CHECK(read(~0ull, 8, &latest).empty() && latest == 3);
    CHECK(sd_dart_progress_read(0, nullptr, 0, nullptr) == 0);
}

void test_keeps_the_promised_number_of_reports() {
    uint64_t base = latest_sequence();
    for (uint64_t n = 1; n <= kKept; n++) {
        report_numbered(n);
    }
    std::vector<sd_dart_progress_t> all = read_all(base);
    CHECK(all.size() == kKept && all.front().sequence == base + 1);

    report_numbered(kKept + 1);
    all = read_all(base);
    CHECK(all.size() == kKept && all.front().sequence == base + 2);
    for (const sd_dart_progress_t& progress : all) {
        CHECK(is_number(progress, progress.sequence - base));
    }

    // Several times around the ring.
    for (uint64_t n = kKept + 2; n <= 5 * kKept + 123; n++) {
        report_numbered(n);
    }
    all = read_all(base);
    CHECK(all.size() == kKept && all.back().sequence == base + 5 * kKept + 123);
    for (const sd_dart_progress_t& progress : all) {
        CHECK(is_number(progress, progress.sequence - base));
    }
}

void test_a_reader_that_keeps_up_gets_every_report() {
    const uint64_t base = latest_sequence();
    uint64_t reported   = 0;
    uint64_t after      = base;
    for (uint64_t burst = 1; reported < 60000; burst = burst * 7 % kKept + 1) {
        for (uint64_t i = 0; i < burst; i++) {
            report_numbered(++reported);
        }
        for (const sd_dart_progress_t& progress : read_all(after)) {
            CHECK(progress.sequence == after + 1);
            CHECK(is_number(progress, progress.sequence - base));
            after = progress.sequence;
        }
        CHECK(after == base + reported);
    }
}

// One reporter numbers its reports from the sequence it starts at, so every
// report read must be the one that belongs to its sequence. One reader
// follows the stream; the others keep asking for the oldest reports, the ones
// the reporter is about to replace.
void test_readers_get_the_report_of_each_sequence() {
    const uint64_t reports = 600000;
    const uint64_t base    = latest_sequence();
    std::atomic<bool> done{false};
    std::atomic<long> read_during_reports{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; i++) {
        readers.emplace_back([&, i] {
            uint64_t after  = base;
            uint64_t latest = base;
            while (!done.load()) {
                if (i > 0) {
                    after = latest - base > kKept + 2 ? latest - kKept - 2 : base;
                }
                for (const sd_dart_progress_t& progress : read(after, 64, &latest)) {
                    CHECK(is_number(progress, progress.sequence - base));
                    after = progress.sequence;
                    if (progress.sequence < base + reports) {
                        read_during_reports++;
                    }
                }
            }
        });
    }
    for (uint64_t n = 1; n <= reports; n++) {
        report_numbered(n);
    }
    done.store(true);
    for (std::thread& reader : readers) {
        reader.join();
    }
    CHECK(read_during_reports.load() > 0);
    CHECK(latest_sequence() == base + reports);
}

void test_concurrent_reporters_lose_no_report() {
    const int reporters    = 4;
    const uint64_t reports = 150000;
    const uint64_t base    = latest_sequence();
    std::atomic<bool> done{false};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; i++) {
        readers.emplace_back([&] {
            uint64_t after  = base;
            uint64_t latest = base;
            while (!done.load()) {
                for (const sd_dart_progress_t& progress : read(after, 64, &latest)) {
                    CHECK(is_numbered(progress));
                    after = progress.sequence;
                }
            }
        });
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < reporters; i++) {
        threads.emplace_back([i] {
            for (uint64_t n = 0; n < reports; n++) {
                report_numbered(n * 7 + static_cast<uint64_t>(i));
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    done.store(true);
    for (std::thread& reader : readers) {
        reader.join();
    }
    CHECK(latest_sequence() == base + static_cast<uint64_t>(reporters) * reports);
    for (const sd_dart_progress_t& progress : read_all(base)) {
        CHECK(is_numbered(progress));
    }
}

int report_until_exit() {
    sd_dart_progress_enable();
    for (int i = 0; i < 4; i++) {
        std::thread([] {
            for (uint64_t n = 0;; n++) {
                report_numbered(n);
            }
        }).detach();
    }
    std::thread([] {
        uint64_t after = 0;
        for (;;) {
            uint64_t latest = 0;
            for (const sd_dart_progress_t& progress : read(after, 64, &latest)) {
                CHECK(is_numbered(progress));
                after = progress.sequence;
            }
        }
    }).detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Runs static destructors while the threads above keep going.
    std::exit(0);
}

}  // namespace

extern "C" void sd_set_progress_callback(sd_progress_cb_t cb, void*) {
    upstream_registrations++;
    // Long enough for a second caller of sd_dart_progress_enable() to return
    // early if it did not wait for this one.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    upstream_callback.store(cb);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "exit") == 0) {
        return report_until_exit();
    }
    test_reads_nothing_before_it_is_enabled();
    test_enable_returns_registered_and_registers_once();
    test_reads_reports_in_order();
    test_keeps_the_promised_number_of_reports();
    test_a_reader_that_keeps_up_gets_every_report();
    test_readers_get_the_report_of_each_sequence();
    test_concurrent_reporters_lose_no_report();
    std::puts("progress tests passed");
    return 0;
}

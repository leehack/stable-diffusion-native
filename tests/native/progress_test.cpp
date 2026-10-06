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

sd_progress_cb_t upstream_callback = nullptr;
void* upstream_data                = nullptr;
std::atomic<int> upstream_registrations{0};

// What upstream does for each progress step.
void report(int step, int steps, float time) {
    upstream_callback(step, steps, time, upstream_data);
}

sd_dart_progress_t read() {
    sd_dart_progress_t progress;
    std::memset(&progress, 0xff, sizeof(progress));
    sd_dart_progress_read(&progress);
    return progress;
}

// A report whose three arguments all follow from `n`, so a reader can tell a
// whole report from a mix of two.
void report_numbered(uint32_t n) {
    const int step = static_cast<int>(n & 0x3fffffff);
    report(step, step ^ 0x2aaaaaaa, static_cast<float>(step & 0xffff));
}

bool is_numbered(const sd_dart_progress_t& progress) {
    return progress.steps == (progress.step ^ 0x2aaaaaaa) &&
           progress.time == static_cast<float>(progress.step & 0xffff);
}

void test_reads_nothing_before_it_is_enabled() {
    const sd_dart_progress_t progress = read();
    CHECK(progress.sequence == 0 && progress.step == 0 && progress.steps == 0 && progress.time == 0.0f);
    CHECK(upstream_registrations == 0);
}

void test_enable_registers_with_upstream_once() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++) {
        threads.emplace_back(sd_dart_progress_enable);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    sd_dart_progress_enable();
    CHECK(upstream_registrations == 1);
    // Upstream prints progress bars only while its callback is null.
    CHECK(upstream_callback != nullptr);
    CHECK(read().sequence == 0);
}

void test_reads_the_latest_report() {
    report(3, 20, 0.5f);
    sd_dart_progress_t progress = read();
    CHECK(progress.sequence == 1 && progress.step == 3 && progress.steps == 20 && progress.time == 0.5f);

    report(4, 20, 0.25f);
    report(5, 21, 0.125f);
    progress = read();
    CHECK(progress.sequence == 3 && progress.step == 5 && progress.steps == 21 && progress.time == 0.125f);
    CHECK(read().sequence == 3);
}

// One reporter numbers its reports from the sequence it starts at, so every
// read must return the report that belongs to the sequence it returns.
void test_readers_get_whole_reports_from_one_reporter() {
    const uint32_t reports = 400000;
    const uint64_t base    = read().sequence;
    std::atomic<bool> done{false};
    std::atomic<long> reads_during_reports{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 3; i++) {
        readers.emplace_back([&] {
            uint64_t last = base;
            while (!done.load()) {
                const sd_dart_progress_t progress = read();
                CHECK(progress.sequence >= last);
                if (progress.sequence > base) {
                    CHECK(is_numbered(progress));
                    CHECK(static_cast<uint64_t>(progress.step) == progress.sequence - base);
                }
                if (progress.sequence > base && progress.sequence < base + reports) {
                    reads_during_reports++;
                }
                last = progress.sequence;
            }
        });
    }
    for (uint32_t n = 1; n <= reports; n++) {
        report_numbered(n);
    }
    done.store(true);
    for (std::thread& reader : readers) {
        reader.join();
    }
    CHECK(reads_during_reports.load() > 0);
    const sd_dart_progress_t progress = read();
    CHECK(progress.sequence == base + reports && static_cast<uint32_t>(progress.step) == reports);
}

void test_concurrent_reporters_lose_no_report() {
    const int reporters    = 4;
    const uint32_t reports = 100000;
    const uint64_t base    = read().sequence;
    std::atomic<bool> done{false};

    std::thread reader([&] {
        uint64_t last = base;
        while (!done.load()) {
            const sd_dart_progress_t progress = read();
            CHECK(progress.sequence >= last);
            CHECK(progress.sequence == base || is_numbered(progress));
            last = progress.sequence;
        }
    });
    std::vector<std::thread> threads;
    for (int i = 0; i < reporters; i++) {
        threads.emplace_back([i] {
            for (uint32_t n = 0; n < reports; n++) {
                report_numbered(n * 7 + static_cast<uint32_t>(i));
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    done.store(true);
    reader.join();
    const sd_dart_progress_t progress = read();
    CHECK(progress.sequence == base + static_cast<uint64_t>(reporters) * reports);
    CHECK(is_numbered(progress));
}

int report_until_exit() {
    sd_dart_progress_enable();
    for (int i = 0; i < 4; i++) {
        std::thread([] {
            for (uint32_t n = 0;; n++) {
                report_numbered(n);
            }
        }).detach();
    }
    std::thread([] {
        for (;;) {
            const sd_dart_progress_t progress = read();
            CHECK(progress.sequence == 0 || is_numbered(progress));
        }
    }).detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Runs static destructors while the threads above keep going.
    std::exit(0);
}

}  // namespace

extern "C" void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    upstream_callback = cb;
    upstream_data     = data;
    upstream_registrations++;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "exit") == 0) {
        return report_until_exit();
    }
    test_reads_nothing_before_it_is_enabled();
    test_enable_registers_with_upstream_once();
    test_reads_the_latest_report();
    test_readers_get_whole_reports_from_one_reporter();
    test_concurrent_reporters_lose_no_report();
    std::puts("progress tests passed");
    return 0;
}

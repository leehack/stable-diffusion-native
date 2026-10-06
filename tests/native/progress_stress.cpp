// Native side of tests/dart/progress_stress.dart, linked with
// src/sd_dart_wrapper.cpp: a stand-in for upstream's progress reporting, and
// the two designs the wrapper replaces, kept as controls.

#include "stable-diffusion.h"

#include <chrono>
#include <mutex>
#include <thread>

#define STRESS_API extern "C" __attribute__((visibility("default")))

namespace {

// As in upstream's util.cpp: an unsynchronized global, read on every report.
sd_progress_cb_t upstream_callback = nullptr;
void* upstream_data                = nullptr;

struct LockedRoute {
    std::recursive_mutex mutex;
    sd_progress_cb_t callback = nullptr;
    void* data                = nullptr;
};

LockedRoute& locked_route() {
    static LockedRoute* route = new LockedRoute;
    return *route;
}

void forward_locked(int step, int steps, float time, void*) {
    LockedRoute& route = locked_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    if (route.callback != nullptr) {
        route.callback(step, steps, time, route.data);
    }
}

}  // namespace

STRESS_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    upstream_callback = cb;
    upstream_data     = data;
}

// Control: a forwarder that holds a lock while it calls the callback, so that
// a one-argument clear can wait for a call in flight. Used with a Dart
// callback and the clear as a NativeFinalizer, it can deadlock VM shutdown.
STRESS_API void control_locked_set_progress_callback(sd_progress_cb_t callback, void* data) {
    LockedRoute& route = locked_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    route.callback = callback;
    route.data     = data;
    sd_set_progress_callback(forward_locked, nullptr);
}

STRESS_API void control_locked_clear_progress_callback(void* callback) {
    LockedRoute& route = locked_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    if (callback == nullptr || callback == reinterpret_cast<void*>(route.callback)) {
        route.callback = nullptr;
        route.data     = nullptr;
    }
}

// Plays new_sd_ctx() or generate_image(): blocks for `milliseconds`, reporting
// progress as upstream does with `gap_microseconds` between reports. Report n
// is (n, n + 1, n % 1024), so a reader can tell a whole report from a mix.
STRESS_API long stress_report_progress(int milliseconds, int gap_microseconds) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    long reports   = 0;
    while (std::chrono::steady_clock::now() < end) {
        if (upstream_callback != nullptr) {
            const int step = static_cast<int>(reports & 0x3fffffff);
            upstream_callback(step, step + 1, static_cast<float>(step % 1024), upstream_data);
        }
        reports++;
        if (gap_microseconds > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(gap_microseconds));
        }
    }
    return reports;
}

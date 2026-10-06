#include "sd_dart_wrapper.h"

#include <mutex>

namespace {

struct ProgressRoute {
    // Held while the callback runs, so replacing it waits for a call in
    // flight. Recursive so the callback may replace itself.
    std::recursive_mutex mutex;
    sd_progress_cb_t callback = nullptr;
    void* data                = nullptr;
    bool registered           = false;
};

ProgressRoute& progress_route() {
    // Never destroyed: a thread can still report progress after exit() has
    // run static destructors.
    static ProgressRoute* route = new ProgressRoute;
    return *route;
}

void forward_progress(int step, int steps, float time, void*) {
    ProgressRoute& route = progress_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    if (route.callback != nullptr) {
        route.callback(step, steps, time, route.data);
    }
}

}  // namespace

void sd_dart_set_progress_callback(sd_progress_cb_t callback, void* data) {
    ProgressRoute& route = progress_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    route.callback = callback;
    route.data     = data;
    // Upstream keeps its callback in an unsynchronized global, so it is
    // written once rather than on every change.
    if (!route.registered) {
        sd_set_progress_callback(forward_progress, nullptr);
        route.registered = true;
    }
}

void sd_dart_clear_progress_callback(void* callback) {
    ProgressRoute& route = progress_route();
    std::lock_guard<std::recursive_mutex> lock(route.mutex);
    if (callback == nullptr || callback == reinterpret_cast<void*>(route.callback)) {
        route.callback = nullptr;
        route.data     = nullptr;
    }
}

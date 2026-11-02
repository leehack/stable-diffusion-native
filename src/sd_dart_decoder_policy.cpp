#include "sd_dart_internal.h"

#include <mutex>
#include <unordered_set>

namespace {
struct Policies {
    std::mutex mutex;
    std::unordered_set<const void*> intrinsic_tiny;
};

Policies& policies() {
    // Native calls and exit teardown can still use the registry at C exit.
    static auto* value = new Policies();
    return *value;
}
}  // namespace

bool sd_dart_intrinsic_tiny_required(const void* engine) {
    auto& state = policies();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.intrinsic_tiny.count(engine) != 0;
}

void sd_dart_set_intrinsic_tiny_required(const void* engine, bool required) {
    auto& state = policies();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (required) {
        state.intrinsic_tiny.insert(engine);
    } else {
        state.intrinsic_tiny.erase(engine);
    }
}

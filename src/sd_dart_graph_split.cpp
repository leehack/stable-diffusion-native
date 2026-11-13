#include "sd_dart_internal.h"
#include "sd_dart_wrapper.h"

#include "ggml.h"

#include <atomic>
#include <thread>
#include <type_traits>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// Set by the top-level CMakeLists.txt: whether Metal is compiled in.
#ifndef SD_DART_METAL
#define SD_DART_METAL 0
#endif

namespace {

// 2^26 elements: on an iPhone 16 Pro the VAE decode then runs in command
// buffers well under the GPU watchdog's limit (llamadart#1047).
#if SD_DART_METAL && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
constexpr uint64_t kDefaultThreshold = 1ull << 26;
#else
constexpr uint64_t kDefaultThreshold = 0;
#endif

enum Registration : int { kUnregistered, kRegistering, kRegistered };

// Never destroyed: upstream's threads can still evaluate graphs after exit()
// has run static destructors.
struct Split {
    std::atomic<uint64_t> threshold{kDefaultThreshold};
    std::atomic<uint64_t> cuts{0};
    std::atomic<int> registration{kUnregistered};
};
static_assert(std::is_trivially_destructible<Split>::value, "Split must outlive exit()");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "the callback must not take a lock");
Split split;

uint64_t elements(const ggml_tensor* tensor) {
    return static_cast<uint64_t>(tensor->ne[0]) * static_cast<uint64_t>(tensor->ne[1]) *
           static_cast<uint64_t>(tensor->ne[2]) * static_cast<uint64_t>(tensor->ne[3]);
}

// Upstream's graph evaluator computes the nodes up to and including the one
// for which `ask` returns true as one backend call, then calls again with
// `ask` false, which is where a cut is counted. Cutting after each heavy node
// bounds one command buffer. Keeps no state but its counters, as it runs on
// upstream's threads.
bool split_graph(ggml_tensor* node, bool ask, void*) {
    if (!ask) {
        split.cuts.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    const uint64_t threshold = split.threshold.load(std::memory_order_relaxed);
    if (threshold == 0) {
        return false;
    }
    uint64_t cost = elements(node);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            cost += elements(node->src[i]);
        }
    }
    if (cost < threshold) {
        return false;
    }
    return true;
}

}  // namespace

void sd_dart_graph_split_install() {
    if (split.threshold.load() == 0) {
        return;
    }
    // Upstream keeps its callback in an unsynchronized global, so it is
    // written once rather than on every call.
    int expected = kUnregistered;
    if (split.registration.compare_exchange_strong(expected, kRegistering)) {
        sd_set_backend_eval_callback(split_graph, nullptr);
        split.registration.store(kRegistered);
        return;
    }
    while (split.registration.load() != kRegistered) {
        std::this_thread::yield();
    }
}

void sd_dart_graph_split_set_threshold(uint64_t elements) {
    split.threshold.store(elements);
}

uint64_t sd_dart_graph_split_threshold(void) {
    return split.threshold.load();
}

uint64_t sd_dart_graph_split_cuts(void) {
    return split.cuts.load();
}

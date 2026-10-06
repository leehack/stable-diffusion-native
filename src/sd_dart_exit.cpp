#include "sd_dart_wrapper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace {

struct Entry {
    void (*free_fn)(void*) = nullptr;
    // Asks the object to end the call in flight on it early. May be null.
    void (*cancel_fn)(void*) = nullptr;
    int32_t stage            = 0;
    uint64_t order           = 0;
};

// A thread that just left a call in flight usually still makes short calls on
// the same objects, such as reading the model version after a load. Teardown
// gives it this long to finish them.
constexpr std::chrono::milliseconds kSettleTime{250};
// How often teardown repeats its cancellation while it waits.
constexpr std::chrono::milliseconds kCancelInterval{20};

struct Registry {
    std::mutex mutex;
    std::condition_variable idle;
    std::unordered_map<void*, Entry> objects;
    std::chrono::steady_clock::time_point last_call_end{};
    uint64_t next_order    = 0;
    int32_t calls          = 0;
    int32_t creating_calls = 0;
    int32_t wait_ms        = 2000;
    std::atomic<bool> armed{false};
    std::atomic<bool> torn_down{false};
};

Registry& state() {
    // Never destroyed: other threads still reach it after static destructors.
    static auto* registry = new Registry();
    return *registry;
}

thread_local bool teardown_thread = false;

// Calls in flight on this thread. The registry counts the outermost one.
thread_local int32_t call_depth = 0;

// Called with the registry locked. Returns whether the calling thread may go
// on to use tracked objects. Once teardown has begun, teardown waits for a
// thread that is in a call in flight, and the teardown thread gets false. Any
// other thread may hold objects that teardown frees, so it never returns to
// its caller.
bool admit(Registry& registry, std::unique_lock<std::mutex>& lock) {
    if (!registry.torn_down) {
        return true;
    }
    if (teardown_thread) {
        return false;
    }
    if (call_depth > 0) {
        return true;
    }
    lock.unlock();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::hours(1));
    }
}

#if defined(__APPLE__)
struct StaticDestructor {
    void (*destroy)(void*);
    void* object;
};

void destroy_static(void* argument) {
    const StaticDestructor destructor = *static_cast<StaticDestructor*>(argument);
    free(argument);
    sd_dart_exit_teardown();
    destructor.destroy(destructor.object);
}

using RegisterDestructor = int (*)(void (*)(void*), void*, void*);

std::atomic<RegisterDestructor> system_cxa_atexit{nullptr};
#endif

// Registers teardown to run at exit once something is tracked, so that it
// does not wait for the first static of this library to be destroyed. atexit
// handlers run in reverse order of registration, so this one runs after the
// destructors of statics created later; __cxa_atexit below covers those.
void arm() {
#if defined(__APPLE__)
    if (!state().armed.exchange(true)) {
        atexit(sd_dart_exit_teardown);
    }
#endif
}

bool insert(void* object, void (*free_fn)(void*), void (*cancel_fn)(void*), int32_t stage) {
    if (object == nullptr || free_fn == nullptr ||
        stage < SD_DART_EXIT_STAGE_CONTEXT_USER || stage > SD_DART_EXIT_STAGE_RESOURCE) {
        return false;
    }
    Registry& registry = state();
    {
        std::unique_lock<std::mutex> lock(registry.mutex);
        if (!admit(registry, lock)) {
            return false;
        }
        registry.objects[object] = {free_fn, cancel_fn, stage, registry.next_order++};
    }
    arm();
    return true;
}

struct Call {
    Call() { sd_dart_exit_call_begin(); }
    ~Call() { sd_dart_exit_call_end(); }
    Call(const Call&)            = delete;
    Call& operator=(const Call&) = delete;
};

// A call in flight that tracks what it creates. Teardown waits for it even
// when nothing is tracked yet.
struct CreatingCall {
    CreatingCall() {
        sd_dart_exit_call_begin();
        Registry& registry = state();
        std::lock_guard<std::mutex> lock(registry.mutex);
        ++registry.creating_calls;
    }
    ~CreatingCall() {
        {
            Registry& registry = state();
            std::lock_guard<std::mutex> lock(registry.mutex);
            --registry.creating_calls;
        }
        sd_dart_exit_call_end();
    }
    CreatingCall(const CreatingCall&)            = delete;
    CreatingCall& operator=(const CreatingCall&) = delete;
};

void free_context(void* object) {
    free_sd_ctx(static_cast<sd_ctx_t*>(object));
}

void cancel_context(void* object) {
    sd_cancel_generation(static_cast<sd_ctx_t*>(object), SD_CANCEL_ALL);
}

}  // namespace

#if defined(__APPLE__)
extern "C" {
// The C++ runtime registers the destructor of every static in this library
// through __cxa_atexit, and the linker binds those calls to this definition.
// Tracked objects use such statics, some of which stable-diffusion.cpp and
// ggml create on first use at any time, so teardown has to run before the
// first of them is destroyed: each destructor is registered behind a call to
// teardown. A plain atexit handler cannot do that, as it only precedes the
// statics that exist when it is registered.
__attribute__((visibility("hidden"))) int __cxa_atexit(void (*destroy)(void*), void* object, void* dso_handle) {
    RegisterDestructor system_register = system_cxa_atexit.load();
    if (system_register == nullptr) {
        system_register = reinterpret_cast<RegisterDestructor>(dlsym(RTLD_NEXT, "__cxa_atexit"));
        if (system_register == nullptr) {
            return -1;
        }
        system_cxa_atexit.store(system_register);
    }
    auto* destructor = static_cast<StaticDestructor*>(malloc(sizeof(StaticDestructor)));
    if (destructor == nullptr) {
        return -1;
    }
    *destructor      = {destroy, object};
    const int status = system_register(destroy_static, destructor, dso_handle);
    if (status != 0) {
        free(destructor);
    }
    return status;
}
}
#endif

bool sd_dart_exit_track(void* object, void (*free_fn)(void*), int32_t stage) {
    return insert(object, free_fn, nullptr, stage);
}

bool sd_dart_exit_untrack(void* object) {
    Registry& registry = state();
    std::unique_lock<std::mutex> lock(registry.mutex);
    if (!admit(registry, lock)) {
        return false;
    }
    return registry.objects.erase(object) != 0;
}

void sd_dart_exit_free(void* object) {
    if (object == nullptr) {
        return;
    }
    Registry& registry      = state();
    void (*free_fn)(void*) = nullptr;
    {
        std::unique_lock<std::mutex> lock(registry.mutex);
        if (!admit(registry, lock)) {
            return;
        }
        const auto found = registry.objects.find(object);
        if (found == registry.objects.end()) {
            return;
        }
        free_fn = found->second.free_fn;
        registry.objects.erase(found);
        if (call_depth++ == 0) {
            ++registry.calls;
        }
    }
    free_fn(object);
    sd_dart_exit_call_end();
}

int32_t sd_dart_exit_tracked_count(void) {
    Registry& registry = state();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return static_cast<int32_t>(registry.objects.size());
}

void sd_dart_exit_call_begin(void) {
    if (call_depth > 0) {
        ++call_depth;
        return;
    }
    Registry& registry = state();
    std::unique_lock<std::mutex> lock(registry.mutex);
    if (!admit(registry, lock)) {
        return;
    }
    ++registry.calls;
    call_depth = 1;
}

void sd_dart_exit_call_end(void) {
    if (call_depth == 0 || --call_depth > 0) {
        return;
    }
    Registry& registry = state();
    std::unique_lock<std::mutex> lock(registry.mutex);
    --registry.calls;
    registry.idle.notify_all();
    if (admit(registry, lock)) {
        registry.last_call_end = std::chrono::steady_clock::now();
    }
}

void sd_dart_exit_set_wait_ms(int32_t wait_ms) {
    Registry& registry = state();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.wait_ms = std::max(wait_ms, 0);
}

void sd_dart_exit_teardown(void) {
    Registry& registry = state();
    std::vector<std::pair<void*, Entry>> objects;
    {
        std::unique_lock<std::mutex> lock(registry.mutex);
        if (registry.torn_down.exchange(true)) {
            return;
        }
        teardown_thread = true;
        // With nothing to free and nothing being created, a call in flight is
        // no reason to hold up the exit.
        if (registry.creating_calls == 0 && registry.objects.empty()) {
            return;
        }
        // A call in flight on this thread cannot end while teardown runs.
        const int32_t own_calls = call_depth > 0 ? 1 : 0;
        const auto deadline     = std::chrono::steady_clock::now() + std::chrono::milliseconds(registry.wait_ms);
        while (registry.calls != own_calls) {
            // generate_image() clears a cancellation that was requested before
            // it started, so the request is repeated until the calls end.
            for (const auto& [object, entry] : registry.objects) {
                if (entry.cancel_fn != nullptr) {
                    entry.cancel_fn(object);
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return;
            }
            registry.idle.wait_until(lock, std::min(deadline, now + kCancelInterval));
        }
        registry.idle.wait_until(lock, registry.last_call_end + kSettleTime, [] { return false; });
        objects.assign(registry.objects.begin(), registry.objects.end());
    }
    std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) {
        return a.second.stage != b.second.stage ? a.second.stage < b.second.stage : a.second.order > b.second.order;
    });
    for (const auto& [object, entry] : objects) {
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            registry.objects.erase(object);
        }
        entry.free_fn(object);
    }
}

sd_ctx_t* sd_dart_new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    CreatingCall call;
    sd_ctx_t* context = new_sd_ctx(sd_ctx_params);
    insert(context, free_context, cancel_context, SD_DART_EXIT_STAGE_CONTEXT);
    return context;
}

bool sd_dart_generate_image(sd_ctx_t* sd_ctx,
                            const sd_img_gen_params_t* sd_img_gen_params,
                            sd_image_t** images_out,
                            int* num_images_out) {
    Call call;
    return generate_image(sd_ctx, sd_img_gen_params, images_out, num_images_out);
}

void sd_dart_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    Registry& registry = state();
    // Teardown and sd_dart_exit_free() untrack a context under this lock
    // before they free it, so a tracked context is alive while it is held.
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.objects.find(sd_ctx);
    if (found != registry.objects.end() && found->second.cancel_fn == cancel_context) {
        sd_cancel_generation(sd_ctx, mode);
    }
}

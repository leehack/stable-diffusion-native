#ifndef SD_DART_WRAPPER_H
#define SD_DART_WRAPPER_H

#include "stable-diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Progress for callers that cannot give stable-diffusion.cpp a callback.
//
// sd_set_progress_callback() calls back on the thread that is loading or
// generating, for as long as that call runs. A callback into a managed
// runtime, such as a Dart NativeCallable, cannot be made safe there: the
// runtime may already be shutting down, and waiting for a call in flight can
// deadlock it. Here the library records the reports itself and the caller
// reads them, in order, whenever it likes.

// One progress report: the arguments stable-diffusion.cpp passed to its
// progress callback. Reports are numbered from 1 in the order they were
// recorded; `sequence` is that number.
typedef struct {
    uint64_t sequence;
    int32_t step;
    int32_t steps;
    float time;
} sd_dart_progress_t;

// Records progress instead of printing progress bars to stdout, for every
// later call in the process, loads included. Idempotent, and when it returns
// the recorder is registered whichever thread registered it.
//
// The first call registers the recorder with sd_set_progress_callback(),
// which is not synchronized: make it before another thread starts a load or
// generation. Do not call sd_set_progress_callback() afterwards; it replaces
// the recorder, and this function does not register it again.
SD_API void sd_dart_progress_enable(void);

// Copies the reports whose sequence is greater than `after` into `reports`,
// oldest first, at most `capacity` of them, and returns how many. If `latest`
// is not NULL it receives the sequence of the newest report recorded, 0 if
// there is none: a call with a `capacity` of 0 only asks for that.
//
// - The sequences of one call's reports are consecutive.
// - The library keeps the 4095 most recent reports. If more than that were
//   recorded since `after`, the older ones are gone and the first report
//   returned is not `after + 1`: that is how a caller sees that it fell
//   behind.
// - A call may return fewer reports than there are, also fewer than
//   `capacity`. The caller has them all once the last sequence it holds
//   equals `latest`.
//
// Callable from any thread at any time, also before
// sd_dart_progress_enable(). It takes no lock, makes no system call, never
// waits for a reporting thread and copies at most `capacity` + 4 reports, so
// it is safe as a Dart leaf call.
//
// Reports are process-wide, as upstream's callback is: every context's loads
// and generations go into the one sequence, and a report does not say which
// call made it. To follow one call, ask for `latest` before starting it and
// read from there.
SD_API size_t sd_dart_progress_read(uint64_t after,
                                    sd_dart_progress_t* reports,
                                    size_t capacity,
                                    uint64_t* latest);

// Exit teardown
//
// The library keeps a registry of live native objects and frees what is left
// of it when the process exits without freeing them itself, as a Flutter quit
// or hot restart does. ggml-metal aborts in its static destructor while any
// Metal buffer is still allocated, so teardown has to run before that
// destructor. On Apple platforms it runs during exit(), before the first
// static of this library is destroyed.
//
// On Linux, Android excepted, exit() frees nothing, destroys no static of the
// library and blocks no thread. Before, it destroyed the statics of the
// library under the threads that were still inside a load or a generation
// (https://github.com/leehack/llamadart/issues/949).
// - The statics of the library are never destroyed, so a call that is still
//   running while the process exits goes on finding them.
// - With no call in flight, exit() goes on as in any process and takes no
//   time in the library, also right after a call.
// - With a call in flight, the rest of exit() would run under it: the exit
//   handlers and static destructors of what the call uses, a GPU driver's
//   among them. Waiting for the call instead would hold exit() up, and a Dart
//   VM aborts when that happens while one of its isolates runs Dart code
//   (https://github.com/leehack/llamadart/issues/977). So with glibc the
//   process ends there, as after dart:io's exit(): buffered stdio output is
//   written and _exit() is called with the status exit() was given. Exit
//   handlers and static destructors that had not run by then do not run,
//   the host's among them. Without glibc's on_exit(), exit() asks the
//   generations to cancel and goes on under the calls.
// - A function below that is called once the exit has begun returns at once
//   and does nothing, so a thread that the host joins at exit is not held:
//   sd_dart_new_sd_ctx() returns NULL, sd_dart_generate_image() false, a
//   device query SD_DART_GPU_UNAVAILABLE, sd_dart_exit_track() and
//   sd_dart_exit_untrack() false, and sd_dart_exit_free() frees nothing.
//   Nothing on Linux needs the objects freed, and an exit with a context
//   left alive is clean, which a free among the exit handlers of a GPU
//   driver may not be.
// - The library registers its exit handler with each static it creates,
//   before the first load or device query, after the first device query and
//   after every load. An exit handler that another library registers later
//   than those runs before it.
// quick_exit() and _exit() run nothing.
//
// On other platforms nothing runs at exit. sd_dart_exit_teardown() runs
// teardown, frees included, on every platform.
//
// Objects are tracked by sd_dart_new_sd_ctx(), before it returns, and by
// sd_dart_exit_track().
//
// Teardown asks every tracked context to cancel its generation, waits a
// bounded time for the calls in flight and then frees the tracked objects in
// sd_dart_exit_stage order, latest tracked first within a stage. If a call is
// still in flight when the wait ends, it frees nothing. With no call in
// flight it waits only until 250 ms have passed since the last one ended, the
// time a thread gets for the short calls that follow a call in flight. It
// does not wait at all when nothing is tracked and no call in flight is
// creating or freeing a tracked object or querying the devices:
// sd_dart_new_sd_ctx(), sd_dart_exit_free(), sd_dart_gpu_device_count() and
// sd_dart_gpu_device_memory() are waited for whatever the registry holds.
//
// A call in flight is the time a thread spends inside sd_dart_new_sd_ctx(),
// sd_dart_generate_image(), sd_dart_exit_free(), sd_dart_gpu_device_count()
// or sd_dart_gpu_device_memory(), or between sd_dart_exit_call_begin() and
// sd_dart_exit_call_end().
//
// The wait ends when the calls do, and it has two bounds:
// - 15 s while a load, a generation or a device query is in flight, that is
//   sd_dart_new_sd_ctx(), sd_dart_generate_image(),
//   sd_dart_gpu_device_count() or sd_dart_gpu_device_memory(). The first
//   device query initializes the device, as a load otherwise does; a later
//   one returns within microseconds. stable-diffusion.cpp
//   reads a cancellation only between the phases of a generation: before a
//   sampling step and before the decode of each image. A load, the text
//   encoder, a sampling step and the VAE decode of an image each run to their
//   end, and the decode alone takes seconds: with SD-Turbo on an M4 Max,
//   4.1 s for 768 x 768 pixels and 7.4 s for 1024 x 1024. So a process that
//   quits during a large generation can take that much longer to exit.
// - 2 s otherwise: for sd_dart_exit_free() and for calls marked with
//   sd_dart_exit_call_begin().
// A phase that outlasts its bound costs the whole wait and still leaves
// everything allocated: the process then exits as it would have without this
// registry, only that much later.
//
// Once teardown has begun, the objects a thread holds may be freed as soon as
// it has no call in flight. From then on, the end of a thread's outermost call
// in flight never returns to its caller. Neither does a function documented
// as "blocks after teardown" when it is called outside a call in flight;
// inside one it works as before, since teardown is waiting for that call. The
// thread that runs teardown is exempt: there, such a function returns without
// tracking, untracking or freeing anything. A thread blocked this way stays
// blocked until the process is gone, so a static destructor or atexit handler
// of another library that joins such a thread hangs the exit.
//
// A native call that is not a call in flight is not waited for. Teardown only
// allows a thread 250 ms after its last call in flight to finish what follows
// it, which covers short calls such as sd_get_model_version_name() after a
// load. A longer call on a tracked context that is not a call in flight, such
// as generate_image() or generate_video(), is a use after free wherever
// teardown frees, also where exiting with the context alive was harmless.
// Contexts created by new_sd_ctx() are not tracked, and exiting with one
// alive behaves as it did before this registry existed, except that on Linux
// exit() no longer destroys the statics of the library under a thread that
// is using one.

// Order in which exit teardown frees tracked objects: every object of a lower
// stage before any object of a higher one, so an object goes before the
// objects it uses.
enum sd_dart_exit_stage {
    // State that uses a context.
    SD_DART_EXIT_STAGE_CONTEXT_USER = 0,
    // sd_ctx_t and the other contexts of stable-diffusion.h.
    SD_DART_EXIT_STAGE_CONTEXT = 1,
    // What a context uses.
    SD_DART_EXIT_STAGE_RESOURCE = 2,
};

// Tracks `object` so that exit teardown frees it with `free_fn`, which
// teardown calls on the thread that runs exit(): pass a native function, never
// a callback into a managed runtime. Tracking an address again replaces its
// entry. Returns false for a null argument or an unknown stage. Blocks after
// teardown.
//
// For C and C++ callers. From Dart an object has to be tracked by the native
// call that creates it: an isolate that is killed during that call never
// reaches a later call that would track it.
SD_API bool sd_dart_exit_track(void* object, void (*free_fn)(void*), int32_t stage);

// Stops tracking `object` without freeing it. Returns whether it was tracked.
// Blocks after teardown.
SD_API bool sd_dart_exit_untrack(void* object);

// Stops tracking `object` and frees it with the function it was tracked with:
// free_sd_ctx() for a context from sd_dart_new_sd_ctx(). Does nothing when
// `object` is not tracked, so each tracked object is freed once, whether by
// this call or by teardown. Usable as a Dart NativeFinalizer callback. Blocks
// after teardown.
SD_API void sd_dart_exit_free(void* object);

// Number of tracked objects.
SD_API int32_t sd_dart_exit_tracked_count(void);

// Mark the start and end of a call in flight on the calling thread, for C and
// C++ callers of native functions that have no sd_dart_ wrapper. They nest.
// Each begin needs one end on the same thread: a thread that never reaches end
// keeps teardown from freeing anything. Do not call them from Dart, where an
// isolate that is killed during the native call in between never reaches end.
// begin blocks after teardown.
SD_API void sd_dart_exit_call_begin(void);
SD_API void sd_dart_exit_call_end(void);

// Sets how long teardown waits for calls in flight: `work_wait_ms` while a
// load, a generation or a device query is among them, `wait_ms` otherwise.
// Negative values are treated as zero. The defaults are 2000 and 15000. A host that would rather
// abort in ggml-metal than exit late passes one value for both.
SD_API void sd_dart_exit_set_wait_ms(int32_t wait_ms, int32_t work_wait_ms);

// Runs exit teardown now, frees included on every platform; later runs do
// nothing. Afterwards tracked objects are unusable and other threads that
// reach the functions above stay blocked, so call it only as the last step
// before the process exits and follow it directly with exit() or _exit() on
// the same thread. It is meant for native hosts. Do not bind it from Dart: a
// Dart program that returns from main after it waits forever for its blocked
// isolates.
SD_API void sd_dart_exit_teardown(void);

// new_sd_ctx() that tracks the context in the CONTEXT stage before it
// returns. Free the context with sd_dart_exit_free(). A load cannot be
// cancelled: teardown waits for it, with the longer bound. Blocks after
// teardown.
SD_API sd_ctx_t* sd_dart_new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);

// generate_image() as a call in flight. Takes and returns what generate_image()
// does. Teardown cancels it with SD_CANCEL_ALL, which stable-diffusion.cpp
// honors before a sampling step and before the decode of each image, and
// waits for it with the longer bound. Blocks after teardown.
SD_API bool sd_dart_generate_image(sd_ctx_t* sd_ctx,
                                   const sd_img_gen_params_t* sd_img_gen_params,
                                   sd_image_t** images_out,
                                   int* num_images_out);

// sd_cancel_generation() for a tracked context, callable from any thread while
// another one generates. Does nothing when `sd_ctx` is not tracked: one that
// sd_dart_exit_free() or teardown already freed, or one from new_sd_ctx().
// Never blocks, also after teardown.
SD_API void sd_dart_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);

// GPU device memory
//
// The total and the free memory of the GPU a model would load on, so that a
// caller can refuse a model that does not fit before it loads it. The figures
// are ggml's (ggml_backend_dev_memory), the ones stable-diffusion.cpp's own
// automatic fit works with:
//
// - Vulkan, discrete GPU: the heaps flagged device-local, added up. With
//   VK_EXT_memory_budget, `free_bytes` is the budget of those heaps less what
//   this process uses of them, which is the driver's estimate of what the
//   process can still allocate and so reflects other processes. Without the
//   extension it is the heap sizes: `free_bytes == total_bytes`.
// - Vulkan, integrated GPU: every heap, added up, in the same way. The heaps
//   are host memory that the device shares with the system, and a driver that
//   exposes that memory as more than one heap counts it more than once, so the
//   host's available memory is a second limit there.
// - Metal: `total_bytes` is the device's recommendedMaxWorkingSetSize, and
//   `free_bytes` is that less the device's currentAllocatedSize, which counts
//   the allocations of this process only. Apple GPUs report as
//   SD_DART_GPU_DEVICE_DISCRETE although their memory is unified.
//
// A budget that the driver reports below the current use reads as 0 free
// bytes, and `free_bytes` never exceeds `total_bytes`.
//
// `free_bytes` is what the device has free now, not what is left once the
// loaded contexts are in use. stable-diffusion.cpp moves a context's weights
// to the device when they are first used, unless sd_ctx_params_t.eager_load
// is set: without it `free_bytes` is the same after sd_dart_new_sd_ctx() as
// before and drops with the first generation, with it at the load (SDXS on
// Metal: by 650 MiB either way). What a generation computes in is allocated
// while it runs. Upstream's API reports neither what a context will take nor
// what it holds, so a caller that admits a second model against `free_bytes`
// sets eager_load, or keeps count of what it has loaded itself.
//
// The functions create no context and load no model. The first call registers
// ggml's backends, as sd_list_devices() does, and with that initializes the
// devices: on Vulkan it creates the instance, and on Metal it compiles the
// shader libraries, which takes about 50 ms when the system has them cached
// and took 16 s on an M4 Max and tens of seconds on GitHub's macOS runners
// when it had not. Make the first call on a thread that may wait that long,
// never on a UI thread. A later call returns within microseconds. They are callable
// from any thread, also while another one loads or generates.
//
// A query reads ggml's device registry, which exit() destroys, so exit
// teardown waits for one that is in flight, also when nothing is tracked,
// and a query blocks after teardown. The wait has the bound of a load, 15 s
// unless sd_dart_exit_set_wait_ms() changed it. Past that bound teardown
// frees nothing and the exit goes on under the query, as it would without
// the registry: the registry may then be destroyed while the query reads it.
// A first query on Metal that compiles its libraries can outlast the bound.

enum sd_dart_gpu_status {
    SD_DART_GPU_OK = 0,
    // A null `out`, or a negative `device_index` other than
    // SD_DART_GPU_DEFAULT_DEVICE.
    SD_DART_GPU_INVALID_ARGUMENT = -1,
    // The library was built without a GPU backend, as the CPU targets are.
    SD_DART_GPU_NO_BACKEND = -2,
    // The library has a GPU backend, but no such device: none was found, or
    // `device_index` is past the last one.
    SD_DART_GPU_NO_DEVICE = -3,
    // The device is there but its memory is not known: the backend reports a
    // total of 0, or it failed to answer.
    SD_DART_GPU_UNAVAILABLE = -4,
};

enum sd_dart_gpu_device_type {
    // A GPU with memory of its own.
    SD_DART_GPU_DEVICE_DISCRETE = 1,
    // A GPU that uses host memory.
    SD_DART_GPU_DEVICE_INTEGRATED = 2,
};

enum {
    // The device stable-diffusion.cpp uses when sd_ctx_params_t.backend names
    // none: the first discrete GPU, or else the first integrated one. The
    // SD_VK_DEVICE environment variable, with which upstream uses the Vulkan
    // device of that number for such a context if it initializes, is not
    // read: a caller that sets it asks for that device by its index.
    SD_DART_GPU_DEFAULT_DEVICE = -1,
};

typedef struct {
    uint64_t total_bytes;
    uint64_t free_bytes;
    // An sd_dart_gpu_device_type.
    int32_t type;
    // The device name of sd_list_devices(), which sd_ctx_params_t.backend
    // accepts, such as "Vulkan0" or "MTL0", and its description. Both are
    // NUL-terminated and cut to fit.
    char name[64];
    char description[256];
} sd_dart_gpu_device_memory_t;

// The number of GPU devices, discrete and integrated, or
// SD_DART_GPU_NO_BACKEND. They are numbered from 0 in the order
// sd_list_devices() lists them, which has other devices, such as the CPU, in
// between.
SD_API int32_t sd_dart_gpu_device_count(void);

// Writes the memory of GPU device `device_index`, or of the default device
// for SD_DART_GPU_DEFAULT_DEVICE, to `out` and returns SD_DART_GPU_OK.
// Otherwise returns another sd_dart_gpu_status and leaves `out` as it was.
// Each call asks the device again.
SD_API int32_t sd_dart_gpu_device_memory(int32_t device_index, sd_dart_gpu_device_memory_t* out);

// Log forwarding
//
// sd_set_log_callback() calls back on whichever thread logs, the model
// loader's own threads among them, with a text that is only valid during the
// call. A managed runtime cannot take that call, for the reasons given for
// progress above, and without a callback stable-diffusion.cpp's messages go
// nowhere: a load that fails returns NULL and nothing else. Here the library
// copies each message into a buffer of its own, and the caller reads the
// messages, in order, whenever it likes.
//
// A message is the text upstream passes to its callback, "<file>:<line> -
// <text>" for stable-diffusion.cpp and "ggml - <text>" for ggml, without the
// line break at its end. Messages are numbered from 1 in the order they were
// recorded.

enum {
    // A buffer of this size holds any message and its terminating NUL.
    // Upstream's longer messages are cut to fit, between two UTF-8 sequences.
    SD_DART_LOG_TEXT_SIZE = 4096,
};

// Records log messages instead of dropping them, for every later call in the
// process. Idempotent, and when it returns the recorder is registered
// whichever thread registered it.
//
// Nothing changes for a process that does not call it. Once it is called,
// ggml's messages are recorded as well, which until the first context exists
// ggml would print to stderr, such as the lines of a device's
// initialization. Nothing recorded is printed: a caller that wants the
// messages on stderr prints what it reads. What ggml's backends write to
// stderr themselves, not through ggml's log, still goes there.
//
// The first call registers the recorder with sd_set_log_callback() and
// ggml's log, neither of which is synchronized: make it before another thread
// starts a load, a generation or a device query, that is sd_list_devices(),
// sd_dart_gpu_device_count() or sd_dart_gpu_device_memory(), which logs
// through ggml for as long as it runs. Do not call sd_set_log_callback()
// afterwards; it replaces the recorder, and this function does not register
// it again.
SD_API void sd_dart_log_enable(void);

// Sets the lowest sd_log_level_t that is recorded, SD_LOG_INFO by default.
// SD_LOG_ERROR + 1 records nothing. A message below the level gets no
// sequence, and sd_dart_last_error() does not depend on the level.
SD_API void sd_dart_log_set_level(int32_t level);

// Reads the oldest message whose sequence is greater than `after` and returns
// its sequence, or returns 0 when there is none. Copies the text to `text`,
// at most `capacity` - 1 bytes of it, ending between two UTF-8 sequences,
// and terminates it. `level` receives the
// message's sd_log_level_t and `length` the bytes of the whole text, also
// when fewer were copied; either may be NULL, and so may `text` with a
// `capacity` of 0. Reading removes nothing: read on with the sequence
// returned.
//
// - The library keeps the most recent messages that fit in 256 KiB, about
//   2500 lines of 100 bytes. If the message after `after` is gone, the
//   sequence returned is not `after + 1`: that is how a caller sees that it
//   fell behind, and by how many messages.
// - Callable from any thread at any time, also before sd_dart_log_enable(),
//   during exit teardown and after it. It allocates nothing, and waits only
//   for the copy of one message by another thread. Should that thread not
//   finish the copy within 100 ms, because it died during it, which only the
//   end of the process can cause, or because the scheduler kept it off its
//   processor that long, the call returns 0.
//
// Messages are process-wide, as upstream's callback is: every context's go
// into the one sequence, and a message does not say which call made it.
SD_API uint64_t sd_dart_log_read(uint64_t after, char* text, size_t capacity, int32_t* level, size_t* length);

// The number of messages that left the buffer newer than every message a
// read had returned by then: what was lost to a caller that reads in order,
// and everything that left for a caller that never reads. It also counts a
// message that could not be recorded because the thread holding the buffer
// did not release it within 100 ms.
SD_API uint64_t sd_dart_log_dropped(void);

// Copies the SD_LOG_ERROR messages that were recorded, by any thread, while
// the calling thread's most recent sd_dart_new_sd_ctx() or
// sd_dart_generate_image() ran: the reason stable-diffusion.cpp gave for a
// call that failed. They are joined by '\n', oldest first, and terminated; at
// most `capacity` - 1 bytes are copied, ending between two UTF-8 sequences.
// Returns the bytes of the whole text, 0 when that call logged no error, when
// the thread has made no such call or when sd_dart_log_enable() was not
// called before it.
//
// - An empty result does not prove that the call logged no error. The errors
//   are kept behind a flag of their own, which only another error or a read
//   of this function holds, for one copy. An error whose thread cannot get
//   that flag within 100 ms is not kept here; with a level that records it,
//   it is still in the log, or counted by sd_dart_log_dropped().
// - Read it on the thread that made the call, before that thread makes
//   another one. A Dart isolate stays on its thread between two native calls
//   that no asynchronous gap separates.
// - Of one call, the 32 most recent error messages are kept, each cut to 511
//   bytes, and only until 32 later ones have replaced them.
// - Errors are process-wide: those that another thread's call logs in the
//   meantime are included.
// - Never blocks after teardown, allocates nothing, and waits only for the
//   copy of one error by another thread, at most 100 ms.
SD_API size_t sd_dart_last_error(char* text, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif  // SD_DART_WRAPPER_H

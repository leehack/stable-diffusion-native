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
// static of this library is destroyed. Elsewhere, where that abort does not
// exist, it runs only when sd_dart_exit_teardown() is called.
//
// Objects are tracked by sd_dart_new_sd_ctx(), before it returns, and by
// sd_dart_exit_track().
//
// Teardown asks every tracked context to cancel its generation, waits a
// bounded time for the calls in flight and then frees the tracked objects in
// sd_dart_exit_stage order, latest tracked first within a stage. If a call is
// still in flight when the wait ends, it frees nothing. With no call in
// flight it does not wait. Neither does it when nothing is tracked and no
// call in flight is creating or freeing a tracked object: sd_dart_new_sd_ctx()
// and sd_dart_exit_free() are waited for whatever the registry holds.
//
// A call in flight is the time a thread spends inside sd_dart_new_sd_ctx(),
// sd_dart_generate_image() or sd_dart_exit_free(), or between
// sd_dart_exit_call_begin() and sd_dart_exit_call_end().
//
// The wait ends when the calls do, and it has two bounds:
// - 15 s while a load or a generation is in flight, that is
//   sd_dart_new_sd_ctx() or sd_dart_generate_image(). stable-diffusion.cpp
//   reads a cancellation only between the phases of a generation: before a
//   sampling step and before the decode of each image. A load, the text
//   encoder, a sampling step and the VAE decode of an image each run to their
//   end, and the decode alone takes seconds: with SD-Turbo on an M4 Max,
//   4.1 s for 768 x 768 pixels and 7.4 s for 1024 x 1024. So a process that
//   quits during a large generation can take that much longer to exit.
// - 2 s otherwise: for sd_dart_exit_free() and for calls marked with
//   sd_dart_exit_call_begin().
// A phase that outlasts its bound leaves everything allocated, and the
// process exits as it would have without this registry.
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
// as generate_image() or generate_video(), is a use after free at exit, also
// where exiting with the context alive was harmless. Contexts created by
// new_sd_ctx() are not tracked, and exiting with one alive behaves as it did
// before this registry existed.

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
// load or a generation is among them, `wait_ms` otherwise. Negative values are
// treated as zero. The defaults are 2000 and 15000. A host that would rather
// abort in ggml-metal than exit late passes one value for both.
SD_API void sd_dart_exit_set_wait_ms(int32_t wait_ms, int32_t work_wait_ms);

// Runs exit teardown now; later runs do nothing. Afterwards tracked objects
// are unusable and other threads that reach the functions above stay blocked,
// so call it only as the last step before the process exits and follow it
// directly with exit() or _exit() on the same thread. It is meant for native
// hosts. Do not bind it from Dart: a Dart program that returns from main
// after it waits forever for its blocked isolates.
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

#ifdef __cplusplus
}
#endif

#endif  // SD_DART_WRAPPER_H

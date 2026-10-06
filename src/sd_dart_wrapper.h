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
// deadlock it. Here the library records progress itself and the caller polls.

// The latest progress report: the arguments stable-diffusion.cpp passed to
// its progress callback. `sequence` counts the reports recorded since
// sd_dart_progress_enable(); 0 means none yet.
typedef struct {
    uint64_t sequence;
    int32_t step;
    int32_t steps;
    float time;
} sd_dart_progress_t;

// Records progress instead of printing progress bars to stdout, for every
// later call in the process, loads included. Idempotent.
//
// The first call registers a recorder with sd_set_progress_callback(), which
// is not synchronized: make it before another thread starts a load or
// generation. Do not call sd_set_progress_callback() afterwards; it replaces
// the recorder, and this function does not register it again.
SD_API void sd_dart_progress_enable(void);

// Copies the latest report into `progress`. Callable from any thread at any
// time, also before sd_dart_progress_enable(). It takes no lock, makes no
// system call and never waits for the reporting thread, so it is safe as a
// Dart leaf call.
//
// Reports are process-wide, as upstream's callback is: two contexts working
// at once share one sequence, and a report does not say which one made it.
// To follow one run, read `sequence` before starting it and poll for larger
// values. Only the latest report is kept, so a slow poller skips steps.
SD_API void sd_dart_progress_read(sd_dart_progress_t* progress);

#ifdef __cplusplus
}
#endif

#endif  // SD_DART_WRAPPER_H

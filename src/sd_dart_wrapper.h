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

#ifdef __cplusplus
}
#endif

#endif  // SD_DART_WRAPPER_H

#ifndef SD_DART_WRAPPER_H
#define SD_DART_WRAPPER_H

#include "stable-diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Progress routing that stays safe when the callback's owner goes away while
// another thread is still inside new_sd_ctx(), generate_image() or any other
// call that reports progress.
//
// Both functions may be called from any thread at any time. When either
// returns, the callback it replaced is not running on another thread and is
// never called again. Called from inside the callback itself, they return at
// once and that one call finishes normally. The callback must not wait for a
// thread that is calling either function.
//
// Use these instead of sd_set_progress_callback(), not together with it: the
// first sd_dart_set_progress_callback() call registers a forwarder there,
// which a later sd_set_progress_callback() call would replace. Make that first
// call before another thread starts a load or generation.

// Routes progress to `callback`, replacing the current one. A NULL `callback`
// discards progress. Either way the library stops printing progress bars to
// stdout.
SD_API void sd_dart_set_progress_callback(sd_progress_cb_t callback, void* data);

// Discards progress from now on if `callback` is the current callback; any
// other value leaves the current one in place. NULL discards whatever is set.
//
// Takes one pointer so it can be a Dart NativeFinalizer callback, with the
// callback's address as the token.
SD_API void sd_dart_clear_progress_callback(void* callback);

#ifdef __cplusplus
}
#endif

#endif  // SD_DART_WRAPPER_H

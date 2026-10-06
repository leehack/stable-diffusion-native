// Native side of tests/dart/exit_teardown.dart, linked against a built
// runtime: it fills in the parameter structs, so that the harness can call
// the runtime's own functions without mirroring those structs in Dart.

#include "sd_dart_wrapper.h"

#include <cstdlib>

#define PROBE_API extern "C" __attribute__((visibility("default")))

namespace {

void quiet_log(enum sd_log_level_t, const char*, void*) {}

}  // namespace

// Keeps the runtime off stdout and stderr without a Dart callback, and gives
// teardown time for a call in flight on a busy machine.
PROBE_API void probe_quiet(void) {
    sd_set_log_callback(quiet_log, nullptr);
    sd_dart_progress_enable();
    sd_dart_exit_set_wait_ms(120000);
}

// Parameters for a model that holds its TAESD decoder itself, as the one
// exit_teardown_runtime_test writes does. Never freed.
PROBE_API sd_ctx_params_t* probe_context_params(const char* model) {
    auto* params = static_cast<sd_ctx_params_t*>(malloc(sizeof(sd_ctx_params_t)));
    sd_ctx_params_init(params);
    params->model_path = model;
    params->taesd_path = model;
    params->eager_load = true;
    return params;
}

PROBE_API sd_img_gen_params_t* probe_generation_params(int steps) {
    auto* params = static_cast<sd_img_gen_params_t*>(malloc(sizeof(sd_img_gen_params_t)));
    sd_img_gen_params_init(params);
    params->prompt                         = "a lighthouse";
    params->negative_prompt                = "";
    params->width                          = 64;
    params->height                         = 64;
    params->seed                           = 7;
    params->sample_params.sample_steps     = steps;
    params->sample_params.guidance.txt_cfg = 1.0f;
    return params;
}

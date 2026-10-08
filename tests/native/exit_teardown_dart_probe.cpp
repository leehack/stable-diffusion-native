// Native side of tests/dart/exit_teardown.dart, linked against a built
// runtime: it fills in the parameter structs, so that the harness can call
// the runtime's own functions without mirroring those structs in Dart.

#include "sd_dart_wrapper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define PROBE_API extern "C" __attribute__((visibility("default")))

namespace {

// Whether the model holds its TAESD decoder itself, as the one
// exit_teardown_runtime_test writes does.
bool holds_taesd(const char* model) {
    std::string header(1 << 16, '\0');
    FILE* file = std::fopen(model, "rb");
    if (file == nullptr) {
        return false;
    }
    header.resize(std::fread(&header[0], 1, header.size(), file));
    std::fclose(file);
    return header.find("\"tae.decoder.") != std::string::npos;
}

}  // namespace

// Keeps the runtime off stdout and stderr without a Dart callback: the
// library records its messages and its progress, and the harness reads the
// messages. Gives teardown time for a call in flight on a busy machine.
PROBE_API void probe_quiet(void) {
    sd_dart_log_enable();
    sd_dart_log_set_level(SD_LOG_DEBUG);
    sd_dart_progress_enable();
    sd_dart_exit_set_wait_ms(120000, 120000);
}

// `default` lets the runtime pick its device. Never freed.
PROBE_API sd_ctx_params_t* probe_context_params(const char* model, const char* backend) {
    auto* params = static_cast<sd_ctx_params_t*>(malloc(sizeof(sd_ctx_params_t)));
    sd_ctx_params_init(params);
    params->model_path = model;
    params->taesd_path = holds_taesd(model) ? model : nullptr;
    params->backend    = std::strcmp(backend, "default") == 0 ? nullptr : backend;
    params->eager_load = true;
    return params;
}

PROBE_API sd_img_gen_params_t* probe_generation_params(int steps, int size) {
    auto* params = static_cast<sd_img_gen_params_t*>(malloc(sizeof(sd_img_gen_params_t)));
    sd_img_gen_params_init(params);
    params->prompt                         = "a lighthouse";
    params->negative_prompt                = "";
    params->width                          = size;
    params->height                         = size;
    params->seed                           = 7;
    params->sample_params.sample_steps     = steps;
    params->sample_params.guidance.txt_cfg = 1.0f;
    return params;
}

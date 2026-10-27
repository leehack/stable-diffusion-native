// Loads a model and generates one image through the guarded wrapper exports,
// as llamadart does, and writes its RGB bytes.
//
// Usage: fp8_runtime_test devices
//        fp8_runtime_test generate <model> <backend> <out.rgb> [<lora> <mode> [<rules>]]
//        fp8_runtime_test plan <model> <backend>
//
// The model is one that `exit_teardown_runtime_test make-model` wrote, or an
// FP8 copy of it: it holds its TAESD decoder, so it needs no VAE. <mode> is a
// LoRA apply mode name: auto, immediately or at_runtime, and <rules> are
// tensor type rules. Both commands print the log, which holds upstream's
// auto-fit plan and LoRA apply mode warnings; `plan` only loads the model.

#include "sd_dart_wrapper.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const int kImageSize = 64;

int list_devices() {
    std::vector<char> devices(sd_list_devices(nullptr, 0) + 1);
    sd_list_devices(devices.data(), devices.size());
    std::printf("%s", devices.data());
    return 0;
}

void print_log() {
    std::vector<char> text(4096);
    for (uint64_t sequence = 0;
         (sequence = sd_dart_log_read(sequence, text.data(), text.size(), nullptr, nullptr)) != 0;) {
        std::printf("%s", text.data());
    }
}

int plan(const char* model, const char* backend) {
    sd_dart_log_enable();
    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    context_params.model_path = model;
    context_params.taesd_path = model;
    context_params.backend    = backend;
    sd_ctx_t* context         = sd_dart_new_sd_ctx(&context_params);
    if (context != nullptr) {
        sd_dart_exit_free(context);
    }
    return context != nullptr ? 0 : 3;
}

int generate(const char* model, const char* backend, const char* output, const char* lora,
             const char* mode, const char* rules) {
    sd_dart_log_enable();
    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    context_params.model_path        = model;
    context_params.taesd_path        = model;
    context_params.backend           = backend;
    context_params.tensor_type_rules = rules;
    if (lora != nullptr) {
        context_params.lora_apply_mode = str_to_lora_apply_mode(mode);
        if (context_params.lora_apply_mode == LORA_APPLY_MODE_COUNT) {
            std::fprintf(stderr, "unknown LoRA apply mode %s\n", mode);
            return 2;
        }
    }
    sd_ctx_t* context         = sd_dart_new_sd_ctx(&context_params);
    if (context == nullptr) {
        std::vector<char> error(4096);
        sd_dart_last_error(error.data(), error.size());
        std::fprintf(stderr, "load failed: %s\n", error.data());
        return 3;
    }

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt                         = "a lighthouse";
    params.negative_prompt                = "";
    params.width                          = kImageSize;
    params.height                         = kImageSize;
    params.seed                           = 7;
    params.sample_params.sample_steps     = 2;
    params.sample_params.guidance.txt_cfg = 1.0f;
    sd_lora_t adapter                     = {false, 1.0f, lora};
    if (lora != nullptr) {
        params.loras      = &adapter;
        params.lora_count = 1;
    }
    sd_image_t* images                    = nullptr;
    int count                             = 0;
    if (!sd_dart_generate_image(context, &params, &images, &count)) {
        std::vector<char> error(4096);
        sd_dart_last_error(error.data(), error.size());
        std::fprintf(stderr, "generation failed: %s\n", error.data());
        sd_dart_exit_free(context);
        return 4;
    }
    int status = 0;
    if (images == nullptr || count != 1 || images[0].width != kImageSize ||
        images[0].height != kImageSize || images[0].channel != 3) {
        std::fprintf(stderr, "unexpected images\n");
        status = 5;
    } else {
        const size_t bytes = static_cast<size_t>(kImageSize) * kImageSize * 3;
        FILE* file         = std::fopen(output, "wb");
        if (file == nullptr || std::fwrite(images[0].data, 1, bytes, file) != bytes ||
            std::fclose(file) != 0) {
            std::fprintf(stderr, "cannot write %s\n", output);
            status = 6;
        }
    }
    free_sd_images(images, count);
    sd_dart_exit_free(context);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "devices") == 0) {
        return list_devices();
    }
    int status = 2;
    if (argc == 4 && std::strcmp(argv[1], "plan") == 0) {
        status = plan(argv[2], argv[3]);
    } else if ((argc == 5 || argc == 7 || argc == 8) && std::strcmp(argv[1], "generate") == 0) {
        status = generate(argv[2], argv[3], argv[4], argc >= 7 ? argv[5] : nullptr,
                          argc >= 7 ? argv[6] : nullptr, argc == 8 ? argv[7] : nullptr);
    } else {
        std::fprintf(stderr,
                     "usage: %s devices | generate <model> <backend> <out.rgb> [<lora> <mode> [<rules>]] | "
                     "plan <model> <backend>\n",
                     argv[0]);
        return status;
    }
    print_log();
    return status;
}

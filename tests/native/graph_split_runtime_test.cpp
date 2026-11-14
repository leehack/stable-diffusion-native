// Generates with the graph split at each given threshold, in one context, and
// reports the cuts the library made.
//
// Usage: graph_split_runtime_test devices
//        graph_split_runtime_test run <model> <backend> <out-prefix> <threshold>...
//
// The model is one that `exit_teardown_runtime_test make-model` wrote: it
// holds its TAESD decoder, so it needs no VAE. Prints `DEFAULT <threshold>`
// before anything else, then per threshold `RUN <i> <threshold> <getter>
// <cuts>` with the process-wide cut count after that generation, and writes
// image i's RGB bytes to <out-prefix>.<i>.rgb.

#include "sd_dart_wrapper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

int fail(const char* what) {
    std::vector<char> error(4096);
    sd_dart_last_error(error.data(), error.size());
    std::fprintf(stderr, "%s: %s\n", what, error.data());
    return 3;
}

int write_image(const sd_image_t& image, const std::string& path) {
    const size_t bytes = static_cast<size_t>(kImageSize) * kImageSize * 3;
    if (image.width != kImageSize || image.height != kImageSize || image.channel != 3) {
        std::fprintf(stderr, "unexpected image\n");
        return 5;
    }
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr || std::fwrite(image.data, 1, bytes, file) != bytes || std::fclose(file) != 0) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 6;
    }
    return 0;
}

int run(const char* model, const char* backend, const std::string& prefix, int count, char** thresholds) {
    sd_dart_log_enable();
    std::printf("DEFAULT %llu\n", static_cast<unsigned long long>(sd_dart_graph_split_threshold()));
    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    context_params.model_path = model;
    context_params.taesd_path = model;
    context_params.backend    = backend;
    sd_ctx_t* context         = sd_dart_new_sd_ctx(&context_params);
    if (context == nullptr) {
        return fail("load failed");
    }
    int status = 0;
    for (int i = 0; i < count && status == 0; ++i) {
        sd_dart_graph_split_set_threshold(std::strtoull(thresholds[i], nullptr, 10));
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt                         = "a lighthouse";
        params.negative_prompt                = "";
        params.width                          = kImageSize;
        params.height                         = kImageSize;
        params.seed                           = 7;
        params.sample_params.sample_steps     = 2;
        params.sample_params.guidance.txt_cfg = 1.0f;
        sd_image_t* images                    = nullptr;
        int generated                         = 0;
        if (!sd_dart_generate_image(context, &params, &images, &generated)) {
            status = fail("generation failed");
            break;
        }
        status = generated == 1 ? write_image(images[0], prefix + "." + std::to_string(i) + ".rgb") : 5;
        free_sd_images(images, generated);
        std::printf("RUN %d %s %llu %llu\n", i, thresholds[i],
                    static_cast<unsigned long long>(sd_dart_graph_split_threshold()),
                    static_cast<unsigned long long>(sd_dart_graph_split_cuts()));
    }
    sd_dart_exit_free(context);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "devices") == 0) {
        return list_devices();
    }
    if (argc >= 6 && std::strcmp(argv[1], "run") == 0) {
        return run(argv[2], argv[3], argv[4], argc - 5, argv + 5);
    }
    std::fprintf(stderr, "usage: %s devices | run <model> <backend> <out-prefix> <threshold>...\n", argv[0]);
    return 2;
}

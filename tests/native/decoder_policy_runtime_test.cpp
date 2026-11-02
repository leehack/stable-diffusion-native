#include "sd_dart_wrapper.h"

#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "CHECK(%s) failed at %d\n", #condition, __LINE__); return 1; } } while (0)

std::string logs() {
    std::string result;
    char text[8192];
    int32_t level;
    size_t length;
    for (uint64_t next = 0; (next = sd_dart_log_read(next, text, sizeof(text), &level, &length)) != 0;) {
        result.append(text, length);
        result += '\n';
    }
    return result;
}

int main(int argc, char** argv) {
    CHECK(argc == 5);
    const std::string mode = argv[1];
    const bool guarded = mode != "unrestricted-reload";
    const bool refuse_load = mode == "refuse-full-load" || mode == "refuse-external-load";
    sd_dart_log_enable();
    sd_dart_log_set_level(SD_LOG_INFO);
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = argv[3];
    params.backend = argv[2];
    params.n_threads = 2;
    params.flash_attn_mode = "off";
    params.conv_direct_mode = "off";
    if (mode == "refuse-external-load") params.taesd_path = argv[4];
    sd_ctx_t* context = guarded ? sd_dart_new_sd_ctx_intrinsic_tiny_v1(&params) : sd_dart_new_sd_ctx(&params);
    if (refuse_load) {
        CHECK(context == nullptr);
        CHECK(sd_dart_exit_tracked_count() == 0);
        const auto recorded = logs();
        CHECK(recorded.find("intrinsic tiny decoder policy v1 rejects") != std::string::npos);
        CHECK(recorded.find("using VAE for encoding / decoding") == std::string::npos);
        std::puts("POLICY_LOAD_REFUSED_BEFORE_RUNNERS");
        return 0;
    }
    if (context == nullptr) std::fprintf(stderr, "%s", logs().c_str());
    CHECK(context != nullptr);
    CHECK(sd_dart_exit_tracked_count() == 1);
    CHECK(std::string(sd_get_model_version_name(context)).find("SDXS") == 0);
    if (mode != "unchanged") CHECK(std::rename(argv[4], argv[3]) == 0);
    sd_img_gen_params_t request;
    sd_img_gen_params_init(&request);
    request.prompt = "a red fox";
    request.width = 256;
    request.height = 256;
    request.sample_params.sample_steps = 1;
    request.sample_params.guidance.txt_cfg = 1;
    request.seed = 42;
    sd_image_t* images = nullptr;
    int count = 0;
    const bool ok = sd_dart_generate_image(context, &request, &images, &count);
    if (mode == "refuse-reload") {
        CHECK(!ok && images == nullptr && count == 0);
        CHECK(std::string(sd_get_model_version_name(context)).find("SDXS") == 0);
        sd_vid_gen_params_t video;
        sd_vid_gen_params_init(&video);
        sd_video_generation_result_t* result = reinterpret_cast<sd_video_generation_result_t*>(1);
        CHECK(!generate_video(context, &video, &result) && result == nullptr);
        std::puts("POLICY_REFRESH_REFUSED_IMAGE_AND_VIDEO");
    } else {
        CHECK(ok && images != nullptr && count == 1);
        CHECK(images[0].width == 256 && images[0].height == 256);
        free_sd_images(images, count);
        std::printf("GENERATED %s %s\n", mode.c_str(), sd_get_model_version_name(context));
    }
    const auto recorded = logs();
    if (guarded) {
        CHECK(recorded.find("using VAE for encoding / decoding") == std::string::npos);
        CHECK(recorded.find("using TAE for encoding / decoding") != std::string::npos);
        if (mode == "refuse-reload") CHECK(recorded.find("intrinsic tiny decoder policy v1 rejects") != std::string::npos);
        if (mode == "refresh-tiny") {
            const auto first = recorded.find("using TAE for encoding / decoding");
            CHECK(recorded.find("using TAE for encoding / decoding", first + 1) != std::string::npos);
        }
    } else {
        CHECK(recorded.find("using VAE for encoding / decoding") != std::string::npos);
        CHECK(std::string(sd_get_model_version_name(context)) == "SD 2.x");
    }
    sd_dart_exit_free(context);
    CHECK(sd_dart_exit_tracked_count() == 0);
    std::puts("CONTEXT_RELEASED");
    return 0;
}

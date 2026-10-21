#include "sd_dart_wrapper.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <type_traits>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "CHECK(%s) failed\n", #condition); return 1; } } while (0)

// Every shipped target is 64-bit. Appended context members require consumers
// to regenerate bindings even when preceding members keep their offsets.
static_assert(sizeof(void*) == 8, "this probe describes the shipped 64-bit ABI");
static_assert(sizeof(sd_ctx_params_t) == 336, "context ABI changed");
static_assert(offsetof(sd_ctx_params_t, batched_cfg) == 289, "CFG ABI changed");
static_assert(offsetof(sd_ctx_params_t, linear_scale) == 292, "context ABI changed");
static_assert(offsetof(sd_ctx_params_t, tokenizer) == 304, "context ABI changed");
static_assert(offsetof(sd_ctx_params_t, conditioning_cache_size) == 316, "context ABI changed");
static_assert(offsetof(sd_ctx_params_t, flash_attn_mode) == 320, "attention mode ABI changed");
static_assert(offsetof(sd_ctx_params_t, conv_direct_mode) == 328, "convolution mode ABI changed");
static_assert(sizeof(sd_img_gen_params_t) == 552, "image ABI changed");
static_assert(sizeof(sd_vid_gen_params_t) == 584, "video ABI changed");
using Convert = bool (*)(const char*, const char*, const char*, const char*, const char*, const char*,
                         const char*, sd_type_t, const char*, bool, int, const sd_lora_t*, int);
static_assert(std::is_same<decltype(&convert_with_components), Convert>::value, "conversion ABI changed");

int main(int argc, char** argv) {
    CHECK(argc == 3);
    struct GuardedParams {
        sd_ctx_params_t params;
        unsigned char guard[32];
    } guarded;
    std::memset(&guarded, 0xA5, sizeof(guarded));
    sd_ctx_params_init(&guarded.params);
    for (unsigned char value : guarded.guard) CHECK(value == 0xA5);
    const sd_ctx_params_t& params = guarded.params;
    CHECK(params.batched_cfg);
    CHECK(params.flash_attn_mode == nullptr && params.conv_direct_mode == nullptr);
    CHECK(!params.flash_attn && !params.diffusion_flash_attn);
    CHECK(!params.diffusion_conv_direct && !params.vae_conv_direct);
    CHECK(params.conditioning_cache_size == 4);
    CHECK(!sd_ctx_supports_ref_image_conditioning(nullptr));
    const sd_preview_info_t preview = sd_get_preview_info();
    CHECK(preview.sample_pass == 0 && preview.total_steps == 0);
    CHECK(convert_with_components(argv[1], nullptr, nullptr, nullptr, nullptr, nullptr,
                                  argv[2], SD_TYPE_F16, "", false, 1, nullptr, 0));
    std::puts("runtime API defaults, bounds, capability query and conversion passed");
    return 0;
}

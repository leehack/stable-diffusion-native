# Upstream 1017 qualification

This branch moves stable-diffusion.cpp from `0518b34` (`master-982`, owner
`v0.3.0`) to `be486498eadc29dc3cd234d944c2180ed42b1415`
(`master-1017-be48649`, ggml `a5915eb0`), 35 commits. It is not released; the
platforms under "Remaining gates" are unqualified. Carries are unchanged from
[upstream 982](upstream_982_qualification.md): the Metal unbatched-CFG guard
(leejet/stable-diffusion.cpp#2149 is still open and ggml-metal is unchanged)
and `patches/pickle_line_span.patch` (`pickle_io.cpp` and `binary_io.h` are
byte-identical to `0518b34`; the patch applies). None is retired.

## Consumer ABI

Executed on macOS arm64 against both headers; bytes:

| Type / function | v0.3.0 | This branch |
| --- | ---: | ---: |
| `sd_ctx_params_t`, `sd_img_gen_params_t`, `sd_sample_params_t`, `sd_image_t`, `sd_tiling_params_t`, `sd_hires_params_t` | 336, 552, 96, 24, 32, 56 | unchanged, same offsets and initializer defaults |
| `sd_vid_gen_params_t` | 584 | **592**: appends `const char* ref_image_args` at 584, init `""` |
| `prediction_t` | `PREDICTION_COUNT` = 8 | appends `FLUX3_FLOW_PRED`, `H3_AV_FLOW_PRED`, `MING_IMAGE_FLOW_PRED`; `PREDICTION_COUNT` = **11** |
| `generate_image` | `(ctx, params, sd_image_t**, int*)` | **`(ctx, params, sd_image_generation_result_t**)`** |
| `generate_video`, `upscale`, `adetail_image` | image/frame arrays | owned result objects |
| New | | `sd_image_result_t` (32), `sd_image_generation_result_t` (16), `sd_video_generation_result_t` (40), `free_sd_image_generation_result`, `free_sd_video_generation_result`, `sd_add_rpc_servers` |
| Removed | `sd_ctx_supports_ref_image_conditioning` | gone, no replacement |
| `sd_dart_generate_image` | `(ctx, params, sd_image_t**, int*)` | **unchanged** |

Upstream [#2206](https://github.com/leejet/stable-diffusion.cpp/pull/2206)
returns each image with an owned `generation_info` JSON string. The wrapper
keeps its image-array contract: it moves the pixel buffers into an array
allocated by the library's C runtime, which `free_sd_images()` frees as
before, and frees the result with its `generation_info` (prompts, negative
prompts, model and LoRA paths). A null `images_out` frees the images. Backend
failure, upstream failure, cancellation without images and allocation failure
free the whole result and return false with both outputs cleared. The error
window, Metal fail-loud poison, call-in-flight bracketing and teardown cancel
are unchanged. `tests/native/exit_teardown_test.cpp` checks the moves and the
frees with a stand-in that returns owned results; `runtime_api_test.cpp` pins
the new signatures and layouts at compile time.

So llamadart regenerates bindings, but its calls and struct sizes stay valid:
it calls only `sd_dart_generate_image`, `free_sd_images`, the initializers and
short queries. It must not bind `sd_ctx_supports_ref_image_conditioning`
(a missing symbol), raw `generate_image`, or the enum by the old count.
`sd_ctx_params_init` writes the new `PREDICTION_COUNT`, the auto value.

## Long calls and wrappers

A Dart caller needs an `sd_dart_` wrapper for any call on a tracked object
that can outlast the 250 ms settle time.

| Call | Change here | Needs |
| --- | --- | --- |
| `generate_image` | owned result | adapted in `sd_dart_generate_image` (this branch) |
| hires-fix (`sd_img_gen_params_t.hires`), ControlNet image, reference images | inside `generate_image`/`new_sd_ctx` | covered by the existing wrappers |
| `sd_ctx_load_control_net` / `_unload_` | unchanged | a wrapper before llamadart hot-swaps ControlNet; loading it through `control_net_path` needs none |
| `new_upscaler_ctx`, `upscale` | `upscale` returns an owned result | load, run and free wrappers with tracking before llamadart#1069 |
| `generate_video` | owned result, `ref_image_args` | a wrapper before any video API |
| `new_adetailer_ctx`, `adetail_image` | owned result | wrappers before any detailer API |
| `sd_add_rpc_servers` | new | a statics-call wrapper if ever exposed: with an endpoint it loads every backend (seconds on a first Metal init). Owner builds have no RPC backend, so it fails |

## Upstream changes relevant to the 0.17 image lanes

| Consumer issue | Upstream in this range | Effect |
| --- | --- | --- |
| llamadart#1224 ControlNet | [#2187](https://github.com/leejet/stable-diffusion.cpp/pull/2187) Z-Image Union ControlNet; `sd_ctx_has_control_net` also counts ControlNet weights kept inside the diffusion runner | A component variant; SD1.5 ControlNet unchanged |
| llamadart#1228 hires-fix | no API change | unchanged |
| llamadart#1222 reference editing | [#2198](https://github.com/leejet/stable-diffusion.cpp/pull/2198) removes the capability export; [#2185](https://github.com/leejet/stable-diffusion.cpp/pull/2185) rejects Z-Image references instead of crashing; [#2207](https://github.com/leejet/stable-diffusion.cpp/pull/2207) skips unconsumed reference latents; [#2191](https://github.com/leejet/stable-diffusion.cpp/pull/2191) one placeholder token per image | the consumer needs its own capability decision (model family), not a native probe |
| llamadart#1069 upscaling | `upscale` result object | wrappers as above |
| Others | [#2223](https://github.com/leejet/stable-diffusion.cpp/pull/2223) GGUF metadata skip without repeated seeks (#2218); [#2224](https://github.com/leejet/stable-diffusion.cpp/pull/2224) FLUX.3 segmented execution (#2219); [#2181](https://github.com/leejet/stable-diffusion.cpp/pull/2181) Vulkan small-BAR heaps; [#2183](https://github.com/leejet/stable-diffusion.cpp/pull/2183) Krea2 memory recheck; [#2202](https://github.com/leejet/stable-diffusion.cpp/pull/2202) LoRA `baked_scale`; [#2180](https://github.com/leejet/stable-diffusion.cpp/pull/2180) Fibo 2.0 | source only; none is qualified here |

## FP8 E4M3 on the CPU

Upstream [#2139](https://github.com/leejet/stable-diffusion.cpp/issues/2139)
(CPU matmul on FP8 weights dereferences a null kernel) was introduced by
`2540a4f` and fixed by [#2140](https://github.com/leejet/stable-diffusion.cpp/pull/2140)
(`2891378`, master-966). `2540a4f` is in every owner pin since v0.2.0-3;
`2891378` is in v0.3.0 (`0518b34`) and this branch, not in v0.2.0-4
(`3f8527a`). #2140 only covers `Linear`: the CPU backend casts FP8 weights to
BF16 before the matmul.

Controls: SD-Turbo F16 GGUF re-encoded as FP8 E4M3 safetensors (round to
nearest even, saturating), either every 2-D weight (330 tensors, the `Linear`
path) or every weight of two or more dimensions (496, convolutions too), and
the F16 original. 128x128, 1 Euler step, seed 7, CFG 1, explicit `"off"`
modes, through `sd_dart_new_sd_ctx`/`sd_dart_generate_image` on an Apple M4
Max. Runtimes: v0.2.0-4 source plus the pickle patch (`6995083e`), v0.3.0
(`e4e1a2be`, PR #27's guarded build) and this branch (`0d52ecc2`).

| Model | Backend | v0.2.0-4 | v0.3.0 | This branch |
| --- | --- | --- | --- | --- |
| F16 | CPU | ok `c4ff6316` | ok `c4ff6316` | ok `c4ff6316` |
| F16 | MTL0 | ok `1e5b2cec` | ok `b452d0f7` | ok `b452d0f7` |
| FP8 2-D weights | CPU | **SIGSEGV** | ok `dcb146f3` | ok `dcb146f3` |
| FP8 all weights | CPU | **SIGSEGV** | ok `5ba7f744` | ok `5ba7f744` |
| FP8 2-D or all weights | MTL0 | **SIGABRT** in load | **SIGABRT** in load | **SIGABRT** in load |

The CPU crash reproduces on v0.2.0-4 and is gone from v0.3.0 on: llamadart
before 0.16 (v0.2.0-3 and -4) crashed on FP8 checkpoints on the CPU, 0.16.x
(v0.3.0) does not, and this branch produces the same pixels as v0.3.0. The FP8 images differ
from F16 (mean absolute difference 13 and 18): unscaled E4M3 weights lose
precision, so this is crash evidence, not quality.

On Metal every runtime aborts the process inside `new_sd_ctx`
(`ggml-metal-device.cpp: not implemented`, the matrix-vector pipeline has no
FP8 kernel). By the source, `ggml_metal_supports_mul_mat_op` does not exclude
FP8, so `Linear` keeps the FP8 weight instead of casting it; not confirmed
under a debugger. `wtype = SD_TYPE_F16` does not avoid it.
This is a pre-existing defect of every published Apple runtime, not caused or
fixed by this upgrade, and the default Apple backend crashes rather than
refusing. Vulkan was not run.

## Executed qualification

macOS arm64 (Apple M4 Max), release build of this branch, dylib SHA-256
prefix `0d52ecc2`, pinned Flutter 3.47.1 Dart SDK:

| Check | Result |
| --- | --- |
| `build.py build --target macos-arm64` | pass; `SD_DART_METAL=1`, carrying `pickle_line_span.patch` |
| `validate_artifacts.py macos-arm64` | pass: exports exactly the header `SD_API` set, new and removed exports included |
| `unittest discover -s tests` with `SD_REQUIRE_DART`, `SD_REQUIRE_RUNTIME`, `SD_REQUIRE_METAL_GENERATION`, 3 runs | 64 tests pass, 7 skipped (Linux-only, real model, libllamadart) |
| `test_exit_teardown_runtime.py`, ASan, Metal generation required | pass, 5 skipped |
| `smoke_test.py macos-arm64 --expect-device MTL` | pass, including the non-ASCII path conversion |
| SD-Turbo F16, CPU and MTL0, through the wrappers | byte-identical to v0.3.0 (table above) |
| `test_exit_teardown_runtime.py` with `SD_EXIT_TEARDOWN_MODEL` = SD-Turbo Q4_0 GGUF, Metal generation required, Dart VM harness | pass, 3 skipped (Linux, libllamadart) |
| Same with `SD_EXIT_TEARDOWN_SANITIZER=address` | pass, 4 skipped (also the Dart VM, which cannot load a sanitized runtime) |

ggml-metal is unchanged since v0.3.0 (the ggml delta is Vulkan small-BAR heaps
and a CUDA SageAttention guard), so the Metal GROUP_NORM defect and its guard
stand as recorded for upstream 982; the `batched-cfg` stand-in passes in both
build variants.

## Remaining gates

- Exact-head independent review.
- Consumer: regenerate bindings; drop any use of the removed export.
- Linux cooperative native/Dart teardown, release and ASan, with a built
  Linux target.
- Windows (CI smoke only), real Vulkan, every Apple slice's privacy imports,
  XCFramework, SwiftPM and signing, iOS and Android devices.
- FP8 on Vulkan and on devices; real FP8 checkpoints.
- Representative new families (Fibo, MOSS, FLUX.3) are not qualified.

# Carried upstream patches

Owner-carried changes the pinned stable-diffusion.cpp public C API cannot provide.
`CMakeLists.txt` applies each patch to build-tree copies of the upstream
sources it names and compiles the copies, so the submodule stays pristine.
Configuring fails when a patch no longer applies or changes nothing; then drop
or rebase it. Build info and the release manifest list the patches next to the
upstream commit.

Every patch needs a regression test that fails against the unpatched pin.
An upstream bug fix must link an upstream issue or pull request that defines
its retirement. A versioned owner-only context contract may instead link the
canonical consumer requirement when upstream's public C API cannot enforce
it. Keep those patches to the minimum insertion points, exercise causal
controls against real refresh/execution paths, and retire them when upstream
exposes an equivalent contract. Consumer requirements are not upstream bug
reports.

| Patch | Sources | Fixes | Test |
| --- | --- | --- | --- |
| `torch_zip_entry_lookup.patch` | `src/model_io/torch_zip_io.cpp` | Cache archive-order storage lookups so PyTorch ZIP metadata opens at most twice the archive entry count; preserve first-match names, indices and tensor bounds ([owner #24](https://github.com/leehack/stable-diffusion-native/issues/24)). **Draft: upstream retirement issue/PR link is unresolved; not merge/release ready.** Retire when upstream provides equivalent bounded lookup behavior. | `tests/test_torch_zip_lookup.py` |
| `pickle_line_span.patch` | `src/model_io/pickle_io.cpp` | A legacy PyTorch checkpoint above 2 GiB is rejected: the pickle scanner narrows the rest of the file to `int` before looking for a newline ([#23](https://github.com/leehack/stable-diffusion-native/issues/23)). | `tests/test_pickle_span.py` |
| `fp8_metal_f16_load.patch` | `src/model_manager.cpp`, `src/core/backend_fit.cpp` | An FP8 (E4M3 or E5M2) checkpoint aborts the process on Metal: ggml-metal claims FP8 `MUL_MAT` but has no FP8 kernel. Weights a Metal backend computes with now load as F16, which holds every FP8 value, so they take twice their file size; other backends keep FP8 ([#29](https://github.com/leehack/stable-diffusion-native/issues/29)). Auto-fit plans them at that size, and FP8 LoRA tensors, which crashed ggml-cpu, load as F16 on every backend ([#31](https://github.com/leehack/stable-diffusion-native/issues/31)). | `tests/test_fp8_runtime.py` |
| `lora_immediate_fallback.patch` | `src/pipeline/diffusion_engine.cpp` | The `immediately` LoRA apply mode, which `auto` selects for unquantized weights, crashes the CPU merging into FP8 weights (ggml-cpu can neither multiply nor write FP8) and aborts Metal merging into F16 weights (ggml-metal adds only equal types), which every model has: convolution weights load as F16. FP8 weights, and every model on a backend that cannot run that add outside host memory, now take the LoRA `at_runtime`, with a warning when `immediately` was requested ([#31](https://github.com/leehack/stable-diffusion-native/issues/31)). | `tests/test_fp8_runtime.py` |

| `intrinsic_tiny_decoder_policy.patch` | `src/stable-diffusion.cpp`, `src/pipeline/diffusion_engine.cpp` | Owner-only `sd_dart_new_sd_ctx_intrinsic_tiny_v1` context contract for [llamadart #1047](https://github.com/leehack/llamadart/issues/1047), not an upstream defect: automatic model refresh is valid upstream behavior. The public C API cannot preserve a decoder invariant, so minimal context creation/free and candidate/execution insertion points reject a full-VAE replacement before runner probes or decode. Real unchanged/refreshed SDXS and unrestricted reload controls exercise the boundary. Retire when upstream exposes an equivalent persistent decoder policy. | `tests/test_decoder_policy_runtime.py` |

Patches touching the same source compose in declaration order. Each next
patch applies to the preceding build-tree copy; the submodule stays pristine.

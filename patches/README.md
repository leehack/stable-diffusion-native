# Carried upstream patches

Fixes the pinned stable-diffusion.cpp lacks, carried until a pin includes them.
`CMakeLists.txt` applies each patch to build-tree copies of the upstream
sources it names and compiles the copies, so the submodule stays pristine.
Configuring fails when a patch no longer applies or changes nothing; then drop
or rebase it. Build info and the release manifest list the patches next to the
upstream commit.

A patch here needs a regression test that fails against the unpatched pin and
an upstream issue or pull request to retire it.

| Patch | Sources | Fixes | Test |
| --- | --- | --- | --- |
| `pickle_line_span.patch` | `src/model_io/pickle_io.cpp` | A legacy PyTorch checkpoint above 2 GiB is rejected: the pickle scanner narrows the rest of the file to `int` before looking for a newline ([#23](https://github.com/leehack/stable-diffusion-native/issues/23)). | `tests/test_pickle_span.py` |
| `fp8_metal_f16_load.patch` | `src/model_manager.cpp`, `src/core/backend_fit.cpp` | An FP8 (E4M3 or E5M2) checkpoint aborts the process on Metal: ggml-metal claims FP8 `MUL_MAT` but has no FP8 kernel. Weights a Metal backend computes with now load as F16, which holds every FP8 value, so they take twice their file size; other backends keep FP8 ([#29](https://github.com/leehack/stable-diffusion-native/issues/29)). Auto-fit plans them at that size, and FP8 LoRA tensors, which crashed ggml-cpu, load as F16 on every backend ([#31](https://github.com/leehack/stable-diffusion-native/issues/31)). | `tests/test_fp8_runtime.py` |
| `lora_immediate_fallback.patch` | `src/pipeline/diffusion_engine.cpp` | The `immediately` LoRA apply mode, which `auto` selects for unquantized weights, crashes the CPU merging into FP8 weights (ggml-cpu can neither multiply nor write FP8) and aborts Metal merging into F16 weights (ggml-metal adds only equal types). Those weights now take the LoRA `at_runtime`, with a warning when `immediately` was requested ([#31](https://github.com/leehack/stable-diffusion-native/issues/31)). | `tests/test_fp8_runtime.py` |

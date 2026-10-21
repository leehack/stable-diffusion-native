# Upstream 982 qualification — local preparation only

This branch prepares upstream `0518b34374761cb82aeda5ad3c2f0686fdad2cec`
(`master-982-0518b34`), with ggml `dc998771e700d49bbdf9a39cede22bb5b3311767`.
It is **not ready for publication or consumer adoption**. No source workaround
has enough evidence for retirement. The public baseline remains owner
`v0.2.0-4` / `c9922d960c79f7ba7381161ca6784f3b7c1e2fce`, upstream
`3f8527a46c54ecf4cb4ed6003da8e8982283c73c` / ggml `89c4413f`.
Live owner main and upstream master were refreshed on 2026-10-21.

## ABI and wrapper audit

The wrappers compile unchanged against the new upstream and ggml headers.
Long operations still enter `Call` inside the native wrapper; create/free
tracking remains atomic with respect to teardown. No callback, long-running
export, object type or fallback to an unguarded upstream call was added.

Executed on macOS arm64 with Apple clang; values below are bytes:

| Type/member | Published baseline | Candidate |
| --- | ---: | ---: |
| `sizeof(sd_ctx_params_t)` | 320 | 336 |
| `disable_segmented_compute` | 288 | 288 |
| `batched_cfg` | absent | 289 (previous padding) |
| `linear_scale`, `attn_scale` | 292, 296 | 292, 296 |
| `tokenizer`, `sage_attn`, `conditioning_cache_size` | 304, 312, 316 | 304, 312, 316 |
| `flash_attn_mode`, `conv_direct_mode` | absent | 320, 328 |
| `sizeof(sd_img_gen_params_t)` | 552 | 552 |
| `sizeof(sd_vid_gen_params_t)` | 584 | 584 |
| `sizeof(sd_sample_params_t)`, `sd_image_t`, `sd_lora_t` | 96, 24, 16 | 96, 24, 16 |
| `sizeof(sd_preview_info_t)` | absent | 8 |

`sd_ctx_params_init` sets `batched_cfg=true`; both mode pointers are null.
Legacy attention/convolution booleans remain false and cache size remains 4.
The added pointers make an old 320-byte Dart allocation unsafe: the candidate
initializer writes beyond it. This is a **blocking consumer ABI gate**, even
though older members keep their offsets. Bindings must be regenerated from
released headers, and the mode/batching policy must be mapped explicitly.
Null modes preserve the legacy flag route; adopting auto mode is separate.

`convert_with_components` now appends `const sd_lora_t* loras, int lora_count`.
The new runtime test compiles that exact function-pointer type and calls the
13-argument API with no LoRAs to convert synthetic Safetensors into GGUF.
`convert` retains its six-argument signature. The two new exports are
`sd_get_preview_info` and `sd_ctx_supports_ref_image_conditioning`.
The former describes a native preview callback; it does not justify adding
Dart callbacks. Video, upscaler, adetailer, control-net hot-swap and conversion
still need native lifetime wrappers before a killable Dart isolate uses them
as long operations. Neither new export creates such public Dart support.

`tests/test_runtime_api.py` checks real initializer defaults and a trailing
canary, null capability-query behavior, initial preview state and conversion
against a built runtime, without a model or GPU. It skips without that runtime;
set `SD_RUNTIME_API_TARGET` for a built Linux target. The source also pins the
64-bit layouts and changed conversion signature at compile time.

## Complete owner carry inventory

“Permanent” means part of the managed-runtime or packaging contract. “Keep”
means an upstream equivalent is absent or insufficiently qualified. Release
names describe provenance, not acceptance of a candidate on every platform.
Owner tag ancestry places page alignment in v0.1.1, SwiftPM packaging in
v0.2.0, privacy/progress/Apple teardown in v0.2.0-1, log/device tracking in
v0.2.0-2, Linux host-exit retention in v0.2.0-3 and Metal fail-loud in v0.2.0-4.

| Carry / class | Purpose, original trigger and owner provenance | Affected revisions / upstream equivalent and earliest evidence | Tests and decision |
| --- | --- | --- | --- |
| Vendored source patches | None: the owner stores a submodule, not an applied patch stack. Upstream's ggml fork is upstream content. | Both baseline and candidate submodules are clean. | **None to remove.** Do not mistake upstream `af0fc72` removing its conv1d workaround for retirement of an owner patch. |
| Ordered native progress ring / permanent | `ea53d85` (#14): Dart callbacks after shutdown aborted; callback locks deadlocked VM shutdown; latest-only polling lost batch transitions. | All owner builds since #14. Candidate still has a process-global native callback, without a managed-runtime polling/lifetime replacement. No equivalent release. | `test_progress`, `test_progress_stress`, real CPU generation. **Keep** all history, atomics and registration lifetime. |
| Native log/error ring / permanent | `6eb1bac` (#17): upstream log threads cannot safely call Dart; failed loads return null without a reason. | All owner builds since #17. Upstream callback still borrows message storage; no equivalent. | `test_log`, missing-model/exit stand-ins. **Keep** bounded error window, UTF-8 boundaries and dropped-message accounting. |
| Log spin abandonment / mitigation | #17: Windows `ExitProcess` can kill a flag holder; a descheduled reader formerly hid load errors. Separate error flag and 100 ms bound prevent indefinite waits. | Candidate has no equivalent recorder or holder-death protocol. | `test_log` plain, ASan and TSan controls; Windows execution remains unavailable. **Keep.** |
| Native context registry and guarded load/generate/free/cancel / permanent | `fd6456e` (#15), `736430e` (#16): abandoned contexts and calls must not race exit teardown. | Candidate raw `new_sd_ctx`/`generate_image` provide no managed-isolate lifetime tracking. | `test_exit_teardown`, CPU tracked-count 1→0 and double-free no-op. **Keep**, including creation/free inside one native call. |
| Apple hidden `__cxa_atexit`, static-destructor ordering / mitigation | #15: ggml-metal aborts if buffers remain when its static destructor runs. | Candidate still has Metal-owned resources/statics; no upstream ordering contract for abandoned managed objects. | Native stand-ins and `validate_artifacts` import/export closure pass. Required real Metal release/ASan teardown remains unavailable. **Keep.** |
| Bounded teardown waits and 250 ms settle / mitigation | #15/#16/#17: isolate can disappear between calls; load/generate/query cannot always cancel promptly. 15 s work / 2 s other waits, then free nothing. | No candidate equivalent or stronger cancellation bound. | `test_exit_teardown`, including in-flight/free/query/race controls. Real-model Metal phases remain unqualified. **Keep.** |
| Linux owned-static retention and cooperative host contract / mitigation | `16df759` (#18), following #949: destructors raced active load/generation. Earlier force-exit/early-flush approach was superseded to preserve host callbacks and stdio. | Candidate has no replacement host-shutdown protocol. Applies to Linux, not Android. | Stand-in host contract is retained; Linux native/Dart runtime and ASan matrix not run here. **Keep**, never reintroduce force `_exit`. |
| Linux GCC 13 floor / build mitigation | #18: pre-13 `<iostream>` injects stream initializers whose destructors owned-static retention drops. Artifact validator rejects imports. | Upstream API changes do not change this compiler/runtime constraint. | `test_tools` ELF fixtures; real Linux builds unavailable. **Keep.** |
| Static-registry device queries / permanent | #17: a query in flight crashed at exit in 2/200 trials before tracking; 0/300 after. Queries use `SdDartStaticsCall` even with no tracked context. | Candidate ggml registry is still static; no equivalent. | `test_device_memory`, `test_exit_teardown` query controls. **Keep** native wait and catch boundaries. |
| GPU build-presence/status mapping / permanent | #17: an absent backend and a compiled backend finding no device are distinct states. | Candidate ggml registry still cannot represent the build-time distinction alone. | `test_device_memory` with/without backend definitions. **Keep.** |
| Free-memory underflow/clamp / defensive mitigation | #17: Vulkan subtracts unsigned usage from budget; overuse wrapped near `SIZE_MAX`; clamp output to `[0,total]`. | ggml `65c60b239f02ab3fb35017af1e817d17a13e4952` now uses min(budget,heap), saturating subtraction. Included by SD `af0fc72` (#2147); earliest locally available containing release `master-973-f0aac52`. SD `60423ab` (master-961) reclaims before wrapped estimates; different layer. | Wrapper under/over-budget stand-ins pass. No real Vulkan comparison or artifact-level retirement proof. **Keep** cheap output invariant; upstream source overlap alone is insufficient. |
| Metal fail-loud output discard/context poison / correctness mitigation | `ea60915` (#21), owner #19: command-buffer failure could return corrupt images as success. Published v0.2.0-4. | Candidate Metal still logs the same failure/status/error-state diagnostics; no verified upstream image-result cleanup/context poison equivalent. | Existing `test_exit_teardown` controlled failures/cleanup/success and `test_log` level/overflow controls retained. CPU image success passes. Physical failure unavailable. **Keep** conservative overlap invalidation. Separate from watchdog prevention (#1047). |
| Windows short upstream build + inherited `EP_PREFIX` / build workaround | #14: Vulkan shader generator hit FTK1011 at 260-character tlog paths; short upstream path alone failed. EP prefix reduced the recorded path to 213 chars. | Candidate still nests an ExternalProject shader build. No measured default-path Windows replacement. | Source retained; Windows build unavailable. **Keep.** |
| Export allowlist / permanent | `935b70e` and #14: separate ggml copies must coexist without symbol collisions. Headers define public closure. | Candidate adds two public exports; does not remove collision risk. | Candidate macOS artifact exact export/dependency validation passes, no imported `__cxa_atexit`. Windows PE and ELF parser fixtures retained. **Keep.** |
| Windows MSVC generator / build selection | Initial pipeline: Ninja could select runner MinGW rather than MSVC. | No evidence candidate changes runner compiler selection. | Tool configuration retained, no Windows run. **Keep.** |
| Android 16 KiB alignment / platform requirement | `31bdfb7` (#7): Google Play page-size acceptance. `ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES` plus linker page size. | API28/Armv8.2 dotprod+fp16 target remains the shipped contract; no upstream update removes it. | Configuration tests retained; Android build/device unavailable. **Keep.** |
| Android CPU-only / support boundary | Initial/Vulkan pipeline #6: Adreno shader compiler and Mali null-address failures; OpenCL slower on tested Adreno. | No qualifying candidate device evidence. | No device runs. **Keep**, never infer GPU support from desktop software Vulkan. |
| Separate CPU desktop payload / packaging fallback | #6: Vulkan payload links a loader that may be absent. | Candidate Vulkan dependency still exists. | Build/dependency rules retained; target configuration tests pass. **Keep.** |
| `GGML_NATIVE=OFF`, OpenMP off, examples/WebP/WebM off / build policy | Initial pipeline: portable, self-contained library payload; no examples/video encoder dependencies. These are configuration choices, not source repairs. | No evidence justifies new runtime dependencies or host-CPU specialization. | macOS dependency validation passes. **Keep.** |
| Apple minimum OS / packaging contract | Initial/`bc8c1c2` (#11): support floors and XCFramework SwiftPM consumption. Info.plist floor comes from each binary's LC_BUILD_VERSION. | Candidate compile retains macOS13.3/iOS16.4; no hard-coded plist substitute. | Single macOS binary import/dependency check passes; full five-image XCFramework/signing/consumer not run. **Keep.** |
| Apple privacy manifest / packaging contract | `68b29da` (#13): SDK must declare required-reason APIs in each framework slice, at the correct versioned/flat location. | Existing stat/fstat/file-stamp access still present. No new categories may be guessed from upgrade success. | Manifest/packaging fixture tests run; full candidate XCFramework import audit unavailable. **Keep** file-timestamp reasons, require all slices before release. |

## Reader controls and issue boundaries

Owner [#23](https://github.com/leehack/stable-diffusion-native/issues/23)
remains reproducible. Baseline and candidate `pickle_io.cpp` and `binary_io.h`
are byte-identical. Using the native fixture from draft
[#25](https://github.com/leehack/stable-diffusion-native/pull/25), exact head
`d9bacef31e2569d5df4c3e508caf278d704280eb`:

| Control | Baseline source | Candidate source | Separate patched source copy |
| --- | --- | --- | --- |
| Valid bounded pickles / malformed and truncated pickles | pass | pass | pass |
| Malformed input in >INT_MAX and >UINT32_MAX spans | rejected | rejected | rejected |
| Same valid prefixes in those large containing spans | fail at GLOBAL | fail at GLOBAL | pass all cases |
| Bounded inputs with ASan | same source as candidate | pass | pass |

The diagnostic copy makes `find_char` accept `size_t`, caps its int-returning
scan at `INT_MAX`, and removes all 13 narrowing remaining-span casts. It is
outside the submodule and **not part of this branch's runtime**. This proves
the scanner mechanism, not a shipped fix. PR #25 remains test-only with its
expected failure; no duplicate test PR or upstream submission was made.
The original trusted >2 GiB real-checkpoint/public-Dart evidence is in #23;
this run does not repeat the huge-checkpoint conversion or loading.

Owner [#24](https://github.com/leehack/stable-diffusion-native/issues/24)
remains unresolved: its original full ZIP load timed out at 45 s. Later
non-reproduction reported in #25 does not identify a causal fix or reconcile
that input/environment. No large ZIPs were created/downloaded here, and the
scanner patch must not be presented as fixing ZIP loading.

## Executed qualification and remaining gates

Built the baseline and candidate macOS-arm64 libraries using the owner build
tool; relocated ccache into the worktree after the sandbox refused its usual
home-directory cache. Candidate `validate_artifacts.py macos-arm64` passes.
No production wrapper source change was needed.

The checksum-verified published v0.2.0-4 archive and dylib are compared
separately from the source rebuild. The archive SHA-256 is
`840df6e27e1cac6d378dd5529f7c3b8d3b176417f950ef523c3535c09c92f528`;
the dylib SHA-256 is
`79ed1ab7c0451596bcb18070d459826a8b04409875873e91f39ffd84b2ffe1ce`.

The guarded native comparison uses existing SD-Turbo F16 GGUF + full TAESD,
CPU/four threads, 128×128, one Euler/discrete step, seed7, CFG2, prompt
`a lighthouse`. It verifies image/count/dimensions, writes raw RGB, frees via
`sd_dart_exit_free`, checks registry count1→0 and double-free no-op. Candidate
explicit mode controls use attention on/convolution off; an explicit
unbatched route and a batched CPU route are compared. This is a correctness
smoke, not a performance benchmark or new-family qualification. All four
RGB controls have SHA-256
`bf06aed2310655953e6a403c264db320ac23aaf407e8f8e67fc7853a05839c86`.
The candidate's null-mode/legacy-flag route was also compared separately.

The selected suite ran 48 tests: 44 passed, three Linux-only import/stdio
checks skipped on macOS, and the newly written conversion probe failed
because it supplied a null `tensor_type_rules` string. The existing smoke
recipe passes an empty string; the corrected probe now does so and passes
its targeted rerun (one test). The independent auditor reproduced the same
invalid-null failure on the baseline, so this was a harness error, not a
candidate regression. The combined completed evidence is 45 passing tests
and three explicit skips; the full suite was not repeated after that
one-argument test correction. Existing coverage includes native ASan/TSan
stand-ins and the pinned-Dart progress/shutdown stress controls. The native
model-dependent exit runtime harness was deliberately excluded.

An initial whole-suite attempt used the default Flutter wrapper instead of
the pinned Dart SDK and auto-detected the library built during the run; it
was stopped and is not qualification evidence. All replacement tests used
the pinned Flutter 3.47.1 Dart binary. No remaining owned test jobs are left
running at the freeze.

**Required before adoption:** exact-head independent review; regenerate and
map consumer bindings/default policy; obtain Metal GROUP_NORM batch1/batch2
sentinels and real batched/unbatched image controls; run release and ASan
Metal teardown and Linux cooperative native/Dart teardown; qualify real
Vulkan memory/performance; audit every Apple slice's privacy/minimum-OS/
XCFramework/SwiftPM/signing; complete platform packaging/export checks.

Upstream [#2149](https://github.com/leejet/stable-diffusion.cpp/issues/2149)
is still open. Candidate Metal GROUP_NORM still dispatches `ngrp,1,1` and
passes dimensions0–2 only; its normalization kernel is unchanged from the
baseline. This source evidence plus the upstream reproduction means the new
`batched_cfg=true` default cannot be claimed supported on Metal. CPU parity
does not resolve that gate. FP8 CPU #2139/#2140 (first master-966) and Vulkan
convolution #2137/#2141 (first master-967) have upstream repair provenance,
but representative FP8/real-Vulkan runtime qualification was unavailable.
No unrelated model substitutes for those acceptance checks.

No consumer pins, published artifacts, tags, workflows, issues or PRs were
changed. All existing primary/active investigation checkouts were preserved.

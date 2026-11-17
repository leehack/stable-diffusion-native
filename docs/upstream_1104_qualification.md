# Master-1104 source candidate

Owner [#20](https://github.com/leehack/stable-diffusion-native/issues/20),
coordinated by [core #2024](https://github.com/leehack/llamadart/issues/2024).
This is a high-risk draft candidate, not release qualification. Proposed owner
artifact **v0.4.0** is distinct from core 0.20.0; no tag or publication is made.

| Layer | Baseline | Candidate |
| --- | --- | --- |
| Owner base | `0ec6c1fa1e60d5124413b6c14b8fe510f112567d` (v0.3.0-6) | PR head, recorded in the PR and external receipts |
| stable-diffusion.cpp | `be486498eadc29dc3cd234d944c2180ed42b1415` (master-1017) | `5dcfa79801213f23e27b3e83617350ebb8ea93aa` (published master-1104-5dcfa79) |
| ggml | `a5915eb0e47adef21c3f25b889a008925ea838c2` | `eeda9ec48fc84a3df0bb9dcc19ae9053075a2539` |

No newer opaque API (#2365), new family, Union, interleaved conditioning,
video/upscaling API or consumer adoption is included. The candidate's additional
raw exports do not establish Dart capability. No Dart callback may be registered;
new long calls on tracked objects need native lifecycle wrappers before use.

## ABI and execution boundary

Compiled macOS arm64 old/candidate header probes confirm context **336**, image
**552 → 560**, video **592** bytes. The built candidate's initializer canaries
confirm `control_type` at **552**, `SD_CONTROL_TYPE_NONE == 0`, default NONE,
512×512 defaults and intact bytes beyond the new allocation. The old allocation's
first four trailing bytes are overwritten. Consumers must regenerate bindings
from released headers and enforce compatible companions before adoption.
`sd_dart_generate_image` retains its image-array signature and ownership adapter;
raw generation still returns an owned result. Native callback setters are not
new wrapper APIs. Explicit flash/conv mode mapping and Metal `batched_cfg=false`
remain mandatory. No long tracked-context call is changed to an unguarded call.

The candidate routes backend evaluation callbacks through the scheduler in
`GraphExecution::execute_segment`. The native graph splitter stays installed
once, with no allocation or registry access in its callback, iOS Metal default
2^26 and zero elsewhere. CPU fixture controls exercise off/low/off/low/high,
increasing cuts, identical nonflat pixels, cancellation with cleared outputs,
and recovery with identical pixels. This proves CPU mechanics through the new
scheduler route, not physical-iPhone watchdog behavior.

## Eight patch decisions

All rows use the exact revisions above. None is retired. Original evidence
and source/issue links are preserved in [patch documentation](../patches/README.md),
the [982 record](upstream_982_qualification.md), [1017 record](upstream_1017_qualification.md)
and the external audit matrix. Historical execution is not candidate execution.

| Carry / origin | Original trigger; candidate status | Decision and proof / gap |
| --- | --- | --- |
| `pickle_line_span.patch`; owner #23/#26, `aa1f6d62` | macOS CPU legacy SD-Turbo pickle >2 GiB; candidate still narrows spans | Retain unchanged. Candidate model-free >INT_MAX/>UINT32_MAX, malformed and ASan tests include unpatched negatives. Trusted 2,612,843,011-byte native checkpoint acceptance remains pending; original report's 2,612,865,076-byte file is a different fixture. |
| `torch_zip_entry_lookup.patch`; #24/#41, `5fde7dd1` | macOS CPU flat/nested PyTorch ZIP metadata repeatedly scans archive order; candidate still does | Retain unchanged. Candidate bounded operation-count, first-match/shared-storage/lifetime, cleanup, sanitizer and semantic-mutation controls. Original model load/output and timing bounds remain pending. Existing #24 upstream-report exception remains unchanged. |
| `fp8_metal_f16_load.patch`; #29/#30/#31/#32, `16949160`/`54e97df3` | SD-Turbo E4M3/E5M2 Metal load abort, CPU FP8 LoRA crash, effective-F16 fit | Retain unchanged. Candidate rejects E4M3 matmul only; E5M2, all-backend LoRA, fit, strides/layout/value contracts remain. Source assertion explicitly binds this partial overlap to eeda9ec4. Generated CPU FP8/value and LoRA controls pass; actual Metal/fit and real checkpoints remain pending. |
| `lora_immediate_fallback.patch`; #31/#32, `54e97df3` | CPU immediate FP8 merge crash; Metal F16 += F32 abort on preallocated weights | Rebase contexts, preserving new packed-INT4 detection/warning and existing fallback. Generated CPU mode/effect and expected-type controls pass. Actual Metal mixed ADD, multi-device/module paths and #33 TAE nonapplication remain separate gaps. |
| `intrinsic_tiny_decoder_policy.patch`; consumer #1047, owner #38, `1c1f03bb` | SDXS source refresh could replace the persistent tiny decoder before execution | Rebase creation context around new raw callback setters; preserve pre-runner candidate refusal, execution checks, cleanup and unrestricted path. No upstream equivalent contract. Real unchanged/refreshed SDXS-512-DS/09 CPU/Metal controls remain pending; no unrelated fixture substitutes for them. |
| `loader_scratch_vm_allocate.patch`; consumer #2004, owner #45/#46, `0ec6c1fa` | iOS18 loader read/convert/ZIP/convert_tensor heap scratch leaves ~240 MiB cached | Retain unchanged. Generated F32/BF16/ZIP CPU fixtures pass. M3 (only ZIP scratch back to vector) fails at 2,121,760-byte worker allocation; M5 (remove vm_deallocate) fails allocation/free balance in all three fixtures. Restored binary matches positive SHA. Test-only dyld observer checks actual addresses/sizes and immediate virtual unmapping. This is not iOS physical reclamation. Existing #45 exception remains unchanged. |
| `ggml_apple_vm_allocate.patch`; consumer #2004, owner #44, `c9ff9f5c`, upstream llama.cpp #31437 | iOS CPU host buffers remain in app footprint after free | Retirement candidate only. Exact eeda9ec4 already has Apple-wide VM allocate/free with TSAN exception. Explicit revision+source-hash CMake path selects upstream experimentally; retain patch and old source apply/reverse controls. Original per-carry iOS three-way proof is missing. |
| `ggml_metal_apple_vm_allocate.patch`; same origin | iOS Metal shared/owned host buffers remain in footprint | Retirement candidate only. Exact upstream preserves shared+owned VM deallocation and borrowed-buffer ownership. Same explicit source selection, historical patch and controls. Per-carry original iOS Metal proof is missing. |

The two allocator patches cannot be duplicated onto fixed code or applied as
no-ops. `AppleAllocatorSelection.cmake` permits only reviewed a5915eb0 backports
or eeda9ec4 upstream files; unknown revisions and changed bytes fail closed.
Old patch application/reversal and candidate deallocation-source mutations are
tested. This provisional selection is not a new patch-policy exception or
approval to declare either carry obsolete. `upstream.patches` is the retained
inventory; per-target `patchApplication` records actual six-patch compilation
and `upstream-experimental` allocator source.

## Twenty-eight integration protections

Every row is retained; no upstream behavior is credited as equivalent without
its original-trigger proof. Origins refer to this owner unless marked consumer
or upstream. Stand-ins and packaging fixtures are local proof only.

| Contract / origin | Preserved behavior and remaining acceptance |
| --- | --- |
| Ordered progress history; #14 `ea53d85` | 4095-report ordered ring, sequence/overflow, coherent copies, bounded reads, one native registration, no Dart callbacks; progress/stress controls retained. |
| Native log/error history; #17 `6eb1bac` | UTF-8, drops, per-call error windows, no callback allocation/exit registry, retained statics; log/exit controls. |
| Log flag abandonment; #17 | Bounded 100ms spin for a Windows holder killed by ExitProcess; stand-in passes, real Windows remains a gate. |
| Tracked context calls; #15 `fd6456e`, #16 `736430e` | Atomic create+track/free+untrack and in-flight long-call guards, cancellation/double-free/race tests; runtime host matrix pending. |
| Apple exit static order; #15 | Hidden owned __cxa_atexit, free contexts before Metal statics; macOS import guard passes, real Metal abort-control and runtime ASan pending. |
| Bounded exit wait; #15/#16/#17 | 15s for load/generation/query, 2s others, 250ms settle; timeout frees nothing; native stand-ins retained. |
| Linux host exit retention; #18 `16df759`, consumer #949/#977 | Retain owned statics, stop/join host workers and await isolate/finalizer completion; preserve libc flush/callbacks; Linux native/Dart release+ASan pending. |
| Linux GCC floor; #18 | GCC≥13, no imported iostream initializer or __cxa_atexit; Android unchanged; fixture checks retained, Linux artifact pending. |
| Statics query guard; #17 | SdDartStaticsCall even at zero contexts, exception boundaries, teardown waits; device/exit stand-ins. |
| Build-presence status; #17 | GPU-absent differs from built-with-no-device; generated SD_DART_GPU_BACKEND/SD_DART_METAL checks retained. |
| Memory clamp; #17 | Keep [0,total] under/over-budget invariant despite upstream saturation overlap; real GPU comparison pending. |
| Metal fail-loud poison; #19/#21 `ea609150` | Discard failed output, clear outputs, poison context and refuse reuse; controlled failures retained; actual watchdog-failure route pending. |
| Metal unbatched CFG; #27 `992d3f70`, upstream #2149 | Preserve false in wrapper copy even with CPU in a Metal build, caller unchanged. Candidate upstream #2354 GROUP_NORM repair does not retire guard; asymmetric batch sentinels/real images pending. |
| VRAM precreate refusal; #34/#35 `82a9a4b3` | Budget needs auto_fit, no params_backend, explicit backend on integrated-only default; reject before raw create; affected-device controls pending. |
| Owned result adapter; #28 `f66e56c9`, upstream #2206 | Move pixels, free metadata/results once, null/failure/cancel/allocation paths clear outputs, free_sd_images contract; native stand-ins and candidate API probe retained. |
| iOS graph split; #42/#43 `4a84ac24`, consumer #1047 | Native threshold/cut ABI, once-only callback, iOS 2^26 default; candidate CPU scheduler/cancel/recovery controls, physical full-VAE matrix pending. |
| Explicit modes; #27/#28 | Null modes select auto probes; preserve explicit consumer flash/conv mapping and old harness fields; real same-model output comparison pending. |
| Windows paths; #14 | Short sd binary dir and inherited EP_PREFIX; actual Windows compiler path/build pending. |
| Windows generator; initial pipeline | MSVC rather than accidental MinGW/Ninja selection; unchanged build policy, Windows CI pending. |
| Export allowlist; `935b70e`/#14 | Exactly both shipped headers, hide ggml/internals, system dependency allowlist; macOS passes, other targets and libllamadart coexistence pending. |
| Android pages/ISA; #7 `31bdfb7` | 16KiB linker/page support, API28, Armv8.2 dotprod+fp16/asimddp consumer probe; artifact/device gates pending. |
| Android CPU only; #6 | Preserve support boundary after Adreno/Mali Vulkan crashes and slower OpenCL; no candidate GPU claim. |
| Desktop CPU payload; #6 | Separate CPU runtime when Vulkan loader is absent; target/dependency policies retained. |
| Portable flags; initial pipeline | GGML_NATIVE/OpenMP/examples/WebP/WebM off; no new dependencies or accelerators. |
| Apple XCFramework floors; #11 `bc8c1c2` | macOS13.3/iOS16.4; plist floors from LC_BUILD_VERSION; all slices/SwiftPM/signing pending. |
| Apple privacy; #13 `68b29da` | Timestamp reasons and correct resource location; fixture audit retained; every candidate slice/architecture needs actual imports audit. Never guess categories. |
| Vulkan triangle guard; #40 `e6dceba8` | Preserve strict 6352/6388-byte layouts, positive-zero dataflow, unique actual module, SHA/bytes capture, archive recheck and strict names. Candidate module unqualified; no fingerprint refresh. |
| Linux producer; #36/#37 | Ubuntu24.04 unchanged, LoRA MAD≤0.5. Ubuntu26.04 x64 ~1.37 cause unresolved; software Vulkan/arm64 cannot clear it. |

## Local receipts and limits

Logs, mutation hashes and the read-only 8/28 audit live outside the repository:
`/opt/UnitySrc/personal/llama/llamadart-evidence/planning-next-release/owner-candidates/stable-diffusion-native/`.
The `implementation/` directory holds build, export/import validation, native
stand-ins/sanitizers, progress/Dart stress, source/ABI controls, generated CPU
FP8/LoRA/graph/loader and CPU exit receipts. Test discovery was inspected before
execution; runtime suites were selected explicitly to avoid GPU/model probes.
Skipped or unrun tests are not qualification. Exact results belong in the PR
and implementation receipt, not historical qualification tables.

Reproduction of authorized local checks:

```sh
python3 tools/build.py build --target macos-arm64 --jobs 4
python3 tools/validate_artifacts.py macos-arm64
PYTHONPATH=tests python3 -m unittest test_upstream_refresh test_runtime_api
PYTHONPATH=tests python3 -m unittest \
  test_loader_scratch_runtime.LoaderScratchRuntimeTest.test_cpu_load_takes_no_tensor_scratch_from_the_heap \
  test_graph_split_runtime.GraphSplitRuntimeTest.test_split_cuts_and_keeps_the_image_on_the_cpu \
  test_fp8_runtime.Fp8RuntimeTest.test_fp8_generates_on_the_cpu \
  test_fp8_runtime.LoraRuntimeTest.test_lora_applies_on_the_cpu
python3 tools/check_loader_scratch_mutations.py --build-dir build/macos-arm64 \
  --evidence /absolute/external/evidence/loader-mutations
```

The mutation command changes only the configured build-tree loader copy, never
vendored sources or bin/ payloads; it restores/rebuilds the positive source in
`finally`, requires the expected causal assertion, and records exact binary and
source hashes. Use an idle build directory. Observer mapping release is narrower
than physical footprint, and does not certify OS reclamation on iOS.

## Pending original-trigger and release matrix

Do not execute this section without the separate model/device/run approval.
For each allocator carry individually, hold every other current-old fix constant:

1. **Old negative:** owner base/source above with only that carry omitted.
2. **Old positive:** same owner base/source with that carry applied.
3. **Candidate unpatched for that carry:** exact master-1104/ggml above, all six
   retained patches and integration protections fixed; reviewed upstream allocator
   source selected. Failure of the negative to reproduce means unresolved proof.
4. Independently qualify the final combined candidate and bind it to exact
   artifact bytes, owner head, build source/patch selection and toolchain.

Compile each native test harness against that variant's own shipped headers.
Do not repoint the historical v6 Dart harness (552-byte image allocation) at
the new initializer. A Flutter/Dart run first needs the parent-owned 560-byte
binding/harness migration, preserving recipe fields and host shutdown behavior.
The historical harness source is `6b191b28b0f39eae1368fc76ea37f8051c5d2cb8`;
its settings and artifact receipts remain the baseline, not a safe new ABI.

On **iPhone16Pro/iOS18.3.2 (22D82)** and **iPhone17Pro/iOS26.5.1 (23F81)**:
touch CPU and actual MTL0 host buffers (including shared+owned and borrowed
non-owned branches), record imports, mappings, physical footprint before/after
free, and TSAN branch ownership semantics. Then use the preserved full-VAE
SD-Turbo recipe: model **2,023,745,376 bytes**, SHA256
`d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580`,
no external TAESD, **512×512, one step, CFG1, seed42**, prompt
`a lighthouse on a cliff at sunset`, MTL0, explicit original mode fields,
four loads/eight decodes per variant/device. Capture immediate/+1/+5/+30s disposal
footprint separately from epilogue churn, outputs, actual backend and failures.
Keep old failing variants explicit; the historical v4 iOS26 load failure remains
a failure. Historical v6 residuals are nonzero and do not isolate either carry.
No memory tolerance or zero-leak claim is invented.

For loader scratch, add original iOS18 repeated lifecycle and iOS26 nonregression;
keep loaded_data/load_tensor_parts vectors outside this carry's scope. Allocation
failure/growth ownership fault injection and physical footprint remain separate
from the observed call/unmapping controls. For FP8/LoRA renew original E4M3 and
E5M2 base/all-weight and FP8-down adapters, CPU/Metal, F16/plain controls,
effective-F16 fit, strides/layouts/values and actual images; do not remove runtime
proof because E4M3 is rejected upstream. Renew trusted legacy/flat+nested ZIP,
SDXS source-refresh refusal and existing-family/hires/FLUX controls.

After separate approval, run release and AddressSanitizer native/Dart shutdown
matrices on actual Metal (require its untracked abort control and
`SD_REQUIRE_METAL_GENERATION=1`) and Linux (`SD_EXIT_TEARDOWN_TARGET`, both ways).
Keep actual-model and libllamadart coexistence controls. Validate every target,
all Apple privacy/imports/floors, XCFramework validate/consumer, Windows and
Android. Vulkan requires original producer identity, unchanged MAD0.5 threshold,
actual reviewed module and physical GPU output; new fingerprints require source
and module proof with negative controls. Ordinary PR CI is additional evidence.
Fresh independent exact-head/current-base review, readiness, merge, artifact
preparation/publication and downstream bindings/pins/checksums/SwiftPM each
remain parent-owned approval gates. Publish only through native_release.yml.

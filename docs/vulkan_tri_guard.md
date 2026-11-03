# Vulkan triangle artifact guard

Some GLSL compilers turn GGML's `tri.comp` `D_TYPE(0)` into a NaN in the
`tri_f16` module. Two builds of identical sources can differ. The owner source
and upstream submodule remain unchanged; compilation is allowed to finish,
then the exact generated and embedded bytes must pass the guard before release.
A rejected build is not qualified by a passing retry or by another target.

`tools/vulkan_tri.py` admits the reviewed 6352-byte and 6388-byte module
layouts. Identification normalizes only a float32 constant narrowed to half;
validation examines the original bytes. It requires one triangle mode switch,
the accepted branch's input-to-output copy, and the rejected branch's float32
positive zero converted to half and stored to output binding 1. NaN, finite
nonzero, negative zero, missing or ambiguous modules, changed layouts and a
missing output data path fail closed. This is a narrow proof for this operation,
not a general shader validator or GPU execution test.

New builds capture the actual generated `tri_f16.spv` in `build-info.json` as
`vulkanTriF16` schema 1, with its SHA-256 and hex bytes. Build, runtime validation
and packaging require this capture to match the unique module in the runtime.
Package verification and publication inspect the actual library inside every
Vulkan runtime archive again, independently of accelerator flags supplied by
that archive. Existing source, manifest, archive digest and target-inventory
checks remain in force. Runtime archives reject noncanonical names, normalized
or case-fold aliases, links and nonregular members before any byte selection.
Canonical directory entries remain supported, so ordinary extraction cannot replace a validated library through another name.

Prepared artifacts from before this field existed use an explicit legacy path:
the verifier identifies and proves the same unique known module directly in
the archive's runtime bytes. A present null, malformed or unsupported-schema
field is rejected. The legacy path supplies no invented build capture or owner
commit; its trusted prepared-run source and archive hashes remain unchanged.
It has the same module identity and positive-zero proof as new artifacts.

A compiler or upstream update that changes the layout needs a fresh independent
source/module review and negative controls before adding an identity. Do not
refresh fingerprints merely to make a failed candidate pass. A source/compiler
workaround can be carried only under `patches/README.md`, including its required
upstream retirement link; this guard does not claim such a fix has landed.

`tests/test_vulkan_tri.py` covers module, flow, layout and metadata failures.
`tests/test_vulkan_tri_packaging.py` exercises the real build, artifact validator,
package writer and prepared replay entry points using deterministic compiled
fixtures. Heavy device and model qualification remains separate.

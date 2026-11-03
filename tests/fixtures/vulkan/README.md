`tri_f16.spv` is the 6352-byte compiled positive-zero module from GGML
`a5915eb0e47adef21c3f25b889a008925ea838c2`, with `tri.comp`'s `D_TYPE(0)`
replaced by `D_TYPE(0.0)`. SHA-256:
`a8691344947943aa581022ecfc95953c02ea3247f85abd0d2f46a23ad49cbe0f`.

Reproduce with shaderc/glslc using `-fshader-stage=compute
--target-env=vulkan1.2 -O -DA_TYPE=float16_t -DD_TYPE=float16_t`, including
GGML's `vulkan-shaders` directory. The regression suite compiles the actual
pinned source when glslc is available. Deterministic mutations replace its
rejected-branch float32 zero constant with NaN/nonzero or remove its use;
they exercise the validator on every platform without a GPU or SDK.

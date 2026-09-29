# AGENTS.md

Guidance for coding agents working in `stable-diffusion-native`.

## Scope

- This repository builds and releases stable-diffusion.cpp runtime libraries
  consumed by `llamadart`. The Dart API and model handling live in `llamadart`.
- Upstream is the `third_party/stable-diffusion.cpp` submodule. Update the
  submodule pin instead of patching vendored sources; report upstream bugs
  there.

## Commands

```bash
git submodule update --init --recursive
python3 -m unittest discover -s tests
python3 tools/build.py build --target <target>
python3 tools/validate_artifacts.py <target>
```

- Every shipped library must pass `validate_artifacts.py`: it exports exactly
  the `SD_API` symbols in `stable-diffusion.h` and links only allowlisted
  system libraries. Never widen the export list to fix a consumer; a leaked
  ggml symbol can collide with llama.cpp's ggml in the same process.
- A new target or backend needs device evidence before release, recorded in
  the PR.

## Releases

- Publish only through `.github/workflows/native_release.yml`.
- Tags are `vMAJOR.MINOR.PATCH` with an optional `-N` rebuild counter starting
  at 1, without leading zeros.

## Handoff to `llamadart`

After a release, update the `llamadart` runtime pin and archive checksums, and
regenerate its bindings from the released `stable-diffusion.h`, in the same
`llamadart` PR.

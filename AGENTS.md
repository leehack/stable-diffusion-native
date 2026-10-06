# AGENTS.md

Guidance for coding agents working in `stable-diffusion-native`.

## Scope

- This repository builds and releases stable-diffusion.cpp runtime libraries
  consumed by `llamadart`. The Dart API and model handling live in `llamadart`.
- Upstream is the `third_party/stable-diffusion.cpp` submodule. Update the
  submodule pin instead of patching vendored sources; report upstream bugs
  there.
- What upstream's C API lacks goes in `src/`, which the top-level
  `CMakeLists.txt` compiles into the same library. A wrapper export is an
  `sd_dart_*` function declared with `SD_API` in `src/sd_dart_wrapper.h`, with
  a test under `tests/` and a README entry.

## Commands

```bash
git submodule update --init --recursive
python3 -m unittest discover -s tests
python3 tools/build.py build --target <target>
python3 tools/validate_artifacts.py <target>
```

- Every shipped library must pass `validate_artifacts.py`: it exports exactly
  the `SD_API` symbols in `stable-diffusion.h` and `src/sd_dart_wrapper.h`, and
  links only allowlisted system libraries. Never export an upstream-internal
  or ggml symbol to fix a consumer; a leaked ggml symbol can collide with
  llama.cpp's ggml in the same process.
- The Apple XCFramework must pass `apple_xcframework.py validate` and
  `consumer`. Its Info.plist minimum OS is read from each binary's
  `LC_BUILD_VERSION`; never hard-code it, since a mismatch fails App Store
  upload.
- Each XCFramework slice embeds `tools/apple/PrivacyInfo.xcprivacy`. When the
  import audit fails after an upstream bump, find the new call sites and update
  the manifest and `docs/apple_privacy_manifest.md` together; never add a
  category only to make it pass.
- A new target or backend needs device evidence before release, recorded in
  the PR.

## Releases

- Publish only through `.github/workflows/native_release.yml`.
- Tags are `vMAJOR.MINOR.PATCH` with an optional `-N` rebuild counter starting
  at 1, without leading zeros.

## Handoff to `llamadart`

After a release, update the `llamadart` runtime pin and archive checksums, the
Apple companion `Package.swift` tag and XCFramework checksum, and regenerate
its bindings from the released `stable-diffusion.h` and `sd_dart_wrapper.h`,
in the same `llamadart` PR.

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
- A wrapper export never takes a callback that stable-diffusion.cpp's threads
  would call, and never waits for such a thread: a Dart callback there aborts
  or deadlocks VM shutdown. Record state natively and let the caller poll.
  Wrapper state has no destructor; threads still use it while `exit()` runs.
  The one wait is exit teardown's, which is bounded and frees nothing when it
  runs out: 15 s while a load, a generation or a device query is in flight,
  2 s for any other call, so quitting during a large generation can delay the
  exit by up to 15 s.
- An export that reads the library's statics without a tracked object, as a
  device query reads ggml's registry, is an `SdDartStaticsCall`: teardown
  returns at once when nothing is tracked, unless such a call is in flight.
- What the library runs inside upstream's log or progress callback allocates
  nothing, never reaches the exit registry and holds its flag only for the
  copy of one record: it runs on upstream's threads until the process is gone.
  The log recorder also gives up on a flag whose holder died, as `ExitProcess`
  lets one on Windows, because whatever is destroyed afterwards may still log.
- Whatever a Dart caller does to a tracked object begins and ends in one
  native call: tracking in the call that creates it, untracking in the call
  that frees it, and a call in flight around anything that can outlast the
  250 ms settle time. An isolate can be killed between any two native calls.
  Give such an upstream function an `sd_dart_` wrapper before `llamadart`
  calls it on a tracked context.
- Whatever a caller polls must keep its history. Consumers derive state from
  the order of progress reports, such as which image of a batch is being
  sampled, so a "latest value" export loses what happened between two polls.

## Commands

```bash
git submodule update --init --recursive
python3 -m unittest discover -s tests
python3 tools/build.py build --target <target>
python3 tools/validate_artifacts.py <target>
```

- `tests/test_progress_stress.py` needs a Dart SDK on `PATH` and skips without
  one; CI runs it with 25 runs per configuration. Run it after any change to
  `src/`.
- `tests/test_exit_teardown_runtime.py` needs the built macOS runtime and
  skips without it. After a change to `src/sd_dart_exit.cpp` or an upstream
  bump, run it against the release build and again with
  `SD_EXIT_TEARDOWN_SANITIZER=address`, on a Mac where its control reports the
  Metal abort, with `SD_REQUIRE_METAL_GENERATION=1`: GitHub's runners reach
  neither the abort nor a generation on Metal.
- Every shipped library must pass `validate_artifacts.py`: it exports exactly
  the `SD_API` symbols in `stable-diffusion.h` and `src/sd_dart_wrapper.h`, and
  links only allowlisted system libraries. Never export an upstream-internal
  or ggml symbol to fix a consumer; a leaked ggml symbol can collide with
  llama.cpp's ggml in the same process.
- An Apple library must not import `__cxa_atexit`: exit teardown depends on
  the hidden definition in `src/sd_dart_exit.cpp` receiving every static
  destructor of the image. Keep that definition hidden; libllamadart has its
  own.
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

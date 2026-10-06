# stable-diffusion-native

Native build and release pipeline for
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) runtime
libraries.

## Purpose

This repository owns the platform-specific stable-diffusion.cpp payload
consumed by [llamadart](https://github.com/leehack/llamadart) and other
wrappers. It stays independent from `llamadart` so the artifacts can be reused
elsewhere.

- Pin an upstream stable-diffusion.cpp commit as a submodule.
- Build one shared library per platform with upstream's C API
  (`stable-diffusion.h`) as the FFI boundary.
- Export only that API and the [wrapper API](#wrapper-api) this repository
  adds. stable-diffusion.cpp embeds its own patched ggml; hiding it lets the
  library share a process with another ggml, such as llama.cpp's.
- Publish runtime archives, unstripped symbol archives, an Apple SwiftPM
  XCFramework, `manifest.json` and `SHA256SUMS`.

The Dart API, model presets and download/cache logic stay in downstream
packages such as `llamadart`.

## Targets

| Target | Library | Acceleration | Built on |
| --- | --- | --- | --- |
| `macos-arm64` | `libstable-diffusion.dylib` | Metal, CPU | macOS |
| `macos-x64` | `libstable-diffusion.dylib` | Metal, CPU | macOS |
| `ios-arm64` | `libstable-diffusion.dylib` | Metal, CPU | macOS |
| `ios-arm64-sim` | `libstable-diffusion.dylib` | Metal, CPU | macOS |
| `ios-x64-sim` | `libstable-diffusion.dylib` | Metal, CPU | macOS |
| `android-arm64` | `libstable-diffusion.so` | CPU | any, with the NDK |
| `linux-x64` | `libstable-diffusion.so` | CPU | Linux x64 |
| `linux-x64-vulkan` | `libstable-diffusion.so` | Vulkan, CPU | Linux x64 |
| `linux-arm64` | `libstable-diffusion.so` | CPU | Linux arm64 |
| `linux-arm64-vulkan` | `libstable-diffusion.so` | Vulkan, CPU | Linux arm64 |
| `windows-x64` | `stable-diffusion.dll` | CPU | Windows x64 |
| `windows-x64-vulkan` | `stable-diffusion.dll` | Vulkan, CPU | Windows x64 |

Minimums: macOS 13.3, iOS 16.4, Android API 28. The Android build requires
Armv8.2 dot-product and fp16 support (Cortex-A55/A75 or newer); consumers
should check for `asimddp` before loading it.

Vulkan variants link the Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`) and
fail to load without it; consumers fall back to the CPU variant. CI checks them
on Mesa lavapipe only, so GPU performance still needs hardware runs.

Android GPU backends are not shipped. On tested devices, Vulkan crashed in
the Adreno 750 driver's shader compiler and on a null `vkGetBufferDeviceAddress`
on Mali-G68, and OpenCL on Adreno 750 was slower than the CPU.

## Wrapper API

`src/sd_dart_wrapper.h` declares the exports this repository adds to
upstream's. It is compiled into the same library and ships next to
`stable-diffusion.h` in every archive and XCFramework slice.

```c
typedef struct {
    uint64_t sequence;  // reports recorded since sd_dart_progress_enable(); 0 = none
    int32_t step;       // the arguments of upstream's progress callback
    int32_t steps;
    float time;
} sd_dart_progress_t;

void sd_dart_progress_enable(void);
void sd_dart_progress_read(sd_dart_progress_t* progress);
```

| Function | Behavior |
| --- | --- |
| `sd_dart_progress_enable` | Records progress in the library instead of printing progress bars to stdout, for every later call in the process, model loads included. Idempotent. |
| `sd_dart_progress_read` | Copies the latest report into `progress`. Callable from any thread at any time, also before `sd_dart_progress_enable`. |

Upstream's `sd_set_progress_callback` calls back on the thread that is loading
or generating, for as long as that call runs. A callback into a managed
runtime cannot be made safe there. A Dart `NativeCallable` is called after
its isolate or the VM has shut down, which aborts the process, and a wrapper
that waits for a call in flight before dropping the callback can deadlock VM
shutdown. So the library takes no callback: it records progress itself, and
the caller polls.

- `sd_dart_progress_read` takes no lock, makes no system call and never waits
  for the reporting thread; the reporting thread never waits for a reader. It
  returns one whole report, never a mix of two. It is safe as a Dart leaf
  call.
- Nothing has to be undone when the caller goes away, and the state is never
  destroyed, so reporting and reading stay valid while `exit()` runs.
- Reports are process-wide, as upstream's callback is. Two contexts working at
  once share one sequence, and a report does not say which context made it.
- To follow one run, read `sequence` before starting it and poll for larger
  values. Only the latest report is kept, so a poller slower than the steps
  skips some; read once more after the call returns to get the last one.
- The first `sd_dart_progress_enable` call registers a recorder with
  `sd_set_progress_callback`, which is not synchronized: make it before
  another thread starts a load or generation. Later calls do nothing.
- Do not call `sd_set_progress_callback` afterwards. It replaces the recorder,
  `sequence` stops advancing, and `sd_dart_progress_enable` does not register
  it again. A caller that wants a C callback uses upstream's function instead
  of this API.

A process whose caller died still waits for a native call that is in flight:
the Dart VM, for one, exits only after the isolate inside `new_sd_ctx` or
`generate_image` returns. Recording progress does not shorten that.

`tests/test_progress_stress.py` runs this against a Dart VM that dies while a
native call is reporting progress; see its docstring.

## Build

```bash
git submodule update --init --recursive
python3 tools/build.py list
python3 tools/build.py build --target macos-arm64
python3 tools/validate_artifacts.py
python3 tools/smoke_test.py macos-arm64 --expect-device MTL
python3 tools/package_release.py --tag v0.1.0
```

Linux Vulkan builds need `libvulkan-dev`, `glslc` and `spirv-headers`; Windows
Vulkan builds need the LunarG Vulkan SDK. Android builds need the NDK (`ANDROID_NDK_HOME`, or the newest NDK under
`ANDROID_HOME`). `build --target all-host` builds every target the current host
supports.

Outputs:

- `bin/<target>/lib/`: stripped runtime library.
- `bin/<target>/symbols/`: unstripped library for crash symbolication.
- `bin/<target>/include/`: `stable-diffusion.h` and `sd_dart_wrapper.h`.
- `bin/<target>/build-info.json`.
- `dist/`: release archives, `manifest.json`, `SHA256SUMS`.

## Apple XCFramework

`package_release.py --apple-xcframework` also wraps the Apple targets in
`stable_diffusion.xcframework` for a Swift Package Manager `binaryTarget`:

| Slice | Targets | Minimum |
| --- | --- | --- |
| `ios-arm64` | `ios-arm64` | iOS 16.4 |
| `ios-arm64_x86_64-simulator` | `ios-arm64-sim`, `ios-x64-sim` | iOS 16.4 |
| `macos-arm64_x86_64` | `macos-arm64`, `macos-x64` | macOS 13.3 |

Each framework exports only the `SD_API` symbols and has a `stable_diffusion`
module map. Its Info.plist minimum OS (`MinimumOSVersion` on iOS,
`LSMinimumSystemVersion` on macOS) is read from the binary's
`LC_BUILD_VERSION`, so App Store validation sees matching values. Each
framework embeds a `PrivacyInfo.xcprivacy` that declares its required-reason
API use; `validate` fails when a slice lacks it or when its categories differ
from the APIs the binary imports (see
[`docs/apple_privacy_manifest.md`](docs/apple_privacy_manifest.md)). The
per-target Apple runtime archives are bare dylibs and carry none. The
frameworks are unsigned; Xcode signs them when it embeds them. The zip is
reproducible from the same slices, and its `sha256` in `manifest.json` is the
SwiftPM checksum (`swift package compute-checksum`).

```bash
python3 tools/apple_xcframework.py build --tag v0.2.0 --dist dist
python3 tools/apple_xcframework.py validate dist/stable-diffusion-native-apple-xcframework-v0.2.0.zip
python3 tools/apple_xcframework.py consumer dist/stable-diffusion-native-apple-xcframework-v0.2.0.zip
```

`consumer` links the zip into `tests/swiftpm_consumer` the way the `llamadart`
Flutter companion does (a dynamic library that re-exports the framework), runs
the macOS probe and builds the iOS device and simulator slices.

## Release

Run the `Native Build & Release` workflow with a tag such as `v0.1.0`. Tags are
`vMAJOR.MINOR.PATCH`, with an optional `-N` rebuild counter starting at 1
(`v0.1.0-2`). Each release records the upstream commit in `manifest.json`.

Archive names:

- `stable-diffusion-native-runtime-<target>-<tag>.tar.gz`
- `stable-diffusion-native-symbols-<target>-<tag>.tar.gz`
- `stable-diffusion-native-apple-xcframework-<tag>.zip`

## License

MIT. Release archives include the stable-diffusion.cpp and ggml licenses.

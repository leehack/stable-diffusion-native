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
- Export only that API. stable-diffusion.cpp embeds its own patched ggml;
  hiding it lets the library share a process with another ggml, such as
  llama.cpp's.
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
- `bin/<target>/include/stable-diffusion.h` and `build-info.json`.
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

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
    uint64_t sequence;  // reports are numbered from 1 in the order recorded
    int32_t step;       // the arguments of upstream's progress callback
    int32_t steps;
    float time;
} sd_dart_progress_t;

void sd_dart_progress_enable(void);
size_t sd_dart_progress_read(uint64_t after, sd_dart_progress_t* reports,
                             size_t capacity, uint64_t* latest);
```

| Function | Behavior |
| --- | --- |
| `sd_dart_progress_enable` | Records progress in the library instead of printing progress bars to stdout, for every later call in the process, model loads included. Idempotent; when it returns the recorder is registered. |
| `sd_dart_progress_read` | Copies the reports with a sequence greater than `after` into `reports`, oldest first, at most `capacity`, and returns how many. `latest`, if not `NULL`, receives the sequence of the newest report (0 if none); a `capacity` of 0 only asks for that. |

Upstream's `sd_set_progress_callback` calls back on the thread that is loading
or generating, for as long as that call runs. A callback into a managed
runtime cannot be made safe there. A Dart `NativeCallable` is called after
its isolate or the VM has shut down, which aborts the process, and a wrapper
that waits for a call in flight before dropping the callback can deadlock VM
shutdown. So the library takes no callback: it records the reports itself,
and the caller reads them in order whenever it likes.

What is guaranteed:

- **Every report, in order.** The sequences of one call's reports are
  consecutive, and each report is whole: the three values of one upstream
  call, never a mix of two.
- **History.** The library keeps the 4095 most recent reports. A caller that
  reads before more than that were recorded since its last read misses none.
  The fastest sources measured on an M4 Max are `convert` (746 reports a
  second, 47 in 50 ms) and a tiled VAE decode with 64-pixel tiles (195 a
  second, 10 in 50 ms); sampling a batch of one-step images reports 16 times
  a second. So a 50 ms poll uses about 1% of the history, and a caller may
  stall for more than five seconds before a report is lost.
- **Overflow is visible.** If more than 4095 reports were recorded since
  `after`, the older ones are gone and the first report returned is not
  `after + 1`.
- **Never blocks.** `sd_dart_progress_read` takes no lock, makes no system
  call, never waits for a reporting thread and copies at most `capacity` + 4
  reports; a reporting thread never waits for a reader. It is safe as a Dart
  leaf call. A call may return fewer reports than there are; the caller has
  them all once the last sequence it holds equals `latest`.
- **Nothing to undo.** No call is needed when the caller goes away, and the
  state is never destroyed, so reporting and reading stay valid while
  `exit()` runs.

What is not:

- **Which call a report belongs to.** Reports are process-wide, as upstream's
  callback is. Every context's loads and generations go into the one
  sequence, so two contexts working at once see each other's reports, load
  reports included, and each takes up part of the other's history.
- **A registration that survives `sd_set_progress_callback`.** The first
  `sd_dart_progress_enable` call registers the recorder there. That function
  is not synchronized, so make the call before another thread starts a load
  or generation. A later `sd_set_progress_callback` call replaces the
  recorder, `latest` stops advancing, and `sd_dart_progress_enable` does not
  register it again. A caller that wants a C callback uses upstream's
  function instead of this API.
- **A shorter exit.** A process whose caller died still waits for a native
  call that is in flight: the Dart VM, for one, exits only after the isolate
  inside `new_sd_ctx` or `generate_image` returns.

To follow one call, ask for `latest` before starting it, read from there on a
timer, and read once more when the call has returned:

```c
uint64_t after;
sd_dart_progress_read(0, NULL, 0, &after);
/* start the call on another thread; then, on each tick and once at the end: */
uint64_t latest;
do {
    size_t count = sd_dart_progress_read(after, reports, capacity, &latest);
    for (size_t i = 0; i < count; i++) {
        /* reports[i].sequence == after + 1 unless reports were lost */
        after = reports[i].sequence;
    }
} while (after < latest);
```

`tests/test_progress_stress.py` runs this against a real Dart VM: one that
dies while a native call is reporting progress, and one that polls a batch of
images; see its docstring.

### Exit teardown

```c
sd_ctx_t* sd_dart_new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
bool sd_dart_generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params,
                            sd_image_t** images_out, int* num_images_out);
void sd_dart_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);
void sd_dart_exit_free(void* object);

bool sd_dart_exit_track(void* object, void (*free_fn)(void*), int32_t stage);
bool sd_dart_exit_untrack(void* object);
int32_t sd_dart_exit_tracked_count(void);
void sd_dart_exit_call_begin(void);
void sd_dart_exit_call_end(void);
void sd_dart_exit_set_wait_ms(int32_t wait_ms, int32_t work_wait_ms);
void sd_dart_exit_teardown(void);
```

ggml-metal aborts in its static destructor
(`GGML_ASSERT([rsets->data count] == 0)`) when a process exits while a Metal
buffer is still allocated. A caller in a managed runtime cannot always free
its contexts first: a Flutter macOS quit calls `exit()` without shutting the
Dart isolates down, a hot restart drops them without running finalizers, and
an isolate that is killed during `new_sd_ctx` never sees the context that
call returns. So the library keeps a registry of live objects and, on Apple
platforms, frees what is left of it during `exit()`, before the first of its
statics is destroyed.

| Function | Behavior |
| --- | --- |
| `sd_dart_new_sd_ctx` | `new_sd_ctx` that tracks the context before it returns. |
| `sd_dart_generate_image` | `generate_image` as a call in flight: teardown cancels it and waits for it. |
| `sd_dart_cancel_generation` | `sd_cancel_generation` for a tracked context, from any thread. Does nothing for a context that is not tracked, including one already freed. Never blocks. |
| `sd_dart_exit_free` | Untracks an object and frees it with the function it was tracked with (`free_sd_ctx` for a context). Does nothing when it is not tracked, so it also works as a Dart `NativeFinalizer`. |
| `sd_dart_exit_track`, `_untrack` | For C and C++ callers: track any other object, such as an `upscaler_ctx_t`, with its free function and a stage, or stop tracking it. |
| `sd_dart_exit_tracked_count` | Number of tracked objects. |
| `sd_dart_exit_call_begin`, `_end` | For C and C++ callers: mark a call in flight around an upstream function that has no wrapper. |
| `sd_dart_exit_set_wait_ms` | How long teardown waits for calls in flight: `work_wait_ms` (15000 by default) while a load or a generation is among them, `wait_ms` (2000) otherwise. |
| `sd_dart_exit_teardown` | Runs teardown now, for native hosts on platforms where it does not run by itself. Follow it directly with `exit()`. |

A Dart caller replaces `new_sd_ctx`, `generate_image`, `sd_cancel_generation`
and `free_sd_ctx` with the first four and binds nothing else: the remaining
functions are for native hosts, and a Dart program that calls
`sd_dart_exit_teardown` and returns from `main` waits forever for its blocked
isolates.

What is guaranteed:

- **Tracking begins and ends in native code.** A context is tracked before
  the call that creates it returns and untracked inside the call that frees
  it, so an isolate that is killed in between leaves nothing untracked and
  nothing is freed twice.
- **No free under a running call.** Teardown asks every tracked context to
  cancel its generation, repeating the request until the calls in flight
  have ended, and frees only then. If one is still running after the wait,
  it frees nothing at all rather than what happens to be idle.
- **A bounded exit, and no wait without a reason.** With no call in flight
  teardown frees once 250 ms have passed since the last one ended: at once
  when that is long ago, and after 255 ms, as measured, when the quit
  directly follows a call. With nothing tracked and no call creating or freeing a tracked
  object, it returns at once. Otherwise the wait ends when the calls do, at
  the latest after 15 s while `sd_dart_new_sd_ctx` or
  `sd_dart_generate_image` is in flight and after 2 s for any other call,
  plus the same 250 ms, which a thread that has just left a call gets to
  finish the short calls that follow it.
- **A free in flight is waited for**, with the 2 s bound, also when it is
  freeing the last tracked object and the registry is already empty.
- **Nothing returns into freed memory.** Once teardown has begun, a thread
  that ends its outermost call in flight, or reaches the registry outside
  one, never returns to its caller.
- **Order.** Objects are freed by stage, an object before what it uses, and
  latest first within a stage. On Apple platforms that happens before any
  static of the library is destroyed, whenever the static was created: the
  library defines `__cxa_atexit` for its own image, hidden and not exported,
  and registers every static destructor behind a call to teardown.
  `validate_artifacts.py` and `apple_xcframework.py validate` fail a slice
  that imports the system function instead.
- **Callers that do not use it are unaffected.** A context from `new_sd_ctx`
  is not tracked, and exiting with one alive behaves as before, the Metal
  abort included. Another ggml in the process, such as libllamadart's with
  its own copy of this registry, is not touched either.

What is not:

- **Most of a generation cannot be interrupted.** stable-diffusion.cpp reads
  a cancellation only between the phases of a generation, before a sampling
  step and before the decode of each image: a load, the text encoder, one
  sampling step and the VAE decode of one image run to their end. Teardown waits up to 15 s for
  them, so **quitting during a large generation can delay the exit of the
  process by up to 15 s**. A phase that outlasts the bound costs the whole
  wait and still leaves everything allocated: Metal aborts as before, 15 s
  later. Measured with SD-Turbo (q8, 4 steps) on an M4 Max:

  | Size | One step, Metal | VAE decode, Metal | VAE decode, CPU |
  | --- | --- | --- | --- |
  | 512 × 512 | 0.3 s | 1.8 s | 6.7 s |
  | 640 × 640 | 0.5 s | 2.8 s | |
  | 768 × 768 | 0.7 s | 4.1 s | about 12 s |
  | 1024 × 1024 | 1.7 s | 7.4 s | |

  The decode scales with the pixel count, so on that machine the bound holds
  to about 1024 × 1024 with a factor of two to spare, and at 768 × 768 on a
  GPU up to three times slower. It is exceeded by larger images, by slower
  devices at those sizes, by a VAE decode on the CPU a little above
  768 × 768, and by a load that takes longer than 15 s. A host that knows its models
  sets its own bounds with `sd_dart_exit_set_wait_ms`; passing 2000 for both
  restores a 2 s limit for every call.
- **Upstream calls on a tracked context are not waited for.** A raw
  `generate_image`, `generate_video` or `adetail_image` on a context from
  `sd_dart_new_sd_ctx` is a use after free at exit. Native callers bracket
  such a call with `sd_dart_exit_call_begin` and `_end`; Dart callers cannot,
  because a killed isolate never reaches the end.
- **Platforms other than Apple's.** The abort is ggml-metal's, so elsewhere
  teardown runs only when a native host calls it. CI checks on Linux that a
  context left alive at exit is harmless on the CPU, and reports what the
  Vulkan backend does on Mesa lavapipe; no hardware Vulkan driver was tried.
- **Threads that teardown blocked stay blocked.** A static destructor or
  `atexit` handler of another library that joins one hangs the exit.

`tests/test_exit_teardown.py` runs the registry against stand-ins for
upstream, also under AddressSanitizer and ThreadSanitizer.
`tests/test_exit_teardown_runtime.py` runs it against a built macOS runtime
and a real `sd_ctx_t`, on Metal and on the CPU: the test writes a 12 MB
PixArt model itself, so CI needs no download. Where no Metal residency set is
live, as on GitHub's macOS runners, an untracked context exits cleanly and the
Metal abort is not exercised; there the test shows each free through the
allocator instead. Those runners also crash in ggml-metal when their virtual
GPU computes, so on Metal the test only loads models there and runs the
scenarios that need a generation on the CPU. It does the same on any machine
whose default device cannot generate and says so in one line; set
`SD_REQUIRE_METAL_GENERATION=1` on a Mac to make that a failure instead.

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

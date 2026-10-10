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
statics is destroyed. On Linux `exit()` frees nothing and keeps the statics
alive; see [Exit on Linux](#exit-on-linux).

| Function | Behavior |
| --- | --- |
| `sd_dart_new_sd_ctx` | `new_sd_ctx` that tracks the context before it returns. |
| `sd_dart_generate_image` | `generate_image` as a call in flight: teardown cancels it and waits for it. |
| `sd_dart_cancel_generation` | `sd_cancel_generation` for a tracked context, from any thread. Does nothing for a context that is not tracked, including one already freed. Never blocks. |
| `sd_dart_exit_free` | Untracks an object and frees it with the function it was tracked with (`free_sd_ctx` for a context). Does nothing when it is not tracked, so it also works as a Dart `NativeFinalizer`. |
| `sd_dart_exit_track`, `_untrack` | For C and C++ callers: track any other object, such as an `upscaler_ctx_t`, with its free function and a stage, or stop tracking it. |
| `sd_dart_exit_tracked_count` | Number of tracked objects. |
| `sd_dart_exit_call_begin`, `_end` | For C and C++ callers: mark a call in flight around an upstream function that has no wrapper. |
| `sd_dart_exit_set_wait_ms` | How long teardown waits for calls in flight: `work_wait_ms` (15000 by default) while a load, a generation or a device query is among them, `wait_ms` (2000) otherwise. |
| `sd_dart_exit_teardown` | Runs teardown now, frees included on every platform, for native hosts. Follow it directly with `exit()`. |

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
  directly follows a call. With nothing tracked and no call creating or
  freeing a tracked object or querying the devices, it returns at once.
  Otherwise the wait ends when the calls do, at the latest after 15 s while
  `sd_dart_new_sd_ctx`, `sd_dart_generate_image`, `sd_dart_gpu_device_count`
  or `sd_dart_gpu_device_memory` is in flight and after 2 s for any other
  call, plus the same 250 ms, which a thread that has just left a call gets
  to finish the short calls that follow it.
- **A device query in flight is waited for**, with the 15 s bound, also when
  nothing is tracked, which is the state of a caller that asks for the
  device's memory before its first load. The query reads ggml's device
  registry, a static that `exit()` destroys. Before this wait a process that
  exited during `sd_dart_gpu_device_memory` crashed in 2 of 200 runs with
  four querying threads on an M4 Max, and in 0 of 300 with it.
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
- **Frees at exit anywhere but on Apple platforms.** The abort is
  ggml-metal's. On Linux `exit()` frees nothing, and on Android and Windows
  nothing runs at exit: there teardown runs only when a native host calls it.
- **Threads that teardown blocked stay blocked.** A static destructor or
  `atexit` handler of another library that joins one hangs the exit.

#### Exit on Linux

Linux, Android excepted, retains the library's owned statics through a hidden
`__cxa_atexit` that drops their destructors. It registers no Linux wrapper exit
handler. C exit runs normal host callbacks, dependency destructors and libc
stream flushing: no forced `_exit`, early flush, wait, automatic free or
late-call refusal. The host remains responsible for freeing its contexts.

Before C exit, the native host stops new work, stops and joins workers, and
shuts down its Dart isolates or Flutter engine. Await each isolate's exit and
native finalizers; requesting a kill is insufficient. A native call in progress
must complete before its isolate stops. Do not invoke explicit teardown before
worker joins: it parks later guarded calls and can deadlock them.

Direct C exit with live Dart isolates can abort the VM independently of this
library ([#977](https://github.com/leehack/llamadart/issues/977)). Owned static
retention cannot protect external GPU-driver or BLAS destructors. C exit while
those dependency-using workers remain alive is outside this contract. A timeout
does not establish quiescence; normal exit must wait or the host must explicitly
choose its own abrupt termination policy. Apple automatic teardown, explicit
teardown, Android and Windows are unchanged. musl and hardware GPU exit behavior
remain unqualified.

The rejected Linux candidate skipped earlier host callbacks and deadlocked
when its unconditional `fflush(NULL)` met a worker's FILE lock that an earlier
host callback would release. Removing only that flush lost buffered output.
Neither behavior is part of this revision. Native preservation tests keep the
FILE-lock scenario, earlier callback, status37 and buffered-output assertions.
The maintained Dart runtime harness adds cooperative shutdown scenarios that
await isolate exit before C exit. Raw exit scenarios are diagnostics and do not
qualify the cooperative contract.

Pre-implementation lifecycle proof in the llama.cpp owner used Dart3.13.1 /
Linuxarm64 GCC14/glibc2.41 / stories15M CPU.60/60 cooperative model exits and
15/15 no-model controls retained callbacks and output;15/15 raw no-model exits
VM-aborted. This is host-lifecycle evidence, not qualification of this image
runtime. Fresh image CPU/Metal/Vulkan runtime, sanitizers and driver qualification
are still required on this exact revision. No prior result obtained by skipping
host callbacks is current readiness evidence.

`tests/test_exit_teardown.py` runs the registry against stand-ins for
upstream, also under AddressSanitizer and ThreadSanitizer.
`tests/test_exit_teardown_runtime.py` runs it against a built macOS runtime
and a real `sd_ctx_t`, on Metal and on the CPU: the test writes a 12 MB
PixArt model itself, so CI needs no download. Against a built Linux runtime
(`SD_EXIT_TEARDOWN_TARGET`) the same supported scenarios must preserve host callbacks and native cleanup
ordering on the CPU; raw exits with a call in flight remain diagnostic; what a Vulkan
target does on its first Vulkan device, Mesa lavapipe in CI, is reported and
not failed. Where no Metal residency set is
live, as on GitHub's macOS runners, an untracked context exits cleanly and the
Metal abort is not exercised; there the test shows each free through the
allocator instead. Those runners also crash in ggml-metal when their virtual
GPU computes, so on Metal the test only loads models there and runs the
scenarios that need a generation on the CPU. It does the same on any machine
whose default device cannot generate and says so in one line; set
`SD_REQUIRE_METAL_GENERATION=1` on a Mac to make that a failure instead.

### GPU device memory

```c
typedef struct {
    uint64_t total_bytes;
    uint64_t free_bytes;
    int32_t type;           // an sd_dart_gpu_device_type
    char name[64];          // as in sd_list_devices() and sd_ctx_params_t.backend
    char description[256];
} sd_dart_gpu_device_memory_t;

int32_t sd_dart_gpu_device_count(void);
int32_t sd_dart_gpu_device_memory(int32_t device_index, sd_dart_gpu_device_memory_t* out);
```

A caller that refuses a model which does not fit needs the memory of the GPU
the model would load on, before it loads it. Upstream's API has no such
figure, and host memory is the wrong one for a discrete GPU.

| Function | Behavior |
| --- | --- |
| `sd_dart_gpu_device_count` | Number of GPU devices, discrete and integrated, or `SD_DART_GPU_NO_BACKEND`. They are numbered from 0 in the order `sd_list_devices` lists them, which has other devices, such as the CPU, in between. |
| `sd_dart_gpu_device_memory` | Writes the memory, type, name and description of a device to `out` and returns `SD_DART_GPU_OK`. `SD_DART_GPU_DEFAULT_DEVICE` (-1) is the device a context without a `backend` uses: the first discrete GPU, or else the first integrated one; upstream's `SD_VK_DEVICE` environment variable is not read. Any other status leaves `out` as it was. Each call asks the device again. |

| Status | Value | Meaning |
| --- | --- | --- |
| `SD_DART_GPU_OK` | 0 | |
| `SD_DART_GPU_INVALID_ARGUMENT` | -1 | A null `out`, or a negative index other than the default device. |
| `SD_DART_GPU_NO_BACKEND` | -2 | The library was built without a GPU backend, as the CPU targets are. |
| `SD_DART_GPU_NO_DEVICE` | -3 | The library has a GPU backend, but no such device: none was found, or the index is past the last one. |
| `SD_DART_GPU_UNAVAILABLE` | -4 | The device is there, but the backend reports a total of 0 or failed to answer. |

The figures are ggml's (`ggml_backend_dev_memory`), the ones
stable-diffusion.cpp's own automatic fit works with:

| Device | `total_bytes` | `free_bytes` |
| --- | --- | --- |
| Vulkan, discrete (`SD_DART_GPU_DEVICE_DISCRETE`) | The heaps flagged device-local, added up. | With `VK_EXT_memory_budget`, the budget of those heaps less what this process uses of them: the driver's estimate of what the process can still allocate, which reflects other processes. Without the extension, the heap sizes, so `free_bytes == total_bytes`. |
| Vulkan, integrated (`SD_DART_GPU_DEVICE_INTEGRATED`) | Every heap, added up. | The same, over every heap. |
| Metal (`SD_DART_GPU_DEVICE_DISCRETE`) | `recommendedMaxWorkingSetSize` | That less `currentAllocatedSize`, which counts this process only. |

- **Integrated GPUs share host memory.** Their heaps are system memory, and a
  driver that exposes it as more than one heap counts it more than once, so
  the host's available memory is a second limit there. Apple GPUs report as
  discrete although their memory is unified: the working set is a share of
  physical memory, 51.8 of 64 GiB on an M4 Max.
- **Never more free than total, never a wrapped value.** ggml-vulkan subtracts
  use from budget per heap in unsigned arithmetic. Use above the budget reads
  as 0 free bytes here.
- **`free_bytes` is what is free now, not what is left once the loaded
  contexts are in use.** stable-diffusion.cpp moves a context's weights to the
  device when they are first used, unless `sd_ctx_params_t.eager_load` is set.
  Measured with SDXS on Metal:

  | `eager_load` | After `sd_dart_new_sd_ctx` | After the first generation | After `sd_dart_exit_free` |
  | --- | --- | --- | --- |
  | `false`, the default | unchanged | -650 MiB | restored |
  | `true` | -652 MiB | -652 MiB | restored |

  What a generation computes in is allocated while it runs. Upstream's API
  reports neither what a context will take nor what it holds, and the wrapper
  adds no such figure: a caller that admits a second model against
  `free_bytes` sets `eager_load`, or keeps count of what it has loaded.
- **No context and no model, but the first call initializes the devices.** It
  registers ggml's backends, as `sd_list_devices` does. On Vulkan that creates
  the instance. On Metal it compiles the shader libraries: about 50 ms when
  the system has them cached, 16 s on an M4 Max and 27 to 45 s in five runs
  on GitHub's macOS runners when it had not. Make the first call on a thread
  that may wait that long, never on a UI thread; it can also outlast the 15 s
  that exit teardown waits for it. A later call takes about a microsecond.
- **The default device ignores `SD_VK_DEVICE`.** Upstream uses the Vulkan
  device of that number for a context without a `backend`, and falls back
  when the device does not initialize, which a query cannot know without
  initializing it. A caller that sets the variable asks for that device by
  its index.
- **Any thread, any time before teardown.** [Exit teardown](#exit-teardown)
  waits for a query in flight, also when nothing is tracked, with the bound
  of a load (15 s by default), and a query blocks after teardown. Past the
  bound the exit goes on under the query, as it would without the registry.
- **Whether `free_bytes` is a live figure is not reported.** On Vulkan that
  depends on `VK_EXT_memory_budget`, which ggml checks without exposing the
  result. Without it the check a caller makes is against the device's size.

`tests/test_device_memory.py` runs the exports against a stand-in for ggml's
device registry, and `tools/smoke_test.py` compares them with `sd_list_devices`
on each built runtime: CI does that on Mesa lavapipe for Vulkan, on the macOS
runners' virtual GPU for Metal, and on the CPU targets for
`SD_DART_GPU_NO_BACKEND`. No hardware Vulkan device was tried.

### Log forwarding

```c
void sd_dart_log_enable(void);
void sd_dart_log_set_level(int32_t level);
uint64_t sd_dart_log_read(uint64_t after, char* text, size_t capacity,
                          int32_t* level, size_t* length);
uint64_t sd_dart_log_dropped(void);
size_t sd_dart_last_error(char* text, size_t capacity);
```

Upstream's `sd_set_log_callback` calls back on whichever thread logs, the
model loader's own threads among them, with a text that is only valid during
the call. A managed runtime cannot take that call, for the reasons given for
progress above. Without a callback the messages go nowhere, and a load that
fails returns `NULL` and nothing else. So the library copies each message into
a buffer of its own, and the caller reads the messages whenever it likes.

| Function | Behavior |
| --- | --- |
| `sd_dart_log_enable` | Records log messages for every later call in the process, ggml's included. Idempotent; when it returns the recorder is registered. |
| `sd_dart_log_set_level` | The lowest `sd_log_level_t` recorded, `SD_LOG_INFO` by default; `SD_LOG_ERROR + 1` records nothing. A message below the level gets no sequence. |
| `sd_dart_log_read` | Copies the oldest message with a sequence greater than `after` into `text`, at most `capacity` - 1 bytes and a NUL, and returns its sequence, or 0 if there is none. `level` and `length`, if not `NULL`, receive its `sd_log_level_t` and the bytes of the whole text. Removes nothing. |
| `sd_dart_log_dropped` | Number of messages that left the buffer newer than every message a read had returned by then. |
| `sd_dart_last_error` | Copies the `SD_LOG_ERROR` messages recorded, by any thread, while the calling thread's most recent `sd_dart_new_sd_ctx` or `sd_dart_generate_image` ran, joined by `\n`, and returns the bytes of the whole text: 0 when that call logged no error. |

A message is the text upstream passes to its callback, without the line break
at its end: `<file>:<line> - <text>` for stable-diffusion.cpp and
`ggml - <text>` for ggml. For a model file that does not exist,
`sd_dart_last_error` returns:

```text
model_loader_files.cpp:25   - cannot inspect model source '/models/missing.gguf': No such file or directory
diffusion_engine.cpp:727  - init model loader from file failed: '/models/missing.gguf'
diffusion_engine.cpp:992  - get sd version from file failed: '/models/missing.gguf'
```

What is guaranteed:

- **Nothing changes for a process that does not enable it.** Messages are
  dropped as before, and ggml prints its own to stderr until the first
  context exists.
- **Every message, in order, with its level.** Messages are numbered from 1 in
  the order they were recorded, by whichever thread, and each is whole. A
  buffer of `SD_DART_LOG_TEXT_SIZE` (4096) bytes holds any of them; upstream's
  longer ones are cut to fit, between two UTF-8 sequences.
- **Bounded memory.** The library keeps the most recent messages that fit in
  256 KiB, about 2500 lines of 100 bytes, and the 32 most recent errors of up
  to 511 bytes: 272 KiB of zero-filled memory, used only once enabled. A load
  of SDXS logs 42 messages at `SD_LOG_INFO` (3.3 KB) and 74 at `SD_LOG_DEBUG`.
- **Overflow is visible.** When the message after `after` is gone, the
  sequence returned is not `after + 1`, which tells a reader how many it
  missed, and `sd_dart_log_dropped` counts them. For a caller that never reads,
  that count is everything that left the buffer.
- **The logging thread allocates nothing and waits for nothing but one copy.**
  The buffer is held for the copy of a single message, by a reader as by a
  writer, and the errors that `sd_dart_last_error` returns are behind a flag
  of their own, so a failed call's reason never waits behind the other
  messages or their reader. A thread waits at most 100 ms for either: longer
  than that only a holder takes that the scheduler keeps off its processor,
  or one that died, as `ExitProcess` ends threads on Windows. A logging
  thread then loses its message, which `sd_dart_log_dropped` counts, and the
  ones after it do not wait.
- **Texts are cut between two UTF-8 sequences**, by the library's limits and
  by the caller's `capacity` alike.
- **Nothing to undo, and valid during exit.** No call is needed when the
  caller goes away. The state is never destroyed, so logging and reading stay
  valid while `exit()` runs, during exit teardown and after it; reading and
  `sd_dart_last_error` never block after teardown.

What is not:

- **A copy on stderr.** Once enabled, nothing recorded is printed, also not
  what ggml used to print before the first context existed, such as the lines
  of Metal's device initialization. A caller that wants the messages on stderr
  prints what it reads. What a ggml backend writes to stderr itself, not
  through ggml's log, still goes there.
- **Which call a message belongs to.** Messages are process-wide, as upstream's
  callback is. `sd_dart_last_error` narrows that down by time only: it includes
  the errors that another thread's call logged while this one ran.
- **A reason for every failure.** It is what upstream logged at `SD_LOG_ERROR`.
  A file in the wrong role, such as a full model given as the only VAE, gets
  one line, `get sd version from file failed: ''`. An empty last error does
  not prove that nothing was logged either: an error whose thread could not
  get the errors' flag within 100 ms is not kept there.
- **A last error on another thread.** It belongs to the thread that made the
  call: read it there, before that thread's next load or generation. A Dart
  isolate stays on its thread between two native calls that no asynchronous
  gap separates.
- **A registration that survives `sd_set_log_callback`.** The first
  `sd_dart_log_enable` call registers the recorder there and with ggml's log.
  Neither is synchronized, so make the call before another thread starts a
  load, a generation or a device query (`sd_list_devices`,
  `sd_dart_gpu_device_count`, `sd_dart_gpu_device_memory`), which logs through
  ggml for as long as it runs. A later `sd_set_log_callback` call replaces the
  recorder, and `sd_dart_log_enable` does not register it again.

To follow the log, read on a timer and once more when a call has returned:

```c
char text[SD_DART_LOG_TEXT_SIZE];
int32_t level;
uint64_t next;
while ((next = sd_dart_log_read(after, text, sizeof(text), &level, NULL)) != 0) {
    /* next - after - 1 messages were lost if next != after + 1 */
    after = next;
}
```

`tests/test_log.py` runs the recorder against stand-ins for upstream and ggml,
also under AddressSanitizer and ThreadSanitizer: messages from several threads
at once, overflow, and logging and reading while `exit()` runs.
`tests/test_exit_teardown.py` and `tests/test_exit_teardown_runtime.py` add a
load that fails for three reasons, a recorder that is read through exit
teardown, and a Dart VM that polls it while worker isolates are killed.
`tools/smoke_test.py` loads a missing model file on each built runtime.

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

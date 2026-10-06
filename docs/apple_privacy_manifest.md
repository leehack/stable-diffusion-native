# Apple Privacy Manifest

Every framework slice of `stable-diffusion-native-apple-xcframework-<tag>.zip`
embeds `tools/apple/PrivacyInfo.xcprivacy`. Apple requires an SDK to report its
own
[required-reason API](https://developer.apple.com/documentation/bundleresources/describing-use-of-required-reason-api)
use; it cannot rely on the host app's manifest. The per-target Apple runtime
archives are bare dylibs and carry none.

## Location

| Slice | Bundle layout | Manifest path inside `stable_diffusion.framework` |
| --- | --- | --- |
| `ios-arm64`, `ios-arm64_x86_64-simulator` | flat | `PrivacyInfo.xcprivacy` |
| `macos-arm64_x86_64` | versioned | `Versions/A/Resources/PrivacyInfo.xcprivacy` |

A manifest in the root of a versioned framework is unsealed content and fails
code signing. The packager does not sign the frameworks; the consumer's Xcode
build seals the manifest when it embeds and signs them.

## Declarations

The SDK does not track, has no tracking domains, and collects no data.

| Category | Reasons | Evidence |
| --- | --- | --- |
| `NSPrivacyAccessedAPICategoryFileTimestamp` | `C617.1`, `3B52.1` | Every slice and architecture imports `stat` and `fstat` (`stat$INODE64`/`fstat$INODE64` on macOS x86_64) and no other required-reason symbol or selector. |

The `stat`/`fstat` call sites are the same in all five images:

| Caller | API | Use |
| --- | --- | --- |
| `file_exists`, `is_directory` (`src/core/util.cpp`) | `stat` | File type of a model path. `ModelLoader::parse_file` calls them for every model and component path the app passes through the C API, such as the `sd_ctx_params_t` paths of `new_sd_ctx`. |
| `MmapWrapper::create` (`src/core/util.cpp`) | `fstat` | Size of a model file before `ModelLoader::process_model_files` memory-maps it (`enable_mmap`). |
| `zip_create`, `zip_entry_fwrite`, `mz_zip_writer_add_file`, `mz_zip_add_mem_to_archive_file_in_place_v2` (vendored zip/miniz) | `stat` | Zip writing. stable-diffusion.cpp only opens zips for reading, so nothing calls these. |

`ModelLoader::read_file_stamp` also reads each model file's size and
modification time through `std::filesystem` to detect a changed file. libc++
is not on Apple's list, but it is the same access for the same purpose.

The paths are whatever the app passes, so both reasons apply:

- `C617.1`: a model the app downloaded or bundled sits inside its container.
- `3B52.1`: a model the user picked, for example through a document picker,
  sits outside it.

`0A2A.1` does not apply (the SDK exposes no wrapper around a timestamp API),
and nothing is displayed to the user (`DDA9.1`).

The binary imports no boot-time, disk-space, user-defaults or active-keyboard
API. `clock_gettime`, `host_statistics64` and `sysctlbyname`
(`hw.perflevel0.physicalcpu`, `hw.physicalcpu`, `machdep.cpu.brand_string`)
are not on Apple's list. `dlsym` resolves only `ggml_backend_init` and
`ggml_backend_score` for ggml's backend loader, plus the CoreFoundation
functions of the compiler runtime's OS-availability check.

## Validation

`apple_xcframework.py build` and `validate` run this check, so
`package_release.py` cannot write a checksum for an archive that fails it.
`native_release.yml` runs it again on the packaged zip before upload:

```bash
python3 tools/apple_privacy_manifest.py --audit-imports \
  dist/stable-diffusion-native-apple-xcframework-<tag>.zip
```

It fails when a slice lacks the manifest, carries it in the wrong location,
ships an invalid property list or an unapproved reason, or when the declared
categories differ from the required-reason symbols and Objective-C selectors
the slice binary references in any architecture (`nm -u -arch all`,
`otool -arch all`). The audit cannot see calls resolved through `dlsym`, and
it compares categories, not reasons.

When an upstream bump makes the audit fail, or changes which files reach the
call sites above, find the call sites, choose the reasons that describe them,
and update the manifest and this page together. Do not add a category only to
make the check pass.

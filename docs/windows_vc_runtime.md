# Windows Visual C++ runtime

Both Windows archives ship the Visual C++ v14 runtime beside
`stable-diffusion.dll`. Keep the DLLs together when unpacking or deploying
the archive. These bundles can load on Windows 10 or later without installing
the Visual C++ Redistributable. The v0.4.0 archives still need the x64
Redistributable. The Vulkan variant also requires the Vulkan loader installed
by the GPU driver.

## Files and source

The build follows the runtime's PE imports transitively. Both variants
currently need four Visual C++ DLLs:

- `msvcp140.dll`
- `msvcp140_codecvt_ids.dll`
- `vcruntime140.dll`
- `vcruntime140_1.dll`

The runtime stays linked with the dynamic CRT (`/MD`) so existing native
clients can free the strings returned by `sd_*_params_to_str` using the
shared CRT. Windows supplies the Universal CRT, including the
`api-ms-win-crt-*` API sets. The Vulkan driver supplies `vulkan-1.dll`.
The archives omit those system and driver files.

The build copies the required DLLs from
`%VCToolsRedistDir%\x64\Microsoft.VC*.CRT`. Without an initialized Visual
Studio environment, it locates the default x64 toolset using `vswhere.exe`.
The distributable files are listed in Visual Studio's `redist.txt`
([Microsoft redistribution terms](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution)).
The lookup is limited to the desktop redistributable directory. Before
staging, Windows must report a valid embedded Authenticode signature from
Microsoft Corporation for each file. The build refreshes these files each
release because Windows Update does not service app-local copies.

A Windows process reuses a runtime DLL of the same name if it is already
loaded. If an application also ships those DLLs, deploy copies at least as
new as the version staged by the build.

## Validation

The build, artifact validator, release packager, and prepared-release replay
check regular and delay-load imports against the same allowlist of Windows
libraries and known API sets. They check PE architecture and the symbols
provided by bundled dependencies. Windows build metadata records the full
MSVC toolset version selected by CMake and the hashes of the staged CRT files.
The CRT file version must meet that toolset version, including the build and
revision components.

Run Windows artifact validation on Windows. It checks Authenticode again on
the staged DLLs and Windows CI compares the import names with
`dumpbin /dependents`. Packaging and replay on other hosts check the CRT
hashes against the Windows build metadata from the verified workflow
artifacts. CI also runs a statically linked host that calls
`sd_version` and `sd_commit` and checks that all four VC++ DLLs loaded from the
bundle directory. A stock Windows Server 2022 load check verifies the same
behavior without a machine-wide Redistributable.

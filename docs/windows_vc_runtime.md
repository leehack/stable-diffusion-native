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
The lookup is limited to the desktop redistributable directory. The build
refreshes these files each release because Windows Update does not service
app-local copies.

A Windows process reuses a runtime DLL of the same name if it is already
loaded. If an application also ships those DLLs, deploy copies at least as
new as the version staged by the build.

## Validation

The build, artifact validator, release packager, and prepared-release replay
check the regular and delay-load imports of every bundled DLL. They check PE
architecture, the symbols provided by bundled dependencies, and the bundled
runtime versions. Windows CI cross-checks the import names with
`dumpbin /dependents`. It also runs a statically linked host that calls
`sd_version` and `sd_commit` and checks that all four VC++ DLLs loaded from the
bundle directory. A stock Windows Server 2022 load check verifies the same
behavior without a machine-wide Redistributable.

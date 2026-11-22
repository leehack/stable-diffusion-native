"""Windows runtime bundle policy, Visual C++ runtime staging and a PE reader.

Adapted from litert-lm-native and llamadart-native to use the same app-local
runtime policy. The PE reader also runs on the macOS packaging runner.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from fnmatch import fnmatch
import hashlib
import json
import mmap
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys

MACHINES = {0x8664: "x64", 0xAA64: "arm64", 0x014C: "x86"}

# Libraries every supported Windows 10+ host provides.
OS_PROVIDED_DLLS = frozenset({
    "advapi32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "cfgmgr32.dll",
    "combase.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d3d12.dll",
    "dbghelp.dll",
    "dxgi.dll",
    "gdi32.dll",
    "gdi32full.dll",
    "imm32.dll",
    "kernel32.dll",
    "kernelbase.dll",
    "msvcrt.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "powrprof.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "sechost.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "ucrtbase.dll",
    "user32.dll",
    "version.dll",
    "winmm.dll",
    "ws2_32.dll",
})
VULKAN_DRIVER_DLL = "vulkan-1.dll"
API_SET_PREFIXES = ("api-ms-win-", "ext-ms-")
# UCRT contracts imported by the current x64 DLLs, available on Windows 10+.
OS_API_SETS = frozenset({
    "api-ms-win-crt-convert-l1-1-0.dll",
    "api-ms-win-crt-environment-l1-1-0.dll",
    "api-ms-win-crt-filesystem-l1-1-0.dll",
    "api-ms-win-crt-heap-l1-1-0.dll",
    "api-ms-win-crt-locale-l1-1-0.dll",
    "api-ms-win-crt-math-l1-1-0.dll",
    "api-ms-win-crt-runtime-l1-1-0.dll",
    "api-ms-win-crt-stdio-l1-1-0.dll",
    "api-ms-win-crt-string-l1-1-0.dll",
    "api-ms-win-crt-time-l1-1-0.dll",
    "api-ms-win-crt-utility-l1-1-0.dll",
})
TOOLSET_VERSION = re.compile(r"[0-9]{1,5}(?:[.][0-9]{1,5}){2,3}")

# Visual C++ v14 runtime DLLs. Release bundles ship them app-locally
# from the build toolset's redistributable folder (VCToolsRedistDir), never
# from System32.
VC_RUNTIME_PATTERNS = (
    "msvcp140.dll",
    "msvcp140_*.dll",
    "vcruntime140.dll",
    "vcruntime140_*.dll",
    "concrt140.dll",
    "vccorlib140.dll",
)
VC_RUNTIME_FOLDER = "Microsoft.VC*.CRT"

# Runtimes that may not be redistributed: what Visual Studio ships only under
# VC\Redist\MSVC\<ver>\debug_nonredist (the debug runtimes and the LLVM OpenMP
# runtime that ClangCL /openmp imports), and the Windows
# SDK's debug CRT, ucrtbased.dll.
NON_REDISTRIBUTABLE_PATTERNS = (
    "libomp140*.dll",
    "ucrtbased.dll",
    "*140d.dll",
    "*140ud.dll",
    "*140d_*.dll",
    "*140_*d.dll",
)


def is_os_provided(name: str) -> bool:
    lowered = name.lower()
    return lowered in OS_PROVIDED_DLLS or lowered in OS_API_SETS


def is_vc_runtime(name: str) -> bool:
    return any(fnmatch(name.lower(), pattern) for pattern in VC_RUNTIME_PATTERNS)


def is_non_redistributable(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch(lowered, pattern) for pattern in NON_REDISTRIBUTABLE_PATTERNS)


def find_vc_redistributable(name: str, arch: str, redist_dir: Path) -> Path | None:
    """The desktop redistributable copy of `name` for `arch`, if the toolset has one."""
    if not is_vc_runtime(name) or is_non_redistributable(name):
        return None
    lowered = name.lower()
    matches = sorted(
        candidate for candidate in (redist_dir / arch).glob(f"{VC_RUNTIME_FOLDER}/*")
        if candidate.is_file() and candidate.name.lower() == lowered
    )
    return matches[-1] if matches else None


class PeError(ValueError):
    pass


@dataclass(frozen=True)
class PeImage:
    machine: str
    linker_version: tuple[int, int]
    imports: dict[str, frozenset[str]]
    delay_imports: dict[str, frozenset[str]]
    exports: frozenset[str]
    export_ordinals: frozenset[str]
    file_version: tuple[int, int, int, int] | None
    crt_sha256: str | None

    def all_imports(self) -> dict[str, frozenset[str]]:
        merged = {name: set(symbols) for name, symbols in self.imports.items()}
        for name, symbols in self.delay_imports.items():
            merged.setdefault(name, set()).update(symbols)
        return {name: frozenset(symbols) for name, symbols in merged.items()}


class _Reader:
    def __init__(self, data) -> None:
        self.data = data
        if data[:2] != b"MZ":
            raise PeError("missing MZ header")
        (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
        if data[pe_offset:pe_offset + 4] != b"PE\0\0":
            raise PeError("missing PE signature")
        coff = pe_offset + 4
        machine, section_count = struct.unpack_from("<HH", data, coff)
        (optional_size,) = struct.unpack_from("<H", data, coff + 16)
        optional = coff + 20
        magic, major, minor = struct.unpack_from("<HBB", data, optional)
        if magic == 0x20B:
            self.pe32_plus = True
            (self.image_base,) = struct.unpack_from("<Q", data, optional + 24)
            directories = optional + 112
        elif magic == 0x10B:
            self.pe32_plus = False
            (self.image_base,) = struct.unpack_from("<I", data, optional + 28)
            directories = optional + 96
        else:
            raise PeError(f"unknown optional header magic 0x{magic:x}")
        (directory_count,) = struct.unpack_from("<I", data, directories - 4)
        self.directories = [
            struct.unpack_from("<II", data, directories + 8 * index)
            for index in range(min(directory_count, 16))
        ]
        self.machine = MACHINES.get(machine, f"0x{machine:04x}")
        self.linker_version = (major, minor)
        self.sections = []
        for index in range(section_count):
            entry = optional + optional_size + 40 * index
            virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", data, entry + 8)
            self.sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def offset(self, rva: int) -> int:
        for virtual_address, size, raw_offset in self.sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        raise PeError(f"RVA 0x{rva:x} is outside every section")

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def u32(self, rva: int) -> int:
        return struct.unpack_from("<I", self.data, self.offset(rva))[0]

    def string(self, rva: int) -> str:
        start = self.offset(rva)
        end = self.data.find(b"\0", start)
        if end < 0:
            raise PeError(f"unterminated string at RVA 0x{rva:x}")
        return self.data[start:end].decode("ascii", errors="replace")

    def thunk_names(self, rva: int) -> frozenset[str]:
        names = set()
        width, ordinal_flag = (8, 1 << 63) if self.pe32_plus else (4, 1 << 31)
        while rva:
            (thunk,) = struct.unpack_from("<Q" if self.pe32_plus else "<I", self.data, self.offset(rva))
            if thunk == 0:
                break
            names.add(f"#{thunk & 0xFFFF}" if thunk & ordinal_flag else self.string(thunk + 2))
            rva += width
        return frozenset(names)

    def imports(self) -> dict[str, frozenset[str]]:
        rva, size = self.directory(1)
        result: dict[str, frozenset[str]] = {}
        while rva and size:
            lookup, _, _, name, address = struct.unpack_from("<IIIII", self.data, self.offset(rva))
            if name == 0:
                break
            dll = self.string(name).lower()
            result[dll] = result.get(dll, frozenset()) | self.thunk_names(lookup or address)
            rva += 20
        return result

    def delay_imports(self) -> dict[str, frozenset[str]]:
        rva, size = self.directory(13)
        result: dict[str, frozenset[str]] = {}
        while rva and size:
            attributes, name, _, _, name_table = struct.unpack_from("<IIIII", self.data, self.offset(rva))
            if name == 0:
                break
            base = 0 if attributes & 1 else self.image_base
            dll = self.string(name - base).lower()
            result[dll] = result.get(dll, frozenset()) | self.thunk_names(name_table - base if name_table else 0)
            rva += 32
        return result

    def exports(self) -> tuple[frozenset[str], frozenset[str]]:
        rva, size = self.directory(0)
        if not rva or not size:
            return frozenset(), frozenset()
        base_offset = self.offset(rva)
        ordinal_base, function_count, name_count, functions, names = struct.unpack_from(
            "<IIIII", self.data, base_offset + 16
        )
        exports = {self.string(self.u32(names + 4 * index)) for index in range(name_count)}
        ordinals = {f"#{ordinal_base + index}" for index in range(function_count)
                    if self.u32(functions + 4 * index)}
        return frozenset(exports), frozenset(ordinals)

    def file_version(self) -> tuple[int, int, int, int] | None:
        rva, size = self.directory(2)
        if not rva or not size:
            return None
        for data_rva, data_size in self._resource_leaves(rva, rva, 16, depth=0):
            start = self.offset(data_rva)
            blob = self.data[start:start + data_size]
            signature = blob.find(struct.pack("<I", 0xFEEF04BD))
            if signature < 0:
                continue
            most, least = struct.unpack_from("<II", blob, signature + 8)
            return most >> 16, most & 0xFFFF, least >> 16, least & 0xFFFF
        return None

    def _resource_leaves(self, root: int, directory: int, type_id: int | None, depth: int):
        start = self.offset(directory)
        named, numbered = struct.unpack_from("<HH", self.data, start + 12)
        for index in range(named + numbered):
            name, target = struct.unpack_from("<II", self.data, start + 16 + 8 * index)
            if depth == 0 and (name & 0x80000000 or name != type_id):
                continue
            if target & 0x80000000:
                yield from self._resource_leaves(root, root + (target & 0x7FFFFFFF), None, depth + 1)
            else:
                yield struct.unpack_from("<II", self.data, self.offset(root + target))


def read_pe_bytes(name: str, data) -> PeImage:
    try:
        reader = _Reader(data)
        exports, export_ordinals = reader.exports()
        return PeImage(
            machine=reader.machine,
            linker_version=reader.linker_version,
            imports=reader.imports(),
            delay_imports=reader.delay_imports(),
            exports=exports,
            export_ordinals=export_ordinals,
            file_version=reader.file_version(),
            crt_sha256=hashlib.sha256(data).hexdigest() if is_vc_runtime(name) else None,
        )
    except (struct.error, ValueError, RecursionError) as error:
        raise PeError(f"{name}: malformed PE image: {error}") from error


def read_pe(path: Path) -> PeImage:
    with path.open("rb") as file:
        if os.fstat(file.fileno()).st_size == 0:
            raise PeError(f"{path.name}: empty PE image")
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return read_pe_bytes(path.name, data)


def read_bundle(directory: Path) -> dict[str, PeImage]:
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() == ".dll":
            name = path.name.lower()
            if name in images:
                raise PeError(f"{path.name}: duplicate DLL name")
            images[name] = read_pe(path)
    return images


def external_import(name: str, allow_vulkan: bool) -> bool:
    return is_os_provided(name) or (allow_vulkan and name == VULKAN_DRIVER_DLL)


def bundle_errors(images: dict[str, PeImage], arch: str, *, allow_vulkan: bool,
                  msvc_toolset: str | None) -> list[str]:
    """Check whether the Windows bundle resolves imports without an installed VC++ runtime."""
    errors: list[str] = []
    for name, image in sorted(images.items()):
        if image.machine != arch:
            errors.append(f"{name}: {image.machine} image in a {arch} bundle")
        if is_os_provided(name) or name == VULKAN_DRIVER_DLL or name.startswith(API_SET_PREFIXES):
            errors.append(f"{name}: the bundle must not ship Windows system files, API sets or GPU loaders")
        if is_non_redistributable(name):
            errors.append(f"{name}: a debug runtime or a Visual Studio debug_nonredist file, "
                          "which may not be redistributed")
        for dependency, symbols in sorted(image.all_imports().items()):
            if dependency in images:
                provided = images[dependency]
                missing = sorted(symbols - provided.exports - provided.export_ordinals)
                if missing:
                    errors.append(f"{name}: {dependency} does not export {', '.join(missing[:5])}")
            elif not external_import(dependency, allow_vulkan):
                errors.append(f"{name}: imports {dependency}, which neither the bundle nor Windows provides")
    errors.extend(runtime_version_errors(images, msvc_toolset))
    return errors


def require_bundle(images: dict[str, PeImage], primary: str, info: dict, *, allow_vulkan: bool) -> None:
    if primary.lower() not in images:
        raise PeError(f"Windows runtime bundle is missing {primary}")
    errors = bundle_errors(images, "x64", allow_vulkan=allow_vulkan,
                           msvc_toolset=info.get("windowsMsvcToolset"))
    errors.extend(crt_receipt_errors(images, info))
    if errors:
        raise PeError("Windows runtime bundle is incomplete:\n" + "\n".join(f"- {error}" for error in errors))


def runtime_version_errors(images: dict[str, PeImage], msvc_toolset: str | None) -> list[str]:
    runtime = {name: image.file_version for name, image in images.items() if is_vc_runtime(name)}
    if not runtime:
        return []
    errors = [f"{name}: no file version resource" for name, version in sorted(runtime.items()) if version is None]
    if errors:
        return errors
    if len(set(runtime.values())) > 1:
        listed = ", ".join(f"{name} {format_version(version)}" for name, version in sorted(runtime.items()))
        return [f"Visual C++ runtime DLLs come from different redistributables: {listed}"]
    version = next(iter(runtime.values()))
    if not isinstance(msvc_toolset, str) or not TOOLSET_VERSION.fullmatch(msvc_toolset):
        return ["missing or invalid full MSVC toolset version"]
    parts = tuple(map(int, msvc_toolset.split(".")))
    minimum = parts + (0,) * (4 - len(parts))
    if version < minimum:
        return [f"bundled Visual C++ runtime {format_version(version)} is older than "
                f"MSVC toolset {format_version(minimum)}"]

    errors = []
    for name, image in sorted(images.items()):
        if not any(is_vc_runtime(dependency) for dependency in image.all_imports()):
            continue
        # Major version 14 is the MSVC v14 toolset family; lld-link writes 14.0.
        if image.linker_version[0] == 14 and image.linker_version > version[:2]:
            errors.append(
                f"{name}: linked by MSVC {format_version(image.linker_version)}, newer than the bundled "
                f"Visual C++ runtime {format_version(version)}"
            )
    return errors


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def vc_redist_dir() -> Path:
    """Find the active MSVC toolset redistributables or the default x64 toolset."""
    configured = os.environ.get("VCToolsRedistDir")
    if configured:
        return Path(configured)
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    installation = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not installation:
        raise RuntimeError("vswhere found no Visual Studio installation with the MSVC x64 toolset")
    vc_dir = Path(installation) / "VC"
    version = (vc_dir / "Auxiliary" / "Build" / "Microsoft.VCRedistVersion.default.txt").read_text().strip()
    return vc_dir / "Redist" / "MSVC" / version


def crt_digests(images: dict[str, PeImage]) -> dict[str, str]:
    return {name: image.crt_sha256 for name, image in images.items() if image.crt_sha256 is not None}


def crt_receipt_errors(images: dict[str, PeImage], info: dict) -> list[str]:
    expected = crt_digests(images)
    if expected and info.get("windowsCrtSha256") != expected:
        return ["bundled Visual C++ DLLs do not match the Windows build's signature-checked bytes"]
    return []


def microsoft_signature_errors(paths: list[Path]) -> list[str]:
    if not paths:
        return []
    if sys.platform != "win32":
        return ["Microsoft Authenticode verification requires Windows"]
    script = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$paths = ConvertFrom-Json $env:SD_NATIVE_SIGNATURE_PATHS
$failed = $false
foreach ($path in $paths) {
    try {
        $signature = Get-AuthenticodeSignature -LiteralPath $path
    } catch {
        [Console]::WriteLine("$([IO.Path]::GetFileName($path)): Microsoft Authenticode verification failed ($($_.Exception.Message))")
        $failed = $true
        continue
    }
    if ($signature.Status -ne 'Valid' -or $signature.SignatureType -ne 'Authenticode' -or
        $signature.SignerCertificate.Subject -notmatch '(?:^|,\s*)O=Microsoft Corporation(?:,|$)') {
        [Console]::WriteLine("$([IO.Path]::GetFileName($path)): Microsoft Authenticode verification failed ($($signature.Status), $($signature.SignatureType), $($signature.SignerCertificate.Subject))")
        $failed = $true
    }
}
if ($failed) { exit 1 }
"""
    system_root = Path(os.environ["SystemRoot"])
    powershell = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    env = os.environ.copy()
    # A pwsh parent can pass down PowerShell 7 modules that Windows PowerShell cannot load.
    env.pop("PSMODULEPATH", None)
    env["SD_NATIVE_SIGNATURE_PATHS"] = json.dumps([str(path.resolve()) for path in paths])
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-OutputFormat", "Text", "-EncodedCommand",
         base64.b64encode(script.encode("utf-16-le")).decode("ascii")],
        env=env,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        return (result.stdout.strip() or result.stderr.strip()
                or "Microsoft Authenticode verification failed").splitlines()
    return []


def stage_vc_runtime(stage_dir: Path, arch: str, redist_dir: Path, *, allow_vulkan: bool,
                     msvc_toolset: str) -> list[str]:
    """Stage the VC++ redistributables needed by every DLL in the bundle."""
    staged: list[str] = []
    images = read_bundle(stage_dir)
    while True:
        missing = sorted({
            dependency
            for image in images.values()
            for dependency in image.all_imports()
            if dependency not in images and not external_import(dependency, allow_vulkan)
        })
        if not missing:
            break
        for name in missing:
            source = find_vc_redistributable(name, arch, redist_dir)
            if source is None:
                raise RuntimeError(
                    f"Windows bundle imports {name}, which is neither bundled, provided by "
                    f"Windows, nor in {redist_dir / arch / VC_RUNTIME_FOLDER}"
                )
            signature_errors = microsoft_signature_errors([source])
            if signature_errors:
                raise PeError("\n".join(signature_errors))
            shutil.copyfile(source, stage_dir / name)
            images[name] = read_pe(stage_dir / name)
            staged.append(name)
            print(f"Staged Visual C++ runtime {name} {format_version(images[name].file_version or ())}",
                  flush=True)
    errors = bundle_errors(images, arch, allow_vulkan=allow_vulkan, msvc_toolset=msvc_toolset)
    if errors:
        raise PeError("Windows runtime bundle is incomplete:\n" + "\n".join(f"- {error}" for error in errors))
    return staged


def dumpbin_errors(dumpbin: str, directory: Path, images: dict[str, PeImage]) -> list[str]:
    errors = []
    for name, image in sorted(images.items()):
        output = subprocess.run(
            [dumpbin, "/nologo", "/dependents", str(directory / name)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Dependency lines hold one bare file name; "Dump of file <path>" has spaces.
        listed = {
            line.strip().lower()
            for line in output.splitlines()
            if line.strip().lower().endswith(".dll") and " " not in line.strip()
        }
        parsed = set(image.all_imports())
        if listed != parsed:
            errors.append(
                f"{name}: dumpbin lists {sorted(listed - parsed)} and the PE reader {sorted(parsed - listed)} "
                "beyond each other"
            )
    return errors

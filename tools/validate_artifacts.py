#!/usr/bin/env python3
"""Validates built runtime payloads under `bin/<target>/`.

Checks that each library exports exactly the `SD_API` symbols of the shipped
headers (so no ggml symbol leaks) and, except on Windows, links only against
allowlisted system libraries.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

from build import ANDROID_PAGE_SIZE, BIN_ROOT, HEADERS, TARGETS, Target, llvm_tool
from sd_api import api_symbols

APPLE_SYSTEM_DEPENDENCIES = (
    r"/usr/lib/libSystem\.B\.dylib"
    r"|/usr/lib/libc\+\+\.1\.dylib"
    r"|/usr/lib/libobjc\.A\.dylib"
    r"|/System/Library/Frameworks/"
    r"(Accelerate|CoreFoundation|Foundation|Metal|MetalKit)\.framework/.+"
)
APPLE_DEPENDENCIES = re.compile(
    rf"^(@rpath/libstable-diffusion\.dylib|{APPLE_SYSTEM_DEPENDENCIES})$"
)
ELF_DEPENDENCIES = {
    "android": {"libc.so", "libm.so", "libdl.so", "liblog.so"},
    "linux": {
        "libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1",
        "libpthread.so.0", "libdl.so.2", "ld-linux-x86-64.so.2",
        "ld-linux-aarch64.so.1",
    },
}


def output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pe_exported_symbols(library: Path) -> set[str]:
    """Reads the export name table of a DLL; no Windows SDK tool is needed."""
    data = library.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    section_count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    # Data directories follow a header that is longer in PE32+ (magic 0x20B).
    is_pe32_plus = struct.unpack_from("<H", data, optional)[0] == 0x20B
    export_rva = struct.unpack_from("<I", data, optional + (112 if is_pe32_plus else 96))[0]
    sections = [struct.unpack_from("<IIII", data, optional + optional_size + 40 * i + 8)
                for i in range(section_count)]

    def offset(rva: int) -> int:
        for virtual_size, virtual_address, raw_size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return rva - virtual_address + raw_offset
        raise ValueError(f"RVA {rva:#x} is outside every section of {library}")

    if export_rva == 0:
        return set()
    directory = offset(export_rva)
    name_count = struct.unpack_from("<I", data, directory + 24)[0]
    names_rva = struct.unpack_from("<I", data, directory + 32)[0]
    names = set()
    for name_rva in struct.unpack_from(f"<{name_count}I", data, offset(names_rva)):
        start = offset(name_rva)
        names.add(data[start:data.index(b"\0", start)].decode())
    return names


def exported_symbols(target: Target, library: Path) -> set[str]:
    if target.os == "windows":
        return pe_exported_symbols(library)
    if target.os in ("macos", "ios"):
        names = output(["nm", "-gUj", str(library)]).split()
        return {n[1:] if n.startswith("_") else n for n in names}
    nm = llvm_tool(target, "llvm-nm") if target.os == "android" else "nm"
    lines = output([nm, "-D", "--defined-only", str(library)])
    return {line.split()[-1] for line in lines.splitlines() if line.strip()}


def dependencies(target: Target, library: Path) -> list[str]:
    if target.os in ("macos", "ios"):
        lines = output(["otool", "-L", str(library)]).splitlines()[1:]
        return [line.split(" (")[0].strip() for line in lines]
    readelf = llvm_tool(target, "llvm-readelf") if target.os == "android" else "readelf"
    lines = output([readelf, "-d", str(library)]).splitlines()
    return re.findall(r"Shared library: \[([^\]]+)\]", "\n".join(lines))


def load_alignments(target: Target, library: Path) -> list[int]:
    headers = output([llvm_tool(target, "llvm-readelf"), "-lW", str(library)])
    return [
        int(line.split()[-1], 16)
        for line in headers.splitlines()
        if line.strip().startswith("LOAD")
    ]


def validate(target: Target) -> list[str]:
    root = BIN_ROOT / target.name
    library = root / "lib" / target.library
    problems: list[str] = []
    headers = [root / "include" / header.name for header in HEADERS]
    for path in (library, *headers, root / "build-info.json"):
        if not path.is_file():
            problems.append(f"missing {path.relative_to(BIN_ROOT)}")
    if problems:
        return problems

    info = json.loads((root / "build-info.json").read_text())
    if info.get("target") != target.name:
        problems.append(f"build-info target is {info.get('target')!r}")

    expected = set(api_symbols(*HEADERS))
    exported = exported_symbols(target, library)
    missing = sorted(expected - exported)
    leaked = sorted(exported - expected)
    if missing:
        problems.append(f"missing API exports: {', '.join(missing[:10])}")
    if leaked:
        problems.append(
            f"{len(leaked)} non-API exports, e.g. {', '.join(leaked[:10])}"
        )

    if target.os == "android":
        small = [a for a in load_alignments(target, library) if a < ANDROID_PAGE_SIZE]
        if small:
            problems.append(
                f"LOAD segments aligned to {', '.join(hex(a) for a in small)}; "
                f"Android needs {hex(ANDROID_PAGE_SIZE)} for 16 KB pages"
            )

    if target.os == "windows":
        return problems

    for dep in dependencies(target, library):
        if target.os in ("macos", "ios"):
            allowed = APPLE_DEPENDENCIES.match(dep)
        else:
            allowed = dep in ELF_DEPENDENCIES[target.os] or (
                "vulkan" in target.accelerators and dep == "libvulkan.so.1"
            )
        if not allowed:
            problems.append(f"unexpected dependency {dep}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Targets to check; default: all built.")
    args = parser.parse_args()
    names = args.targets or [t for t in TARGETS if (BIN_ROOT / t).is_dir()]
    if not names:
        print("error: no built targets under bin/", file=sys.stderr)
        sys.exit(1)
    failed = False
    for name in names:
        problems = validate(TARGETS[name])
        status = "ok" if not problems else "FAILED"
        print(f"{name}: {status}")
        for problem in problems:
            print(f"  - {problem}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

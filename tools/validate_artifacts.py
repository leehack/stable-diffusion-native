#!/usr/bin/env python3
"""Validates built runtime payloads under `bin/<target>/`.

Checks that each library exports exactly the `stable-diffusion.h` API (so no
ggml symbol leaks) and links only against allowlisted system libraries.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from build import ANDROID_PAGE_SIZE, BIN_ROOT, HEADER, TARGETS, Target, llvm_tool
from sd_api import api_symbols

APPLE_DEPENDENCIES = re.compile(
    r"^(@rpath/libstable-diffusion\.dylib"
    r"|/usr/lib/libSystem\.B\.dylib"
    r"|/usr/lib/libc\+\+\.1\.dylib"
    r"|/usr/lib/libobjc\.A\.dylib"
    r"|/System/Library/Frameworks/"
    r"(Accelerate|CoreFoundation|Foundation|Metal|MetalKit)\.framework/.+)$"
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


def exported_symbols(target: Target, library: Path) -> set[str]:
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
    for path in (library, root / "include" / HEADER.name, root / "build-info.json"):
        if not path.is_file():
            problems.append(f"missing {path.relative_to(BIN_ROOT)}")
    if problems:
        return problems

    info = json.loads((root / "build-info.json").read_text())
    if info.get("target") != target.name:
        problems.append(f"build-info target is {info.get('target')!r}")

    if target.os == "windows":
        return problems

    expected = set(api_symbols(HEADER))
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

#!/usr/bin/env python3
"""Validates built runtime payloads under `bin/<target>/`.

Checks that each library exports exactly the `SD_API` symbols of the shipped
headers (so no ggml symbol leaks) and resolves imports from the bundle or
allowlisted platform libraries. An Apple or Linux library must also register its
static destructors through its own `__cxa_atexit`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from build import ANDROID_PAGE_SIZE, BIN_ROOT, HEADERS, TARGETS, Target, llvm_tool
from sd_api import api_symbols
import windows_runtime

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
# `src/sd_dart_exit.cpp` defines `__cxa_atexit`, so that every static destructor
# of an Apple library runs exit teardown first. A library that imports the
# system function registers its destructors past that definition, and exit
# teardown then runs after the statics it needs are gone. This only shows that
# the definition is there. That it wraps each destructor is what
# `tests/test_exit_teardown.py` and `tests/test_exit_teardown_runtime.py` show.
APPLE_STATIC_DESTRUCTOR_IMPORT = "___cxa_atexit"
# The same definition is in a Linux library, where it registers the wait of
# exit teardown in the place of each destructor. A library that imports the C
# library's function has its statics destroyed by exit() under the threads that
# are still inside it (leehack/llamadart#949). Android libraries are left as
# they were and import it.
ELF_STATIC_DESTRUCTOR_IMPORT = "__cxa_atexit"
# ggml's host buffers on iOS come from `posix_memalign`, and libmalloc keeps
# their freed pages in the app's memory footprint for as long as it likes, so a
# model that is unloaded still counts toward the jetsam limit
# (leehack/llamadart#2004). `patches/ggml_apple_vm_allocate.patch` makes every
# Apple build use `vm_allocate`, as macOS already did.
IOS_HOST_BUFFER_IMPORT = "_posix_memalign"
# `std::ios_base::Init::Init()`. Before GCC 13 every source that includes
# <iostream> holds an object of that class, whose destructor flushes the
# standard streams when the last one goes. A Linux library drops that
# destructor with the others, and a host that writes to `std::cout` after
# `std::ios::sync_with_stdio(false)` then loses what it had not flushed.
ELF_STREAM_INIT_IMPORT = "_ZNSt8ios_base4InitC1Ev"
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


def imports_symbol(library: Path, symbol: str, arch: str | None = None) -> bool:
    """Whether a Mach-O file imports `symbol`; a universal file needs `arch`."""
    select = ["-arch", arch] if arch else []
    return symbol in output(["nm", "-uj", *select, str(library)]).split()


def elf_undefined_symbols(dynamic_symbols: str) -> set[str]:
    """The names a `readelf --dyn-syms -W` listing leaves undefined, without versions."""
    names = set()
    for line in dynamic_symbols.splitlines():
        # Num: Value Size Type Bind Vis Ndx Name, and GNU readelf may add the
        # version's index in brackets.
        fields = line.split()
        if len(fields) >= 8 and fields[0].rstrip(":").isdigit() and fields[6] == "UND":
            names.add(fields[7].split("@")[0])
    return names


def elf_imports_symbol(library: Path, symbol: str, readelf: str = "readelf") -> bool:
    """Whether an ELF file imports `symbol` from another library."""
    return symbol in elf_undefined_symbols(output([readelf, "--dyn-syms", "-W", str(library)]))


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


def validate(target: Target, *, dumpbin: str | None = None) -> list[str]:
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

    if 'vulkan' in target.accelerators:
        from vulkan_tri import validate_artifact
        try:
            validate_artifact(library, info.get('vulkanTriF16'))
        except ValueError as error:
            problems.append(str(error))

    expected = set(api_symbols(*HEADERS))
    if target.os == "windows":
        images = windows_runtime.read_bundle(library.parent)
        exported = images[target.library.lower()].exports
        problems.extend(windows_runtime.bundle_errors(
            images, "x64", allow_vulkan="vulkan" in target.accelerators,
            msvc_toolset=info.get("windowsMsvcToolset"),
        ))
        problems.extend(windows_runtime.crt_receipt_errors(images, info))
        problems.extend(windows_runtime.microsoft_signature_errors([
            path for path in sorted(library.parent.iterdir())
            if path.is_file() and windows_runtime.is_vc_runtime(path.name)
        ]))
        if dumpbin:
            problems.extend(windows_runtime.dumpbin_errors(dumpbin, library.parent, images))
    else:
        exported = exported_symbols(target, library)
    missing = sorted(expected - exported)
    leaked = sorted(exported - expected)
    if missing:
        problems.append(f"missing API exports: {', '.join(missing[:10])}")
    if leaked:
        problems.append(
            f"{len(leaked)} non-API exports, e.g. {', '.join(leaked[:10])}"
        )

    if target.os in ("macos", "ios") and imports_symbol(library, APPLE_STATIC_DESTRUCTOR_IMPORT):
        problems.append(
            f"imports {APPLE_STATIC_DESTRUCTOR_IMPORT}: static destructors bypass exit teardown"
        )

    if target.os == "ios" and imports_symbol(library, IOS_HOST_BUFFER_IMPORT):
        problems.append(
            f"imports {IOS_HOST_BUFFER_IMPORT}: libmalloc keeps freed host-buffer pages in the app footprint"
        )

    if target.os == "linux":
        imports = elf_undefined_symbols(output(["readelf", "--dyn-syms", "-W", str(library)]))
        if ELF_STATIC_DESTRUCTOR_IMPORT in imports:
            problems.append(
                f"imports {ELF_STATIC_DESTRUCTOR_IMPORT}: exit() destroys statics under calls in flight"
            )
        if ELF_STREAM_INIT_IMPORT in imports:
            problems.append(
                f"imports {ELF_STREAM_INIT_IMPORT}: built with a compiler older than GCC 13, "
                "so the host's unflushed std::cout is lost at exit"
            )

    if target.os == "android":
        small = [a for a in load_alignments(target, library) if a < ANDROID_PAGE_SIZE]
        if small:
            problems.append(
                f"LOAD segments aligned to {', '.join(hex(a) for a in small)}; "
                f"Android needs {hex(ANDROID_PAGE_SIZE)} for 16 KB pages"
            )

    for dep in dependencies(target, library) if target.os != "windows" else []:
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
    parser.add_argument("--dumpbin", help="cross-check Windows imports using dumpbin /dependents")
    args = parser.parse_args()
    names = args.targets or [t for t in TARGETS if (BIN_ROOT / t).is_dir()]
    if not names:
        print("error: no built targets under bin/", file=sys.stderr)
        sys.exit(1)
    failed = False
    for name in names:
        try:
            problems = validate(TARGETS[name], dumpbin=args.dumpbin)
        except windows_runtime.PeError as error:
            problems = [str(error)]
        status = "ok" if not problems else "FAILED"
        print(f"{name}: {status}")
        for problem in problems:
            print(f"  - {problem}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

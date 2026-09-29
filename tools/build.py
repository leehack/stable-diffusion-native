#!/usr/bin/env python3
"""Builds stable-diffusion.cpp runtime libraries per target.

Each build exports only the public `stable-diffusion.h` API so the bundled
ggml copy cannot collide with another ggml loaded in the same process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sd_api import api_symbols

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = REPO_ROOT / "third_party" / "stable-diffusion.cpp"
HEADER = UPSTREAM_DIR / "include" / "stable-diffusion.h"
BUILD_ROOT = REPO_ROOT / "build"
BIN_ROOT = REPO_ROOT / "bin"

APPLE_MACOS_MIN = "13.3"
APPLE_IOS_MIN = "16.4"
ANDROID_PLATFORM = "android-28"
# Armv8.2 dot-product and fp16 arithmetic: every arm64 SoC shipped with
# Cortex-A55/A75 or newer. Consumers must probe `asimddp` before loading.
ANDROID_ARM64_CPU_ARCH = "armv8.2-a+dotprod+fp16"

COMMON_CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DSD_BUILD_SHARED_LIBS=ON",
    "-DSD_BUILD_EXAMPLES=OFF",
    "-DSD_WEBP=OFF",
    "-DSD_WEBM=OFF",
    "-DGGML_NATIVE=OFF",
    "-DGGML_OPENMP=OFF",
]


@dataclass(frozen=True)
class Target:
    name: str
    os: str
    library: str
    accelerators: tuple[str, ...]
    cmake_args: tuple[str, ...] = field(default_factory=tuple)
    host: str = ""
    # Native targets must build on a matching host CPU; cross targets leave it empty.
    host_arch: str = ""


def _apple(name: str, sdk: str, arch: str, min_version: str) -> Target:
    system = "Darwin" if sdk == "macosx" else "iOS"
    args = [
        f"-DCMAKE_SYSTEM_NAME={system}",
        f"-DCMAKE_OSX_SYSROOT={sdk}",
        f"-DCMAKE_OSX_ARCHITECTURES={arch}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={min_version}",
        "-DSD_METAL=ON",
        "-DGGML_METAL_EMBED_LIBRARY=ON",
    ]
    return Target(
        name,
        "macos" if sdk == "macosx" else "ios",
        "libstable-diffusion.dylib",
        ("metal", "cpu"),
        tuple(args),
        host="darwin",
    )


TARGETS: dict[str, Target] = {
    t.name: t
    for t in [
        _apple("macos-arm64", "macosx", "arm64", APPLE_MACOS_MIN),
        _apple("macos-x64", "macosx", "x86_64", APPLE_MACOS_MIN),
        _apple("ios-arm64", "iphoneos", "arm64", APPLE_IOS_MIN),
        _apple("ios-arm64-sim", "iphonesimulator", "arm64", APPLE_IOS_MIN),
        Target(
            "android-arm64",
            "android",
            "libstable-diffusion.so",
            ("cpu",),
            (
                "-DANDROID_ABI=arm64-v8a",
                f"-DANDROID_PLATFORM={ANDROID_PLATFORM}",
                f"-DGGML_CPU_ARM_ARCH={ANDROID_ARM64_CPU_ARCH}",
            ),
        ),
        Target("linux-x64", "linux", "libstable-diffusion.so", ("cpu",),
               host="linux", host_arch="x86_64"),
        Target("linux-arm64", "linux", "libstable-diffusion.so", ("cpu",),
               host="linux", host_arch="aarch64"),
        Target("windows-x64", "windows", "stable-diffusion.dll", ("cpu",),
               host="windows", host_arch="amd64"),
    ]
}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def android_ndk() -> Path:
    for key in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK_LATEST_HOME"):
        value = os.environ.get(key)
        if value and Path(value, "build/cmake/android.toolchain.cmake").is_file():
            return Path(value)
    sdk = Path(os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk"))
    candidates = sorted((sdk / "ndk").glob("*/build/cmake/android.toolchain.cmake"))
    if candidates:
        return candidates[-1].parent.parent.parent
    fail("Android NDK not found; set ANDROID_NDK_HOME.")
    raise AssertionError


def llvm_tool(target: Target, name: str) -> str:
    if target.os == "android":
        prebuilt = android_ndk() / "toolchains" / "llvm" / "prebuilt"
        host_dirs = sorted(prebuilt.iterdir())
        return str(host_dirs[0] / "bin" / name)
    return shutil.which(name) or name


def export_linker_flags(target: Target, work_dir: Path) -> list[str]:
    symbols = api_symbols(HEADER)
    if target.os in ("macos", "ios"):
        path = work_dir / "exported_symbols.txt"
        path.write_text("".join(f"_{s}\n" for s in symbols))
        return [f"-Wl,-exported_symbols_list,{path}"]
    if target.os in ("android", "linux"):
        path = work_dir / "exports.map"
        body = "".join(f"    {s};\n" for s in symbols)
        path.write_text(f"{{\n  global:\n{body}  local:\n    *;\n}};\n")
        return [f"-Wl,--version-script={path}"]
    # Windows exports only SD_API (dllexport) declarations already.
    return []


def configure_args(target: Target, work_dir: Path) -> list[str]:
    args = [*COMMON_CMAKE_ARGS, *target.cmake_args]
    if target.os == "android":
        toolchain = android_ndk() / "build" / "cmake" / "android.toolchain.cmake"
        args.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
    if shutil.which("ninja"):
        args += ["-G", "Ninja"]
    flags = " ".join(export_linker_flags(target, work_dir))
    if flags:
        args.append(f"-DCMAKE_SHARED_LINKER_FLAGS={flags}")
    return args


def find_library(build_dir: Path, library: str) -> Path:
    matches = [p for p in build_dir.rglob(library) if p.is_file() and not p.is_symlink()]
    if not matches:
        fail(f"{library} not produced under {build_dir}")
    return matches[0]


def strip(target: Target, source: Path, dest: Path) -> None:
    if target.os in ("macos", "ios"):
        shutil.copy2(source, dest)
        run(["strip", "-x", str(dest)])
    elif target.os in ("android", "linux"):
        run([llvm_tool(target, "llvm-strip") if target.os == "android" else "strip",
             "--strip-unneeded", "-o", str(dest), str(source)])
    else:
        shutil.copy2(source, dest)


def upstream_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM_DIR, text=True
    ).strip()


def build(target: Target, jobs: int) -> Path:
    if not HEADER.is_file():
        fail("third_party/stable-diffusion.cpp is missing; run "
             "`git submodule update --init --recursive`.")
    work_dir = BUILD_ROOT / target.name
    work_dir.mkdir(parents=True, exist_ok=True)
    cmake_args = configure_args(target, work_dir)
    run(["cmake", "-S", str(UPSTREAM_DIR), "-B", str(work_dir), *cmake_args])
    run(["cmake", "--build", str(work_dir), "--config", "Release", "-j", str(jobs)])

    out_dir = BIN_ROOT / target.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "lib").mkdir(parents=True)
    (out_dir / "symbols").mkdir()
    (out_dir / "include").mkdir()

    built = find_library(work_dir, target.library)
    shutil.copy2(built, out_dir / "symbols" / target.library)
    strip(target, built, out_dir / "lib" / target.library)
    if target.os == "windows":
        import_lib = next(work_dir.rglob("stable-diffusion.lib"), None)
        if import_lib:
            shutil.copy2(import_lib, out_dir / "lib" / import_lib.name)
    shutil.copy2(HEADER, out_dir / "include" / HEADER.name)

    info = {
        "target": target.name,
        "os": target.os,
        "library": target.library,
        "accelerators": list(target.accelerators),
        "upstream": {
            "repository": "leejet/stable-diffusion.cpp",
            "commit": upstream_commit(),
        },
        "cmakeArgs": [a for a in cmake_args if not a.startswith("-DCMAKE_SHARED_LINKER_FLAGS")],
        "exportedSymbols": api_symbols(HEADER),
    }
    (out_dir / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"built {target.name}: {out_dir / 'lib' / target.library}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List build targets.")
    build_parser = sub.add_parser("build", help="Build one or more targets.")
    build_parser.add_argument("--target", action="append", required=True,
                              help="Target name, repeatable, or `all-host`.")
    build_parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    if args.command == "list":
        for target in TARGETS.values():
            print(f"{target.name:16} {target.library:28} {','.join(target.accelerators)}")
        return

    host = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}[platform.system()]
    host_arch = {"arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    names: list[str] = []
    for name in args.target:
        if name == "all-host":
            names += [
                t.name for t in TARGETS.values()
                if t.host in ("", host) and t.host_arch in ("", host_arch)
            ]
        elif name in TARGETS:
            names.append(name)
        else:
            fail(f"unknown target {name}; see `build.py list`.")
    for name in names:
        target = TARGETS[name]
        if target.host and target.host != host:
            fail(f"{name} must be built on a {target.host} host.")
        if target.host_arch and target.host_arch != host_arch:
            fail(f"{name} must be built on a {target.host_arch} host, not {host_arch}.")
        build(target, args.jobs)


if __name__ == "__main__":
    main()

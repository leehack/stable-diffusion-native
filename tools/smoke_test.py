#!/usr/bin/env python3
"""Loads a built runtime and lists its devices through the C API.

`--expect-device vulkan` fails unless a device name starts with that prefix,
which checks that a GPU backend initialized rather than silently falling back
to the CPU.

Also checks the `sd_dart_wrapper.h` progress recorder against real progress,
which converting a few synthetic tensors reports without needing a model, and
that the GPU device memory exports agree with the device list.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from build import BIN_ROOT, TARGETS

PROGRESS_TENSORS = 8
PROGRESS_MODES = ("upstream", "recorded")


GPU_OK, GPU_NO_BACKEND, GPU_NO_DEVICE, GPU_UNAVAILABLE = 0, -2, -3, -4
GPU_DEFAULT_DEVICE = -1
GPU_STATUS_NAMES = {GPU_OK: "OK", -1: "INVALID_ARGUMENT", GPU_NO_BACKEND: "NO_BACKEND",
                    GPU_NO_DEVICE: "NO_DEVICE", GPU_UNAVAILABLE: "UNAVAILABLE"}
GPU_DEVICE_TYPES = {1: "discrete", 2: "integrated"}


class GpuDeviceMemory(ctypes.Structure):
    _fields_ = [("total_bytes", ctypes.c_uint64), ("free_bytes", ctypes.c_uint64),
                ("type", ctypes.c_int32), ("name", ctypes.c_char * 64),
                ("description", ctypes.c_char * 256)]


class Progress(ctypes.Structure):
    _fields_ = [("sequence", ctypes.c_uint64), ("step", ctypes.c_int32),
                ("steps", ctypes.c_int32), ("time", ctypes.c_float)]


def write_safetensors(path: Path) -> None:
    header, offset = {}, 0
    for index in range(PROGRESS_TENSORS):
        header[f"probe.weight.{index}"] = {
            "dtype": "F32", "shape": [4, 4], "data_offsets": [offset, offset + 64]}
        offset += 64
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def run_progress_probe(library: str, mode: str, work_dir: str) -> None:
    """Converts the synthetic tensors, with the recorder enabled if `mode` says.

    Runs in a child process so that the parent sees exactly what the runtime
    printed to stdout. Writes what the recorder held before and after to
    `progress.json`: the latest sequence, then each report.
    """
    lib = ctypes.CDLL(library)
    lib.sd_dart_progress_enable.argtypes = []
    lib.sd_dart_progress_enable.restype = None
    lib.sd_dart_progress_read.argtypes = [ctypes.c_uint64, ctypes.POINTER(Progress),
                                          ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    lib.sd_dart_progress_read.restype = ctypes.c_size_t
    lib.str_to_sd_type.argtypes = [ctypes.c_char_p]
    lib.convert.restype = ctypes.c_bool
    lib.convert.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                            ctypes.c_int, ctypes.c_char_p, ctypes.c_bool]

    def read() -> list[list[int]]:
        reports = (Progress * (2 * PROGRESS_TENSORS))()
        latest = ctypes.c_uint64(1)
        count = lib.sd_dart_progress_read(0, reports, len(reports), ctypes.byref(latest))
        return [[latest.value], *([r.sequence, r.step, r.steps] for r in reports[:count])]

    if mode == "recorded":
        lib.sd_dart_progress_enable()
        lib.sd_dart_progress_enable()
    before = read()
    work = Path(work_dir)
    write_safetensors(work / "probe.safetensors")
    converted = lib.convert(os.fsencode(work / "probe.safetensors"), None,
                            os.fsencode(work / "probe.gguf"),
                            lib.str_to_sd_type(b"f32"), b"", False)
    (work / "progress.json").write_text(
        json.dumps({"converted": converted, "before": before, "after": read()}))


def check_progress_recording(library: Path) -> list[str]:
    problems: list[str] = []
    last_step = f"{PROGRESS_TENSORS}/{PROGRESS_TENSORS}"
    for mode in PROGRESS_MODES:
        with tempfile.TemporaryDirectory() as work_dir:
            child = subprocess.run(
                [sys.executable, __file__, "--progress-probe", mode, str(library), work_dir],
                stdout=subprocess.PIPE)
            result_path = Path(work_dir) / "progress.json"
            if child.returncode != 0 or not result_path.is_file():
                problems.append(f"progress probe {mode} exited with {child.returncode}")
                continue
            result = json.loads(result_path.read_text())
        printed_bar = last_step.encode() in child.stdout
        if not result["converted"]:
            problems.append(f"progress probe {mode} could not convert its tensors")
        # Without the recorder the runtime prints a bar, which shows that the
        # probe would notice one, and records nothing.
        if printed_bar != (mode == "upstream"):
            problems.append(f"progress probe {mode} printed {child.stdout!r}")
        expected = [[0]]
        if mode == "recorded":
            expected = [[PROGRESS_TENSORS], *([step, step, PROGRESS_TENSORS]
                                              for step in range(1, PROGRESS_TENSORS + 1))]
        if result["before"] != [[0]] or result["after"] != expected:
            problems.append(f"progress probe {mode} read {result['before']} before "
                            f"and {result['after']} after, expected {expected}")
    return problems


def host_memory() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def check_gpu_device_memory(lib: ctypes.CDLL, has_gpu_backend: bool, listed: list[str],
                            expected: list[str]) -> list[str]:
    """Compares the GPU memory exports with `listed`, the names of `sd_list_devices`.

    `expected` are the device name prefixes the caller requires: a GPU among
    them must report its memory.
    """
    lib.sd_dart_gpu_device_count.restype = ctypes.c_int32
    lib.sd_dart_gpu_device_memory.restype = ctypes.c_int32
    lib.sd_dart_gpu_device_memory.argtypes = [ctypes.c_int32, ctypes.POINTER(GpuDeviceMemory)]

    def query(index: int) -> tuple[int, GpuDeviceMemory]:
        memory = GpuDeviceMemory()
        return lib.sd_dart_gpu_device_memory(index, ctypes.byref(memory)), memory

    def status_name(status: int) -> str:
        return GPU_STATUS_NAMES.get(status, str(status))

    problems: list[str] = []
    count = lib.sd_dart_gpu_device_count()
    default_status, default = query(GPU_DEFAULT_DEVICE)
    if not has_gpu_backend:
        if (count, default_status) != (GPU_NO_BACKEND, GPU_NO_BACKEND):
            problems.append(f"a build without a GPU backend reported {count} GPU devices and "
                            f"{status_name(default_status)} for the default one")
        print("  gpu memory: no GPU backend in this build")
        return problems
    if lib.sd_dart_gpu_device_memory(0, None) != -1:
        problems.append("a null result struct was not refused")
    if count < 0:
        problems.append(f"the GPU device count is {status_name(count)}")
        return problems
    if query(count)[0] != GPU_NO_DEVICE:
        problems.append(f"device {count}, past the last one, is not NO_DEVICE")
    names = []
    for index in range(count):
        status, memory = query(index)
        if status == GPU_UNAVAILABLE:
            print(f"  gpu memory: device {index} reports none")
            continue
        if status != GPU_OK:
            problems.append(f"device {index} returned {status_name(status)}")
            continue
        name = memory.name.decode()
        names.append(name)
        print(f"  gpu memory: {name} ({GPU_DEVICE_TYPES.get(memory.type, memory.type)}, "
              f"{memory.description.decode()}): {memory.free_bytes / 2**20:.0f} MiB free of "
              f"{memory.total_bytes / 2**20:.0f} MiB")
        if memory.type not in GPU_DEVICE_TYPES:
            problems.append(f"{name} has type {memory.type}")
        if not 0 < memory.total_bytes < 2**50 or memory.free_bytes > memory.total_bytes:
            problems.append(f"{name} reports {memory.free_bytes} bytes free of "
                            f"{memory.total_bytes}")
        # Metal's working set is a share of the host's memory.
        if name.startswith("MTL") and memory.total_bytes > host_memory():
            problems.append(f"{name} reports more memory than the host has")
    # GPU devices keep the order of the list, which has other devices between.
    remaining = iter(listed)
    if not all(name in remaining for name in names):
        problems.append(f"GPU devices {names} are not in sd_list_devices order: {listed}")
    if count == 0:
        if default_status != GPU_NO_DEVICE:
            problems.append(f"no GPU device, but the default one is "
                            f"{status_name(default_status)}")
        print("  gpu memory: no GPU device")
    elif default_status == GPU_OK and default.name.decode() not in names:
        problems.append(f"the default device {default.name.decode()} is not among {names}")
    elif default_status not in (GPU_OK, GPU_UNAVAILABLE):
        problems.append(f"the default device returned {status_name(default_status)}")
    for prefix in expected:
        if prefix.lower() == "cpu":
            continue
        if not any(name.lower().startswith(prefix.lower()) for name in names):
            problems.append(f"no {prefix} device reports its memory")
    return problems


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--progress-probe":
        run_progress_probe(sys.argv[3], sys.argv[2], sys.argv[4])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("--expect-device", action="append", default=[],
                        help="Device name prefix that must be listed, e.g. Vulkan, MTL, CPU.")
    args = parser.parse_args()

    target = TARGETS[args.target]
    library = BIN_ROOT / target.name / "lib" / target.library
    lib = ctypes.CDLL(str(library))
    lib.sd_version.restype = ctypes.c_char_p
    lib.sd_list_devices.restype = ctypes.c_size_t
    lib.sd_list_devices.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

    size = lib.sd_list_devices(None, 0)
    buffer = ctypes.create_string_buffer(size + 1)
    lib.sd_list_devices(buffer, size + 1)
    devices = [line for line in buffer.value.decode().splitlines() if line]

    print(f"{target.name}: stable-diffusion.cpp {lib.sd_version().decode()}")
    for device in devices:
        print(f"  {device}")
    missing = [p for p in args.expect_device
               if not any(d.lower().startswith(p.lower()) for d in devices)]
    if missing:
        print(f"error: no device starting with {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    problems = check_gpu_device_memory(
        lib, any(a != "cpu" for a in target.accelerators),
        [device.split("\t")[0] for device in devices], args.expect_device)
    problems += check_progress_recording(library)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print("  progress recording ok")


if __name__ == "__main__":
    main()

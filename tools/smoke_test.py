#!/usr/bin/env python3
"""Loads a built runtime and lists its devices through the C API.

`--expect-device vulkan` fails unless a device name starts with that prefix,
which checks that a GPU backend initialized rather than silently falling back
to the CPU.

Also checks the `sd_dart_wrapper.h` progress recorder against real progress,
which converting a few synthetic tensors reports without needing a model.
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

    problems = check_progress_recording(library)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print("  progress recording ok")


if __name__ == "__main__":
    main()

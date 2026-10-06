#!/usr/bin/env python3
"""Loads a built runtime and lists its devices through the C API.

`--expect-device vulkan` fails unless a device name starts with that prefix,
which checks that a GPU backend initialized rather than silently falling back
to the CPU.

Also checks the `sd_dart_wrapper.h` progress routing against real progress,
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

PROGRESS_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p)
PROGRESS_TENSORS = 8
PROGRESS_MODES = ("upstream", "callback", "cleared", "discard")


def write_safetensors(path: Path) -> None:
    header, offset = {}, 0
    for index in range(PROGRESS_TENSORS):
        header[f"probe.weight.{index}"] = {
            "dtype": "F32", "shape": [4, 4], "data_offsets": [offset, offset + 64]}
        offset += 64
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def run_progress_probe(library: str, mode: str, work_dir: str) -> None:
    """Converts the synthetic tensors with progress routed as `mode` says.

    Runs in a child process so that the parent sees exactly what the runtime
    printed to stdout. Writes the progress calls it received to `calls.json`.
    """
    lib = ctypes.CDLL(library)
    lib.sd_dart_set_progress_callback.argtypes = [PROGRESS_CALLBACK, ctypes.c_void_p]
    lib.sd_dart_set_progress_callback.restype = None
    lib.sd_dart_clear_progress_callback.argtypes = [ctypes.c_void_p]
    lib.sd_dart_clear_progress_callback.restype = None
    lib.str_to_sd_type.argtypes = [ctypes.c_char_p]
    lib.convert.restype = ctypes.c_bool
    lib.convert.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                            ctypes.c_int, ctypes.c_char_p, ctypes.c_bool]

    calls: list[list[int]] = []
    callback = PROGRESS_CALLBACK(lambda step, steps, _time, _data: calls.append([step, steps]))
    if mode in ("callback", "cleared"):
        lib.sd_dart_set_progress_callback(callback, None)
    if mode == "cleared":
        lib.sd_dart_clear_progress_callback(ctypes.cast(callback, ctypes.c_void_p))
    if mode == "discard":
        lib.sd_dart_set_progress_callback(PROGRESS_CALLBACK(0), None)

    work = Path(work_dir)
    write_safetensors(work / "probe.safetensors")
    converted = lib.convert(os.fsencode(work / "probe.safetensors"), None,
                            os.fsencode(work / "probe.gguf"),
                            lib.str_to_sd_type(b"f32"), b"", False)
    (work / "calls.json").write_text(json.dumps({"converted": converted, "calls": calls}))


def check_progress_routing(library: Path) -> list[str]:
    problems: list[str] = []
    last_step = f"{PROGRESS_TENSORS}/{PROGRESS_TENSORS}"
    all_steps = [[step, PROGRESS_TENSORS] for step in range(1, PROGRESS_TENSORS + 1)]
    for mode in PROGRESS_MODES:
        with tempfile.TemporaryDirectory() as work_dir:
            child = subprocess.run(
                [sys.executable, __file__, "--progress-probe", mode, str(library), work_dir],
                stdout=subprocess.PIPE)
            result_path = Path(work_dir) / "calls.json"
            if child.returncode != 0 or not result_path.is_file():
                problems.append(f"progress probe {mode} exited with {child.returncode}")
                continue
            result = json.loads(result_path.read_text())
        printed_bar = last_step.encode() in child.stdout
        if not result["converted"]:
            problems.append(f"progress probe {mode} could not convert its tensors")
        # Without the wrapper the runtime prints a bar; the other modes are
        # checked against a probe that is known to make it do so.
        if printed_bar != (mode == "upstream"):
            problems.append(f"progress probe {mode} printed {child.stdout!r}")
        expected = all_steps if mode == "callback" else []
        if sorted(result["calls"]) != expected:
            problems.append(f"progress probe {mode} received calls {result['calls']}")
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

    problems = check_progress_routing(library)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print("  progress routing ok")


if __name__ == "__main__":
    main()

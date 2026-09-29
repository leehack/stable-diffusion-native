#!/usr/bin/env python3
"""Loads a built runtime and lists its devices through the C API.

`--expect-device vulkan` fails unless a device name starts with that prefix,
which checks that a GPU backend initialized rather than silently falling back
to the CPU.
"""

from __future__ import annotations

import argparse
import ctypes
import sys

from build import BIN_ROOT, TARGETS


def main() -> None:
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


if __name__ == "__main__":
    main()

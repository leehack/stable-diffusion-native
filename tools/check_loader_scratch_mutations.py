#!/usr/bin/env python3
"""Opt-in local M3/M5 controls; changes only a configured build-tree copy.

Run build.py first. Never edits a submodule or staged bin/ payload. Restores and
rebuilds the positive source even on failure. CPU fixtures only; no downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    work = args.build_dir.resolve()
    source = work / "patched/loader_scratch_vm_allocate/src/model_loader.cpp"
    library = work / "bin/libstable-diffusion.dylib"
    if sys.platform != "darwin" or not source.is_file() or not library.is_file():
        parser.error("requires an existing macOS build and its applied loader carry")
    if source.is_symlink() or not source.resolve().is_relative_to(work):
        parser.error("scratch source must be a regular build-tree copy")
    original = source.read_text()
    changes = {
        "M3": ("TensorScratch zip_entry_buffer;", "std::vector<uint8_t> zip_entry_buffer;"),
        "M5": ("vm_deallocate(mach_task_self(), (vm_address_t)data_, capacity_);", "/* M5: missing deallocation */"),
    }
    for before, _ in changes.values():
        if original.count(before) != 1:
            parser.error("unexpected loader source; re-audit mutation insertion points")
    args.evidence.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "tests"), "SD_REQUIRE_RUNTIME": "1",
           "SD_LOADER_SCRATCH_LIBRARY": str(library)}
    records = []

    def build(label):
        with (args.evidence / f"{label}-build.log").open("w") as log:
            subprocess.run(["cmake", "--build", str(work), "-j", "4"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def probe(label, expected_failure=None):
        result = subprocess.run([sys.executable, "-m", "unittest", "-v",
                                 "test_loader_scratch_runtime.LoaderScratchRuntimeTest."
                                 "test_cpu_load_takes_no_tensor_scratch_from_the_heap"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        (args.evidence / f"{label}-probe.log").write_text(output)
        record = {"variant": label, "runtimeSha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                  "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "returncode": result.returncode}
        records.append(record)
        print(json.dumps(record), flush=True)
        if expected_failure:
            if result.returncode == 0 or expected_failure not in output:
                raise RuntimeError(f"{label} did not fail for its causal assertion")
        elif result.returncode:
            raise RuntimeError(f"{label} positive control failed")

    try:
        probe("positive")
        for label, (before, after) in changes.items():
            source.write_text(original.replace(before, after))
            build(label)
            if after not in source.read_text():
                raise RuntimeError("CMake regenerated the mutated source; rebuild before running this tool")
            probe(label, "a loader thread took tensor scratch from the heap" if label == "M3"
                  else "scratch VM allocations were not freed")
    finally:
        source.write_text(original)
        build("restored")
        (args.evidence / "mutation-records.json").write_text(json.dumps(records, indent=2) + "\n")
    probe("restored")
    (args.evidence / "mutation-records.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

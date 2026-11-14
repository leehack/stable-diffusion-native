"""On Apple, the model loader takes no tensor-sized scratch memory from the heap.

Upstream's loader reads and converts tensors in scratch vectors on its own
threads. libmalloc can keep such a freed block in the process footprint, about
200 MiB of an iPhone app after an SD-Turbo unload (leehack/llamadart#2004), so
`patches/loader_scratch_vm_allocate.patch` takes them from `vm_allocate`.

The probe replaces `operator new` and reports the largest request the library
made off the calling thread while it loaded the 12 MB PixArt model
`exit_teardown_runtime_test make-model` writes and generated once. Without the
patch that is a whole tensor: a convolution weight read for conversion on the
CPU, every tensor on Metal. A copy with BF16 convolution weights reaches the
conversion's own scratch. Nothing is downloaded. GitHub's macOS runners list
no Metal device; run this on a Mac with `SD_REQUIRE_METAL_GENERATION=1`.

    SD_REQUIRE_RUNTIME=1            fail instead of skipping
    SD_REQUIRE_METAL_GENERATION=1   fail when no Metal device generates
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import build  # noqa: E402
from test_fp8_runtime import read_safetensors, write_safetensors, write_weight_copies  # noqa: E402

COMPILER = os.environ.get("CXX") or shutil.which("c++")
MODEL_WRITER = REPO_ROOT / "tests" / "native" / "exit_teardown_runtime_test.cpp"
PROBE = REPO_ROOT / "tests" / "native" / "loader_scratch_runtime_test.cpp"
# The loader's threads ask for a few hundred bytes; the model's smallest
# scratch block without the patch is a 73728-byte convolution weight.
MAX_WORKER_NEW = 16 * 1024
WORKER_NEW = re.compile(r"^WORKER_NEW (\d+)$", re.MULTILINE)


def write_bf16_convolutions(source: Path, copy: Path) -> None:
    """Stores the 4-D weights as BF16, which the loader converts to F16."""
    header, data = read_safetensors(source)
    header.pop("__metadata__", None)
    blobs = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        blobs[name] = data[begin:end]
        if len(info["shape"]) == 4 and info["dtype"] == "F32":
            blob = blobs[name]
            blobs[name] = bytes(b for pair in zip(blob[2::4], blob[3::4]) for b in pair)
            info["dtype"] = "BF16"
    write_safetensors(copy, header, blobs)


@unittest.skipUnless(sys.platform == "darwin", "the carried patch changes Apple builds only")
class LoaderScratchRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skip_reason = ""
        target = build.TARGETS["macos-arm64" if platform.machine() == "arm64" else "macos-x64"]
        library = build.BIN_ROOT / target.name / "lib" / target.library
        if not library.is_file() or COMPILER is None:
            cls.skip_reason = f"no built runtime for {target.name} or no C++ compiler"
            return
        cls.directory = tempfile.TemporaryDirectory()
        work = Path(cls.directory.name)
        binaries = {}
        for source in (MODEL_WRITER, PROBE):
            binary = work / source.stem
            subprocess.run(
                [COMPILER, "-std=c++17", "-O1", "-pthread", f"-I{REPO_ROOT / 'src'}",
                 f"-I{build.HEADER.parent}", str(source), f"-L{library.parent}",
                 "-lstable-diffusion", f"-Wl,-rpath,{library.parent}", "-o", str(binary)],
                check=True)
            binaries[source] = binary
        cls.probe = binaries[PROBE]
        source_model = work / "source.safetensors"
        subprocess.run([str(binaries[MODEL_WRITER]), "make-model", str(source_model)],
                       check=True, timeout=600, capture_output=True)
        cls.models = {"F32": work / "f32.safetensors", "BF16 convolutions": work / "bf16.safetensors"}
        write_weight_copies(source_model, {"F32": cls.models["F32"]})
        write_bf16_convolutions(cls.models["F32"], cls.models["BF16 convolutions"])
        listing = subprocess.run([str(cls.probe), "devices"], check=True, capture_output=True,
                                 text=True, timeout=120).stdout
        cls.devices = [line.split("\t")[0] for line in listing.splitlines() if line.strip()]

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "directory"):
            cls.directory.cleanup()

    def setUp(self) -> None:
        if self.skip_reason:
            if os.environ.get("SD_REQUIRE_RUNTIME"):
                self.fail(self.skip_reason)
            self.skipTest(self.skip_reason)

    def check_device(self, device: str) -> None:
        for name, model in self.models.items():
            with self.subTest(device=device, model=name):
                result = subprocess.run([str(self.probe), "run", str(model), device],
                                        capture_output=True, text=True, timeout=600)
                if result.returncode != 0 and device.startswith("MTL") and \
                        not os.environ.get("SD_REQUIRE_METAL_GENERATION"):
                    self.skipTest(f"{device} cannot generate:\n{result.stderr[-2000:]}")
                self.assertEqual(0, result.returncode, f"{device}:\n{result.stderr[-4000:]}")
                requests = WORKER_NEW.findall(result.stdout)
                self.assertEqual(1, len(requests), result.stdout[-4000:])
                self.assertLess(int(requests[0]), MAX_WORKER_NEW,
                                "a loader thread took tensor scratch from the heap")

    def test_cpu_load_takes_no_tensor_scratch_from_the_heap(self) -> None:
        self.check_device("CPU")

    def test_metal_load_takes_no_tensor_scratch_from_the_heap(self) -> None:
        metal = [device for device in self.devices if device.startswith("MTL")]
        if not metal:
            if os.environ.get("SD_REQUIRE_METAL_GENERATION"):
                self.fail(f"no Metal device in {self.devices}")
            self.skipTest(f"no Metal device in {self.devices}")
        for device in metal:
            self.check_device(device)


if __name__ == "__main__":
    unittest.main()

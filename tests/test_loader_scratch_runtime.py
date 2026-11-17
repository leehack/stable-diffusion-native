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
import struct
import zipfile
import hashlib
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
VM_SCRATCH = re.compile(r"^VM_SCRATCH (\d+) (\d+) (\d+) (\d+)$", re.MULTILINE)


def write_torch_zip(source: Path, copy: Path) -> None:
    """Trusted tensor-only pickle, never unpickled; shared-storage offset path.

    Padding makes entry_size != tensor bytes even for a single stored tensor,
    so ModelLoader::load_tensors must use zip_entry_buffer, not direct reads.
    """
    from test_torch_zip_lookup import integer, string
    header, data = read_safetensors(source)
    tensors = []
    with zipfile.ZipFile(copy, "w", compression=zipfile.ZIP_STORED) as archive:
        def write_entry(name, payload):
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            archive.writestr(entry, payload)
        for index, (name, info) in enumerate(header.items()):
            if name == "__metadata__":
                continue
            assert info["dtype"] == "F32"
            begin, end = info["data_offsets"]
            shape = info["shape"]
            stride, strides = 1, []
            for dim in reversed(shape):
                strides.insert(0, stride)
                stride *= dim
            def tuple_of(values):
                return b"(" + b"".join(integer(v) for v in values) + b"t"
            tensors.append(string(name) + b"ctorch._utils\n_rebuild_tensor_v2\n((" +
                           string("storage") + b"ctorch\nFloatStorage\n" + string(str(index)) +
                           string("cpu") + integer(stride + 8) + b"tQ" + integer(8) +
                           tuple_of(shape) + tuple_of(strides) + b"\x89" +
                           b"ccollections\nOrderedDict\n)RtR")
            write_entry(f"archive/data/{index}", bytes(32) + data[begin:end] + bytes(65536))
        write_entry("archive/data.pkl", b"\x80\x02}(" + b"".join(tensors) + b"u.")


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
        library = Path(os.environ.get("SD_LOADER_SCRATCH_LIBRARY", "")) if os.environ.get(
            "SD_LOADER_SCRATCH_LIBRARY") else build.BIN_ROOT / target.name / "lib" / target.library
        if not library.is_file() or COMPILER is None:
            cls.skip_reason = f"no built runtime for {target.name} or no C++ compiler"
            return
        cls.directory = tempfile.TemporaryDirectory()
        work = Path(cls.directory.name)
        cls.observer = work / "loader_vm_observer.dylib"
        subprocess.run([COMPILER, "-std=c++17", "-O1", "-shared", "-fPIC",
                        str(REPO_ROOT / "tests/native/loader_vm_observer.cpp"),
                        "-o", str(cls.observer)], check=True)
        binaries = {}
        for source in (MODEL_WRITER, PROBE):
            binary = work / source.stem
            subprocess.run(
                [COMPILER, "-std=c++17", "-O1", "-pthread", f"-I{REPO_ROOT / 'src'}",
                 f"-I{build.HEADER.parent}", str(source), f"-L{library.parent}",
                 "-lstable-diffusion", f"-Wl,-rpath,{library.parent}",
                 *([str(cls.observer)] if source == PROBE else []), "-o", str(binary)],
                check=True)
            binaries[source] = binary
        cls.probe = binaries[PROBE]
        source_model = work / "source.safetensors"
        subprocess.run([str(binaries[MODEL_WRITER]), "make-model", str(source_model)],
                       check=True, timeout=600, capture_output=True)
        cls.models = {"F32": work / "f32.safetensors", "BF16 convolutions": work / "bf16.safetensors"}
        write_weight_copies(source_model, {"F32": cls.models["F32"]})
        write_bf16_convolutions(cls.models["F32"], cls.models["BF16 convolutions"])
        cls.models["ZIP shared-storage offset"] = work / "offset.pt"
        write_torch_zip(cls.models["F32"], cls.models["ZIP shared-storage offset"])
        for name, fixture in cls.models.items():
            print(f"FIXTURE {name}: sha256={hashlib.sha256(fixture.read_bytes()).hexdigest()} "
                  f"bytes={fixture.stat().st_size}", flush=True)
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
        reference = None
        for name, model in self.models.items():
            with self.subTest(device=device, model=name):
                output = Path(self.directory.name) / "image.rgb"
                output.unlink(missing_ok=True)
                result = subprocess.run([str(self.probe), "run", str(model), device, str(output)],
                                        capture_output=True, text=True, timeout=600,
                                        env={**os.environ, "DYLD_INSERT_LIBRARIES": str(self.observer)})
                if result.returncode != 0 and device.startswith("MTL") and \
                        not os.environ.get("SD_REQUIRE_METAL_GENERATION"):
                    self.skipTest(f"{device} cannot generate:\n{result.stderr[-2000:]}")
                self.assertEqual(0, result.returncode, f"{device}:\n{result.stderr[-4000:]}")
                image = output.read_bytes()
                if name == "F32":
                    reference = image
                    self.assertGreaterEqual(len(set(image)), 16, "flat output hides corrupted tensor reads")
                elif name == "ZIP shared-storage offset":
                    self.assertEqual(reference, image, "ZIP tensor offsets changed the F32 image")
                requests = WORKER_NEW.findall(result.stdout)
                self.assertEqual(1, len(requests), result.stdout[-4000:])
                self.assertLess(int(requests[0]), MAX_WORKER_NEW,
                                "a loader thread took tensor scratch from the heap")
                observations = VM_SCRATCH.findall(result.stdout)
                self.assertEqual(1, len(observations), result.stdout[-4000:])
                allocated, released, unmapped, live = map(int, observations[0])
                self.assertGreater(allocated, 0, "no actual worker scratch allocation observed")
                self.assertEqual(allocated, released, "scratch VM allocations were not freed")
                self.assertEqual(released, unmapped, "released scratch mapping remained readable")
                self.assertEqual(0, live, "scratch VM allocations remain live")

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

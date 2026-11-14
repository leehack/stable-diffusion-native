"""The graph split cuts upstream's graph evaluation and leaves the image alone.

The model is the 12 MB PixArt model `exit_teardown_runtime_test make-model`
writes, with the fixed weight patterns of test_fp8_runtime.py so that its image
is not flat; nothing is downloaded. One process generates at several thresholds,
and the library's cut count tells whether the split ran: the default of
llamadart#1047's iPhone split is 0 everywhere else, so a host build cuts
nothing until a threshold is set.

    SD_GRAPH_SPLIT_TARGET=<target>   a built Linux target; macOS defaults to
                                     the host's
    SD_REQUIRE_RUNTIME=1             fail instead of skipping
    SD_REQUIRE_METAL_GENERATION=1    fail when no Metal device generates
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
from test_fp8_runtime import write_weight_copies  # noqa: E402

COMPILER = os.environ.get("CXX") or shutil.which("c++")
MODEL_WRITER = REPO_ROOT / "tests" / "native" / "exit_teardown_runtime_test.cpp"
PROBE = REPO_ROOT / "tests" / "native" / "graph_split_runtime_test.cpp"
# Small enough that the tiny model's matrix multiplications cross it, and far
# above anything it computes.
LOW = 4096
HIGH = 1 << 60
# Mean absolute difference per channel byte that a GPU's changed rounding may
# cause; the CPU must match exactly.
MAX_GPU_MEAN_DIFFERENCE = 0.5
RUN = re.compile(r"^RUN (\d+) (\d+) (\d+) (\d+)$", re.MULTILINE)


def mean_difference(first: bytes, second: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


class GraphSplitRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skip_reason = ""
        target_name = os.environ.get("SD_GRAPH_SPLIT_TARGET", "")
        if not target_name and sys.platform == "darwin":
            target_name = "macos-arm64" if platform.machine() == "arm64" else "macos-x64"
        if not target_name or sys.platform == "win32":
            cls.skip_reason = "set SD_GRAPH_SPLIT_TARGET to a built Linux target"
            return
        target = build.TARGETS[target_name]
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
        cls.model = work / "model.safetensors"
        write_weight_copies(source_model, {"F32": cls.model})
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

    def generate(self, device: str, *thresholds: int) -> tuple[list[tuple[int, int]], list[bytes], str]:
        """Returns (getter, cuts) and the image per threshold, and the output."""
        prefix = Path(self.directory.name) / "image"
        result = subprocess.run(
            [str(self.probe), "run", str(self.model), device, str(prefix),
             *map(str, thresholds)],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0 and device.startswith("MTL") and \
                not os.environ.get("SD_REQUIRE_METAL_GENERATION"):
            self.skipTest(f"{device} cannot generate:\n{result.stderr[-2000:]}")
        self.assertEqual(0, result.returncode, f"{device}:\n{result.stderr[-4000:]}")
        runs = [(int(getter), int(cuts)) for _, _, getter, cuts in RUN.findall(result.stdout)]
        self.assertEqual(len(thresholds), len(runs), result.stdout[-4000:])
        images = [(prefix.parent / f"image.{i}.rgb").read_bytes() for i in range(len(thresholds))]
        return runs, images, result.stdout

    def check_device(self, device: str) -> None:
        runs, images, output = self.generate(device, 0, LOW, 0, LOW, HIGH)
        self.assertIn("DEFAULT 0\n", output, "a host build splits by default")
        self.assertEqual([0, LOW, 0, LOW, HIGH], [getter for getter, _ in runs])
        cuts = [count for _, count in runs]
        self.assertEqual(0, cuts[0], "a threshold of 0 cut the graph")
        self.assertGreater(cuts[1], cuts[0], "cuts increase with a low threshold")
        self.assertEqual(cuts[1], cuts[2], "a threshold of 0 cut after a split")
        self.assertGreater(cuts[3], cuts[2], "cuts increase when the threshold returns")
        self.assertEqual(cuts[3], cuts[4], "a threshold above every node cut the graph")
        self.assertEqual(cuts[1] - cuts[0], cuts[3] - cuts[2], "the same graph cuts differently")
        self.assertGreaterEqual(len(set(images[0])), 16, "the image is flat")
        for image in images[1:]:
            self.assertEqual(len(images[0]), len(image))
        for i, image in enumerate(images[1:], 1):
            if device == "CPU":
                self.assertEqual(images[0], image, f"run {i} changed the CPU image")
            else:
                self.assertLessEqual(mean_difference(images[0], image), MAX_GPU_MEAN_DIFFERENCE,
                                     f"run {i} changed the image on {device}")

    def test_split_cuts_and_keeps_the_image_on_the_cpu(self) -> None:
        self.check_device("CPU")

    def test_split_cuts_and_keeps_the_image_on_each_gpu(self) -> None:
        # BLAS is a matrix accelerator beside the CPU, not a device a graph runs on.
        gpus = [device for device in self.devices if device not in ("CPU", "BLAS")]
        if sys.platform == "darwin" and not any(d.startswith("MTL") for d in gpus) and \
                os.environ.get("SD_REQUIRE_METAL_GENERATION"):
            self.fail(f"no Metal device in {self.devices}")
        if not gpus:
            self.skipTest(f"no GPU device in {self.devices}")
        for device in gpus:
            with self.subTest(device=device):
                self.check_device(device)


if __name__ == "__main__":
    unittest.main()

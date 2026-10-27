"""Generates from FP8 E4M3 weights on every device of a built runtime.

The model is the 12 MB PixArt model `exit_teardown_runtime_test make-model`
writes, with its weights replaced by fixed patterns. Its 2-D weights hold
values E4M3 represents exactly, kept as F32 in a control copy and stored as
F8_E4M3 in a second copy, so on each device the two images differ only by
matrix precision. Nothing is downloaded.

On Metal `patches/fp8_metal_f16_load.patch` loads FP8 weights as F16; without
it the FP8 generation aborts the process (#29). GitHub's macOS runners list
no Metal device; run this on a Mac with `SD_REQUIRE_METAL_GENERATION=1`.

    SD_FP8_TARGET=<target>          a built Linux target; macOS defaults to
                                    the host's
    SD_REQUIRE_RUNTIME=1            fail instead of skipping
    SD_REQUIRE_METAL_GENERATION=1   fail when no Metal device generates the
                                    control
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build  # noqa: E402

COMPILER = os.environ.get("CXX") or shutil.which("c++")
MODEL_WRITER = REPO_ROOT / "tests" / "native" / "exit_teardown_runtime_test.cpp"
PROBE = REPO_ROOT / "tests" / "native" / "fp8_runtime_test.cpp"
GGML_METAL = REPO_ROOT / "third_party" / "stable-diffusion.cpp" / "ggml" / "src" / "ggml-metal"
# Mean absolute difference per channel byte between the FP8 and F32 images of
# one device. An M4 Max measured 0.03 on the CPU and 0 on Metal.
MAX_MEAN_DIFFERENCE = 0.5
# Fewer distinct bytes would mean a flat image that hides any difference.
MIN_DISTINCT_BYTES = 16
# 16 E4M3 values, +-2^-6 * (1 + m/8): sign, exponent field 1, mantissa m.
E4M3_PATTERN = bytes(sign | 0x08 | m for sign in (0, 0x80) for m in range(8))
# Every other tensor: values in [-0.125, 0.125). Larger ones make the tiny
# network chaotic, so that matrix precision alone changes the image.
OTHER_PATTERN = struct.pack("<61f", *(((i * 37) % 61 - 30.5) / 61 * 0.25 for i in range(61)))


def e4m3_value(byte: int) -> float:
    value = 2.0 ** -6 * (1 + (byte & 0x07) / 8)
    return -value if byte & 0x80 else value


def read_safetensors(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8:8 + size]), data[8 + size:]


def write_safetensors(path: Path, header: dict, blobs: dict[str, bytes]) -> None:
    offset = 0
    for name, info in header.items():
        info["data_offsets"] = [offset, offset + len(blobs[name])]
        offset += len(blobs[name])
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(text)))
        file.write(text)
        for name in header:
            file.write(blobs[name])


def write_fp8_pair(source: Path, control: Path, fp8: Path) -> int:
    """Writes the F32 control and FP8 copies of `source`; returns the FP8 count."""
    header, _ = read_safetensors(source)
    header.pop("__metadata__", None)
    control_header, fp8_header, control_blobs, fp8_blobs = {}, {}, {}, {}
    pattern_f32 = struct.pack(f"<{len(E4M3_PATTERN)}f", *map(e4m3_value, E4M3_PATTERN))
    converted = 0
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise ValueError(f"{name} is {info['dtype']}, not F32")
        begin, end = info["data_offsets"]
        count = (end - begin) // 4
        control_header[name] = dict(info)
        fp8_header[name] = dict(info)
        if len(info["shape"]) == 2 and name.endswith(".weight"):
            repeats = -(-count // len(E4M3_PATTERN))
            control_blobs[name] = (pattern_f32 * repeats)[:4 * count]
            fp8_blobs[name] = (E4M3_PATTERN * repeats)[:count]
            fp8_header[name]["dtype"] = "F8_E4M3"
            converted += 1
        else:
            repeats = -(-count // 61)
            control_blobs[name] = fp8_blobs[name] = (OTHER_PATTERN * repeats)[:4 * count]
    write_safetensors(control, control_header, control_blobs)
    write_safetensors(fp8, fp8_header, fp8_blobs)
    return converted


def mean_difference(first: bytes, second: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


class Fp8RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skip_reason = ""
        target_name = os.environ.get("SD_FP8_TARGET", "")
        if not target_name and sys.platform == "darwin":
            target_name = "macos-arm64" if platform.machine() == "arm64" else "macos-x64"
        if not target_name or sys.platform == "win32":
            cls.skip_reason = "set SD_FP8_TARGET to a built Linux target"
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
        source_model = work / "model.safetensors"
        subprocess.run([str(binaries[MODEL_WRITER]), "make-model", str(source_model)],
                       check=True, timeout=600, capture_output=True)
        cls.control = work / "control.safetensors"
        cls.fp8 = work / "fp8.safetensors"
        cls.converted = write_fp8_pair(source_model, cls.control, cls.fp8)
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

    def generate(self, model: Path, device: str) -> tuple[subprocess.CompletedProcess, bytes]:
        output = Path(self.directory.name) / f"{model.stem}-{device}.rgb"
        output.unlink(missing_ok=True)
        result = subprocess.run([str(self.probe), "generate", str(model), device, str(output)],
                                capture_output=True, text=True, timeout=600)
        return result, output.read_bytes() if output.is_file() else b""

    def check_device(self, device: str) -> None:
        control, control_image = self.generate(self.control, device)
        if control.returncode != 0:
            if device.startswith("MTL") and os.environ.get("SD_REQUIRE_METAL_GENERATION"):
                self.fail(f"the F32 control does not generate on {device}:\n{control.stderr}")
            self.skipTest(f"{device} cannot generate the F32 control:\n{control.stderr[-2000:]}")
        fp8, fp8_image = self.generate(self.fp8, device)
        self.assertEqual(0, fp8.returncode, f"FP8 weights on {device}:\n{fp8.stderr[-4000:]}")
        self.assertGreaterEqual(len(set(control_image)), MIN_DISTINCT_BYTES)
        self.assertEqual(len(control_image), len(fp8_image))
        difference = mean_difference(control_image, fp8_image)
        self.assertLessEqual(difference, MAX_MEAN_DIFFERENCE,
                             f"the FP8 image on {device} is not the F32 control's")

    def test_model_carries_fp8_weights(self) -> None:
        self.assertGreater(self.converted, 0)
        header, _ = read_safetensors(self.fp8)
        self.assertIn("F8_E4M3", {info.get("dtype") for info in header.values()})

    def test_fp8_generates_on_the_cpu(self) -> None:
        self.check_device("CPU")

    def test_fp8_generates_on_each_gpu(self) -> None:
        gpus = [device for device in self.devices if device != "CPU"]
        if sys.platform == "darwin" and not any(d.startswith("MTL") for d in gpus) and \
                os.environ.get("SD_REQUIRE_METAL_GENERATION"):
            self.fail(f"no Metal device in {self.devices}")
        if not gpus:
            self.skipTest(f"no GPU device in {self.devices}")
        for device in gpus:
            with self.subTest(device=device):
                self.check_device(device)


class PinnedGgmlMetalTest(unittest.TestCase):
    # The patch can go once ggml-metal multiplies FP8 matrices or stops
    # claiming that it does. Checked on the sources, so it runs without a GPU.
    def test_pinned_ggml_metal_still_needs_the_patch(self) -> None:
        metal = GGML_METAL
        if not (metal / "ggml-metal-device.m").is_file():
            if os.environ.get("CI"):
                self.fail("third_party/stable-diffusion.cpp is not checked out")
            self.skipTest("third_party/stable-diffusion.cpp is not checked out")
        supports = (metal / "ggml-metal-device.m").read_text()
        start = supports.index("static bool ggml_metal_supports_mul_mat_op(")
        claim = supports[start:supports.index("\n}\n", start)]
        kernels = "".join(path.read_text() for path in (metal / "kernels").glob("*.metal"))
        self.assertNotIn("F8_E", claim, "ggml-metal now excludes FP8 from MUL_MAT; "
                         "re-evaluate patches/fp8_metal_f16_load.patch")
        self.assertNotRegex(kernels, r"kernel_mul_m[mv]\w*_f8",
                            "ggml-metal now has an FP8 matrix kernel; "
                            "re-evaluate patches/fp8_metal_f16_load.patch")


if __name__ == "__main__":
    unittest.main()

"""Generates from FP8 weights, and with LoRAs, on every device of a built runtime.

The model is the 12 MB PixArt model `exit_teardown_runtime_test make-model`
writes, with its weights replaced by fixed patterns. Its 2-D weights hold
values that E4M3, E5M2, F16 and BF16 represent exactly, kept as F32 in a control
copy and stored in each of those types in further copies, so on each device
the images differ only by matrix precision. Nothing is downloaded.

On Metal `patches/fp8_metal_f16_load.patch` loads FP8 weights as F16; without
it an FP8 generation aborts the process (#29). The LoRA tests apply small
linear and convolution adapters in every apply mode to each weight type.
Without the patches an FP8 LoRA tensor crashes the CPU in every mode, and the
immediately mode, which auto selects for these weights, crashes the CPU for
FP8 weights and aborts Metal, whose F16 convolution weights it cannot add
into (#31). GitHub's macOS runners list no Metal device; run this on a Mac
with `SD_REQUIRE_METAL_GENERATION=1`.

    SD_FP8_TARGET=<target>          a built Linux target; macOS defaults to
                                    the host's
    SD_REQUIRE_RUNTIME=1            fail instead of skipping
    SD_REQUIRE_METAL_GENERATION=1   fail when no Metal device generates the
                                    control
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
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
# Mean absolute difference per channel byte between two images of one device
# that should match. An M4 Max measured at most 0.11 on the CPU and 0 on Metal.
MAX_MEAN_DIFFERENCE = 0.5
# Fewer distinct bytes would mean a flat image that hides any difference.
MIN_DISTINCT_BYTES = 16
FP8_TYPES = {"F8_E4M3": (7, 3), "F8_E5M2": (15, 2)}  # exponent bias, mantissa bits
# 2-D weights: +-2^-6 * (1 + m/4), which every type here holds exactly.
WEIGHT_TYPES = ("F32", "F16", "BF16", *FP8_TYPES)
WEIGHT_VALUES = [sign * 2.0 ** -6 * (1 + m / 4) for sign in (1, -1) for m in range(4)]
# Every other tensor: values in [-0.125, 0.125). Larger ones make the tiny
# network chaotic, so that matrix precision alone changes the image.
OTHER_PATTERN = struct.pack("<61f", *(((i * 37) % 61 - 30.5) / 61 * 0.25 for i in range(61)))
# Upstream loads a convolution weight as F16 whatever the file holds.
LINEAR_TARGET = "model.diffusion_model.final_layer.linear"
CONV_TARGET = "model.diffusion_model.x_embedder.proj"
# Applied immediately, these merge on the text encoder's and the TAESD's own
# backend. Upstream applies no LoRA to the TAESD at runtime.
TE_TARGET = "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q"
TAE_TARGET = "tae.decoder.layers.11"
LORAS = {"linear": (LINEAR_TARGET, "F16"), "linear-fp8": (LINEAR_TARGET, "F8_E4M3"),
         "conv": (CONV_TARGET, "F16"), "te": (TE_TARGET, "F16"), "tae": (TAE_TARGET, "F16")}
LORA_RANK = 4
LORA_DOWN_VALUES = [((i * 5) % 9 - 4) / 32 for i in range(9)]
LORA_UP_VALUES = [((i * 7) % 11 - 5) / 8 for i in range(11)]
# Auto-fit prints whole MiB, too coarse for the model's own weights. The model
# ignores this extra DiT tensor, and auto-fit counts it: 16 MiB as F16.
AUTO_FIT_PAD = ("model.diffusion_model.auto_fit_pad.weight", [8192, 1024])


def encode(values: list[float], dtype: str, count: int) -> bytes:
    """Repeats `values` to `count` elements of `dtype`, which must hold them exactly."""
    if dtype in ("F32", "F16"):
        code = "f" if dtype == "F32" else "e"
        return struct.pack(f"<{count}{code}", *(values[i % len(values)] for i in range(count)))
    if dtype == "BF16":
        f32 = encode(values, "F32", len(values))
        if any(f32[i:i + 2] != b"\0\0" for i in range(0, len(f32), 4)):
            raise ValueError(f"BF16 does not hold every value of {values}")
        encoded = b"".join(f32[i + 2:i + 4] for i in range(0, len(f32), 4))
        return (encoded * -(-count // len(values)))[:2 * count]
    bias, bits = FP8_TYPES[dtype]
    encoded = []
    for value in values:
        if value == 0:
            encoded.append(0)
            continue
        fraction, exponent = math.frexp(abs(value))
        field = exponent - 1 + bias
        mantissa = (fraction * 2 - 1) * (1 << bits)
        if field < 1 or mantissa != int(mantissa):
            raise ValueError(f"{dtype} does not hold {value} exactly")
        encoded.append((0x80 if value < 0 else 0) | field << bits | int(mantissa))
    return bytes(encoded[i % len(encoded)] for i in range(count))


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


def write_weight_copies(source: Path, copies: dict[str, Path]) -> int:
    """Writes a copy of `source` per 2-D weight type; returns the 2-D weight count."""
    header, _ = read_safetensors(source)
    header.pop("__metadata__", None)
    outputs = {dtype: ({}, {}) for dtype in copies}
    converted = 0
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise ValueError(f"{name} is {info['dtype']}, not F32")
        begin, end = info["data_offsets"]
        count = (end - begin) // 4
        weight = len(info["shape"]) == 2 and name.endswith(".weight")
        converted += int(weight)
        for dtype, (copy_header, blobs) in outputs.items():
            copy_header[name] = dict(info)
            if weight:
                copy_header[name]["dtype"] = dtype
                blobs[name] = encode(WEIGHT_VALUES, dtype, count)
            else:
                blobs[name] = (OTHER_PATTERN * -(-count // 61))[:4 * count]
    for dtype, path in copies.items():
        write_safetensors(path, *outputs[dtype])
    return converted


def write_lora(path: Path, target: str, weight_shape: list[int], down_dtype: str) -> None:
    """Writes a LoRA for the linear or convolution weight `target`."""
    out_features, *in_shape = weight_shape
    down = f"{target}.lora_down.weight"
    up = f"{target}.lora_up.weight"
    up_shape = [out_features, LORA_RANK] + [1] * (len(in_shape) - 1)
    header = {down: {"dtype": down_dtype, "shape": [LORA_RANK, *in_shape]},
              up: {"dtype": "F16", "shape": up_shape}}
    write_safetensors(path, header, {
        down: encode(LORA_DOWN_VALUES, down_dtype, LORA_RANK * math.prod(in_shape)),
        up: encode(LORA_UP_VALUES, "F16", out_features * LORA_RANK)})


def write_padded(source: Path, padded: Path) -> None:
    header, data = read_safetensors(source)
    blobs = {name: data[info["data_offsets"][0]:info["data_offsets"][1]] for name, info in header.items()}
    name, shape = AUTO_FIT_PAD
    dtype = next(info["dtype"] for info in header.values() if len(info["shape"]) == 2)
    header[name] = {"dtype": dtype, "shape": shape}
    blobs[name] = encode([0.0], dtype, shape[0] * shape[1])
    write_safetensors(padded, header, blobs)


def mean_difference(first: bytes, second: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


class RuntimeTestCase(unittest.TestCase):
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
        cls.models = {dtype: work / f"{dtype.lower()}.safetensors" for dtype in WEIGHT_TYPES}
        cls.converted = write_weight_copies(source_model, cls.models)
        header, _ = read_safetensors(cls.models["F32"])
        cls.loras = {name: work / f"lora-{name}.safetensors" for name in LORAS}
        for name, (target, dtype) in LORAS.items():
            write_lora(cls.loras[name], target, header[f"{target}.weight"]["shape"], dtype)
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

    def generate(self, dtype: str, device: str, lora: str | None = None, mode: str = "",
                 rules: str = "", model: Path | None = None) -> tuple[subprocess.CompletedProcess, bytes]:
        """Generates one image; the result's stdout holds the runtime's log."""
        output = Path(self.directory.name) / "image.rgb"
        output.unlink(missing_ok=True)
        model = model or self.models[dtype]
        command = [str(self.probe), "generate", str(model), device, str(output)]
        if lora is not None:
            command += [str(self.loras[lora]), mode] + ([rules] if rules else [])
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        return result, output.read_bytes() if output.is_file() else b""

    def generate_control(self, device: str) -> bytes:
        control, image = self.generate("F32", device)
        if control.returncode != 0:
            if device.startswith("MTL") and os.environ.get("SD_REQUIRE_METAL_GENERATION"):
                self.fail(f"the F32 control does not generate on {device}:\n{control.stderr}")
            self.skipTest(f"{device} cannot generate the F32 control:\n{control.stderr[-2000:]}")
        self.assertGreaterEqual(len(set(image)), MIN_DISTINCT_BYTES)
        return image

    def assert_matches(self, expected: bytes, actual: bytes, what: str) -> None:
        self.assertEqual(len(expected), len(actual), what)
        self.assertLessEqual(mean_difference(expected, actual), MAX_MEAN_DIFFERENCE, what)

    def gpus(self) -> list[str]:
        gpus = [device for device in self.devices if device != "CPU"]
        if sys.platform == "darwin" and not any(d.startswith("MTL") for d in gpus) and \
                os.environ.get("SD_REQUIRE_METAL_GENERATION"):
            self.fail(f"no Metal device in {self.devices}")
        if not gpus:
            self.skipTest(f"no GPU device in {self.devices}")
        return gpus


class Fp8RuntimeTest(RuntimeTestCase):
    def check_device(self, device: str) -> None:
        control = self.generate_control(device)
        for dtype in FP8_TYPES:
            with self.subTest(dtype=dtype):
                fp8, image = self.generate(dtype, device)
                self.assertEqual(0, fp8.returncode, f"{dtype} weights on {device}:\n{fp8.stderr[-4000:]}")
                self.assert_matches(control, image, f"the {dtype} image on {device} is not the F32 control's")

    def test_model_carries_fp8_weights(self) -> None:
        self.assertGreater(self.converted, 0)
        for dtype in FP8_TYPES:
            header, _ = read_safetensors(self.models[dtype])
            self.assertIn(dtype, {info.get("dtype") for info in header.values()})

    def test_fp8_generates_on_the_cpu(self) -> None:
        self.check_device("CPU")

    def test_fp8_generates_on_each_gpu(self) -> None:
        for device in self.gpus():
            with self.subTest(device=device):
                self.check_device(device)

    def test_auto_fit_plans_fp8_as_f16_on_metal(self) -> None:
        metal = [device for device in self.gpus() if device.startswith("MTL")]
        if not metal:
            self.skipTest(f"no Metal device in {self.devices}")
        planned = {}
        for dtype in ("F16", *FP8_TYPES):
            padded = Path(self.directory.name) / f"padded-{dtype.lower()}.safetensors"
            write_padded(self.models[dtype], padded)
            result = subprocess.run([str(self.probe), "plan", str(padded), metal[0]],
                                    capture_output=True, text=True, timeout=600)
            self.assertEqual(0, result.returncode, f"{dtype} on {metal[0]}:\n{result.stderr[-4000:]}")
            match = re.search(r"DiT +params +(\d+) MiB", result.stdout)
            self.assertIsNotNone(match, f"no auto-fit plan for {dtype}:\n{result.stdout[-4000:]}")
            planned[dtype] = int(match.group(1))
        self.assertGreaterEqual(planned["F16"], 16)
        self.assertEqual({dtype: planned["F16"] for dtype in planned}, planned,
                         "auto-fit plans FP8 weights below the F16 size Metal loads them at")


FALLBACK_WARNING = "do not support the immediately LoRA apply mode; using at_runtime instead"
APPLIED = re.compile(r"\((\d+) / (\d+)\) LoRA tensors have been applied")


class LoraRuntimeTest(RuntimeTestCase):
    def generate_with_lora(self, what: str, *args, **kwargs) -> tuple[str, bytes]:
        result, image = self.generate(*args, **kwargs)
        self.assertEqual(0, result.returncode, f"{what}:\n{result.stderr[-4000:]}")
        self.assertTrue(any(int(applied) > 0 and applied == total
                            for applied, total in APPLIED.findall(result.stdout)),
                        f"{what} applies no LoRA tensor:\n{result.stdout[-4000:]}")
        return result.stdout, image

    def check_device(self, device: str) -> None:
        self.generate_control(device)
        for weights in WEIGHT_TYPES:
            plain = self.generate(weights, device)[1]
            for lora in ("linear", "linear-fp8", "conv"):
                what = f"{weights} weights, {lora} LoRA on {device}"
                _, expected = self.generate_with_lora(f"{what} at_runtime", weights, device, lora, "at_runtime")
                self.assertNotEqual(plain, expected, f"{what} does not change the image")
                for mode in ("immediately", "auto"):
                    with self.subTest(weights=weights, lora=lora, mode=mode):
                        log, image = self.generate_with_lora(f"{what} {mode}", weights, device, lora, mode)
                        self.assert_matches(expected, image, f"{what} {mode} is not the at_runtime image")
                        # Metal cannot add into the F16 weights every model has.
                        falls_back = mode == "immediately" and (weights in FP8_TYPES or device.startswith("MTL"))
                        if device == "CPU" or device.startswith("MTL"):
                            self.assertEqual(falls_back, FALLBACK_WARNING in log, f"{what} {mode}:\n{log[-4000:]}")

    def test_lora_applies_on_the_cpu(self) -> None:
        self.check_device("CPU")

    def test_lora_applies_on_each_gpu(self) -> None:
        for device in self.gpus():
            with self.subTest(device=device):
                self.check_device(device)

    def test_lora_falls_back_for_a_module_on_metal(self) -> None:
        metal = [device for device in self.gpus() if device.startswith("MTL")]
        if not metal:
            self.skipTest(f"no Metal device in {self.devices}")
        for module, lora in (("te", "te"), ("vae", "tae")):
            with self.subTest(module=module):
                backend = f"CPU,{module}={metal[0]}"
                result, _ = self.generate("F16", backend, lora, "immediately")
                self.assertEqual(0, result.returncode, f"{lora} LoRA on {backend}:\n{result.stderr[-4000:]}")
                self.assertIn(FALLBACK_WARNING, result.stdout)

    def test_lora_merges_fp8_weights_a_type_rule_loads_as_f16(self) -> None:
        # Upstream converts only some weights; FP8 elsewhere would keep the fallback.
        fp8_header, fp8_data = read_safetensors(self.models["F8_E4M3"])
        control_header, control_data = read_safetensors(self.models["F32"])
        header, blobs = {}, {}
        for name, info in fp8_header.items():
            source, data = (info, fp8_data) if ".blocks." in name else (control_header[name], control_data)
            header[name] = dict(source)
            blobs[name] = data[source["data_offsets"][0]:source["data_offsets"][1]]
        model = Path(self.directory.name) / "fp8-blocks.safetensors"
        write_safetensors(model, header, blobs)
        self.assertIn("F8_E4M3", {info["dtype"] for info in header.values()})
        _, expected = self.generate_with_lora("at_runtime", "", "CPU", "linear", "at_runtime",
                                              rules="blocks=f16", model=model)
        log, image = self.generate_with_lora("immediately", "", "CPU", "linear", "immediately",
                                             rules="blocks=f16", model=model)
        self.assertNotIn(FALLBACK_WARNING, log)
        self.assert_matches(expected, image, "the merged image is not the at_runtime image")


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
        self.assertNotRegex(claim, r"(?i)f8_e|fp8", "ggml-metal now excludes FP8 from MUL_MAT; "
                            "re-evaluate patches/fp8_metal_f16_load.patch")
        self.assertNotRegex(kernels, r"(?i)kernel_mul_m[mv]\w*_fp?8",
                            "ggml-metal now has an FP8 matrix kernel; "
                            "re-evaluate patches/fp8_metal_f16_load.patch")


if __name__ == "__main__":
    unittest.main()

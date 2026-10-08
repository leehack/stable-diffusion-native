"""Builds and runs `tests/native/device_memory_test.cpp`.

The test links `src/sd_dart_device.cpp` against a stand-in for ggml's device
registry, so it needs the submodules' headers but no built runtime and no GPU.
`tools/smoke_test.py` checks the same exports against each built runtime.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "third_party" / "stable-diffusion.cpp"
SOURCES = [
    REPO_ROOT / "src" / "sd_dart_device.cpp",
    REPO_ROOT / "tests" / "native" / "device_memory_test.cpp",
]
INCLUDE_DIRS = [REPO_ROOT / "src", UPSTREAM / "include", UPSTREAM / "ggml" / "include"]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
# The build without a GPU backend, and one with.
MODES = {"no-backend": 0, "backend": 1}
ADDRESS_SANITIZER = "-fsanitize=address"


@unittest.skipIf(sys.platform == "win32", "uses a GCC or Clang command line")
class DeviceMemoryTest(unittest.TestCase):
    def require(self, condition: bool, reason: str) -> None:
        """Skips locally, but fails in CI, where a skip would hide the test."""
        if condition:
            return
        if os.environ.get("CI"):
            self.fail(reason)
        self.skipTest(reason)

    def setUp(self) -> None:
        self.require(COMPILER is not None, "no C++ compiler; set CXX")
        self.require((INCLUDE_DIRS[2] / "ggml-backend.h").is_file(),
                     "third_party/stable-diffusion.cpp/ggml is not checked out; run "
                     "`git submodule update --init --recursive`")

    def build_and_run(self, *flags: str) -> None:
        includes = [f"-I{directory}" for directory in INCLUDE_DIRS]
        with tempfile.TemporaryDirectory() as directory:
            for mode, backend in MODES.items():
                with self.subTest(mode=mode):
                    binary = Path(directory) / f"device_memory_test_{mode}"
                    subprocess.run(
                        [COMPILER, "-std=c++17", "-O1", "-g", *flags,
                         f"-DSD_DART_GPU_BACKEND={backend}", *includes, *map(str, SOURCES),
                         "-o", str(binary)], check=True)
                    result = subprocess.run([str(binary), mode], capture_output=True, text=True,
                                            timeout=600)
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def supports(self, flag: str) -> bool:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.cpp"
            source.write_text("int main() { return 0; }\n")
            binary = Path(directory) / "empty"
            built = subprocess.run([COMPILER, flag, str(source), "-o", str(binary)],
                                   capture_output=True)
            return built.returncode == 0 and subprocess.run(
                [str(binary)], capture_output=True).returncode == 0

    def test_reports_status_and_memory_of_gpu_devices(self) -> None:
        self.build_and_run()

    def test_copies_within_the_caller_s_struct(self) -> None:
        self.require(self.supports(ADDRESS_SANITIZER),
                     f"{COMPILER} cannot build or run {ADDRESS_SANITIZER} binaries")
        self.build_and_run(ADDRESS_SANITIZER)

if __name__ == "__main__":
    unittest.main()

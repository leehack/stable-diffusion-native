"""Builds and runs `tests/native/pickle_span_test.cpp`.

The test compiles upstream's `pickle_io.cpp` from the pinned submodule, so it
needs the submodules' sources but no built runtime and no model. Spans above
INT_MAX are reserved read-only and cost no memory.
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
    UPSTREAM / "src" / "model_io" / "pickle_io.cpp",
    REPO_ROOT / "tests" / "native" / "pickle_span_test.cpp",
]
INCLUDE_DIRS = [UPSTREAM / "src", UPSTREAM / "include", UPSTREAM / "ggml" / "include"]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
ADDRESS_SANITIZER = "-fsanitize=address"


@unittest.skipIf(sys.platform == "win32", "uses a GCC or Clang command line and mmap")
class PickleSpanTest(unittest.TestCase):
    def require(self, condition: bool, reason: str) -> None:
        """Skips locally, but fails in CI, where a skip would hide the test."""
        if condition:
            return
        if os.environ.get("CI"):
            self.fail(reason)
        self.skipTest(reason)

    def setUp(self) -> None:
        self.require(COMPILER is not None, "no C++ compiler; set CXX")
        self.require((INCLUDE_DIRS[2] / "ggml.h").is_file() and SOURCES[0].is_file(),
                     "third_party/stable-diffusion.cpp is not checked out; run "
                     "`git submodule update --init --recursive`")

    def run_mode(self, mode: str, *flags: str) -> subprocess.CompletedProcess:
        includes = [f"-I{directory}" for directory in INCLUDE_DIRS]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "pickle_span_test"
            subprocess.run([COMPILER, "-std=c++17", "-O1", "-g", *flags, *includes,
                            *map(str, SOURCES), "-o", str(binary)], check=True)
            return subprocess.run([str(binary), mode], capture_output=True, text=True,
                                  timeout=600)

    def assert_passes(self, mode: str, *flags: str) -> None:
        result = self.run_mode(mode, *flags)
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

    def test_skips_valid_and_rejects_malformed_bounded_pickles(self) -> None:
        self.assert_passes("bounded")

    def test_bounded_pickles_stay_in_bounds_under_address_sanitizer(self) -> None:
        self.require(self.supports(ADDRESS_SANITIZER),
                     f"{COMPILER} cannot build or run {ADDRESS_SANITIZER} binaries")
        self.assert_passes("bounded", ADDRESS_SANITIZER)

    def test_rejects_malformed_pickles_in_spans_above_int_max(self) -> None:
        self.assert_passes("large")

    # Upstream narrows the remaining span to int before scanning for a newline
    # (#23), so the pinned scanner rejects these. A submodule bump that fixes it
    # makes this an unexpected success, which fails the run: remove the marker.
    @unittest.expectedFailure
    def test_skips_valid_pickles_in_spans_above_int_max(self) -> None:
        self.assert_passes("large-ok")


if __name__ == "__main__":
    unittest.main()

"""Builds and runs `tests/native/progress_callback_test.cpp`.

The test links `src/sd_dart_wrapper.cpp` against a stand-in for upstream, so it
needs the submodule's header but no built runtime.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    REPO_ROOT / "src" / "sd_dart_wrapper.cpp",
    REPO_ROOT / "tests" / "native" / "progress_callback_test.cpp",
]
INCLUDE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "third_party" / "stable-diffusion.cpp" / "include",
]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
THREAD_SANITIZER = "-fsanitize=thread"


def compile_command(output: Path, sources: list[Path], *flags: str) -> list[str]:
    includes = [f"-I{directory}" for directory in INCLUDE_DIRS]
    return [COMPILER, "-std=c++17", "-O1", "-g", "-pthread", *flags, *includes,
            *map(str, sources), "-o", str(output)]


@unittest.skipIf(sys.platform == "win32", "uses a GCC or Clang command line")
class ProgressCallbackTest(unittest.TestCase):
    def require(self, condition: bool, reason: str) -> None:
        """Skips locally, but fails in CI, where a skip would hide the test."""
        if condition:
            return
        if os.environ.get("CI"):
            self.fail(reason)
        self.skipTest(reason)

    def setUp(self) -> None:
        self.require(COMPILER is not None, "no C++ compiler; set CXX")
        self.require((INCLUDE_DIRS[1] / "stable-diffusion.h").is_file(),
                     "third_party/stable-diffusion.cpp is not checked out")

    def build_and_run(self, *flags: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "progress_callback_test"
            subprocess.run(compile_command(binary, SOURCES, *flags), check=True)
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=600,
                env={**os.environ, "TSAN_OPTIONS": "halt_on_error=1"})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def supports(self, flag: str) -> bool:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.cpp"
            source.write_text("int main() { return 0; }\n")
            binary = Path(directory) / "empty"
            built = subprocess.run(compile_command(binary, [source], flag),
                                   capture_output=True)
            return built.returncode == 0 and subprocess.run(
                [str(binary)], capture_output=True).returncode == 0

    def test_routes_and_clears_progress(self) -> None:
        self.build_and_run()

    def test_clearing_during_progress_is_free_of_data_races(self) -> None:
        self.require(self.supports(THREAD_SANITIZER),
                     f"{COMPILER} cannot build or run {THREAD_SANITIZER} binaries")
        self.build_and_run(THREAD_SANITIZER)


if __name__ == "__main__":
    unittest.main()

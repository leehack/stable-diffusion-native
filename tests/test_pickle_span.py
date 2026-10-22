"""Builds and runs `tests/native/pickle_span_test.cpp`.

The test compiles upstream's `pickle_io.cpp` from the pinned submodule with
`patches/pickle_line_span.patch` applied, as the library does, so it needs the
submodules' sources but no built runtime and no model. Spans above INT_MAX are
reserved read-only and cost no memory.
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
SCANNER = Path("src") / "model_io" / "pickle_io.cpp"
PATCH = REPO_ROOT / "patches" / "pickle_line_span.patch"
TEST_SOURCE = REPO_ROOT / "tests" / "native" / "pickle_span_test.cpp"
INCLUDE_DIRS = [UPSTREAM / "src", UPSTREAM / "src" / "model_io", UPSTREAM / "include",
                UPSTREAM / "ggml" / "include"]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
GIT = shutil.which("git")
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
        self.require(GIT is not None, "no git")
        self.require((INCLUDE_DIRS[3] / "ggml.h").is_file() and (UPSTREAM / SCANNER).is_file(),
                     "third_party/stable-diffusion.cpp is not checked out; run "
                     "`git submodule update --init --recursive`")

    def patched_scanner(self, directory: Path) -> Path:
        """Applies the carried patch as CMakeLists.txt does."""
        root = directory / "patched"
        scanner = root / SCANNER
        scanner.parent.mkdir(parents=True)
        scanner.write_bytes((UPSTREAM / SCANNER).read_bytes().replace(b"\r\n", b"\n"))
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(directory.resolve())}
        subprocess.run([GIT, "apply", str(PATCH)], cwd=root, env=env, check=True)
        self.assertNotEqual((UPSTREAM / SCANNER).read_bytes(), scanner.read_bytes())
        return scanner

    def run_mode(self, mode: str, *flags: str,
                 patched: bool = True) -> subprocess.CompletedProcess:
        includes = [f"-I{directory}" for directory in INCLUDE_DIRS]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scanner = self.patched_scanner(directory) if patched else UPSTREAM / SCANNER
            binary = directory / "pickle_span_test"
            subprocess.run([COMPILER, "-std=c++17", "-O1", "-g", *flags, *includes,
                            str(scanner), str(TEST_SOURCE), "-o", str(binary)], check=True)
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

    def test_skips_valid_pickles_in_spans_above_int_max(self) -> None:
        self.assert_passes("large-ok")

    # The pinned upstream narrows the remaining span to int before scanning for
    # a newline (#23). Once a pin no longer does, the carried patch can go.
    def test_pinned_upstream_still_needs_the_patch(self) -> None:
        result = self.run_mode("large-ok", patched=False)
        self.assertNotEqual(
            0, result.returncode,
            "the pinned upstream now skips valid pickles in spans above INT_MAX; "
            "remove patches/pickle_line_span.patch")


if __name__ == "__main__":
    unittest.main()

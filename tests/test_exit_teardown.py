"""Builds and runs `tests/native/exit_teardown_test.cpp`.

The test links `src/sd_dart_exit.cpp`, `src/sd_dart_log.cpp` and
`src/sd_dart_device.cpp` against stand-ins for the upstream functions they
wrap and for ggml's device registry, so it needs the submodules' headers but
no built runtime. On Linux it also runs the scenarios of an exit that frees
nothing and destroys no static.
Each scenario runs in its own process, as teardown runs once per process.
"""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    REPO_ROOT / "src" / "sd_dart_device.cpp",
    REPO_ROOT / "src" / "sd_dart_exit.cpp",
    REPO_ROOT / "src" / "sd_dart_log.cpp",
    REPO_ROOT / "tests" / "native" / "exit_teardown_test.cpp",
]
INCLUDE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "third_party" / "stable-diffusion.cpp" / "include",
    REPO_ROOT / "third_party" / "stable-diffusion.cpp" / "ggml" / "include",
]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
SANITIZERS = ("address", "thread")
# Scenarios that try one moment per run, and how often the unsanitized build
# runs them, four at a time. With the two halves of a creating call's
# bookkeeping in separate critical sections, 2 to 5 runs in 100 failed. A
# sanitized build runs them once; it is slower and looks for something else.
REPEATED = {"load-race": 400}
# What only Linux does at exit: free nothing, destroy no static, block no
# thread, and end the process where a call is in flight.
LINUX_SCENARIOS = ("exit-ends-process-in-generation", "exit-ends-process-in-load",
                   "exit-ends-process-in-query", "exit-ends-process-in-marked-call",
                   "exit-from-own-call", "exit-after-driver-load", "exit-after-driver-query",
                   "exit-in-fork-child", "exit-idle-after-call", "exit-refuses-late-calls")
SANITIZER_ENV = {
    # The scenarios leave contexts and blocked threads behind on purpose.
    "ASAN_OPTIONS": "detect_leaks=0",
    "TSAN_OPTIONS": "halt_on_error=1",
}


def compile_command(output: Path, sources: list[Path], *flags: str) -> list[str]:
    includes = [f"-I{directory}" for directory in INCLUDE_DIRS]
    # dlsym() is in libdl before glibc 2.34.
    libraries = ["-ldl"] if sys.platform == "linux" else []
    return [COMPILER, "-std=c++17", "-O1", "-g", "-pthread", "-DSD_DART_GPU_BACKEND=1", *flags,
            *includes, *map(str, sources), "-o", str(output), *libraries]


@unittest.skipIf(sys.platform == "win32", "uses a GCC or Clang command line")
class ExitTeardownTest(unittest.TestCase):
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
        self.require((INCLUDE_DIRS[2] / "ggml.h").is_file(),
                     "third_party/stable-diffusion.cpp/ggml is not checked out; run "
                     "`git submodule update --init --recursive`")

    def build_and_run(self, *flags: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "exit_teardown_test"
            subprocess.run(compile_command(binary, SOURCES, *flags), check=True)
            scenarios = subprocess.run([str(binary), "list"], check=True, capture_output=True,
                                       text=True).stdout.split()
            self.assertIn("generate-in-flight", scenarios)
            self.assertIn("log-at-exit", scenarios)
            self.assertIn("device-memory-in-flight", scenarios)
            for scenario in LINUX_SCENARIOS if sys.platform == "linux" else ("exit-in-flight",):
                self.assertIn(scenario, scenarios)

            def run(scenario: str) -> subprocess.CompletedProcess:
                return subprocess.run(
                    [str(binary), scenario], capture_output=True, text=True, timeout=600,
                    env={**os.environ, **SANITIZER_ENV})

            for scenario in scenarios:
                with self.subTest(scenario=scenario):
                    result = run(scenario)
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for scenario, runs in ({} if flags else REPEATED).items():
                    failures = [result.stdout + result.stderr
                                for result in pool.map(run, [scenario] * runs)
                                if result.returncode != 0]
                    with self.subTest(scenario=scenario, runs=runs):
                        self.assertEqual([], failures[:3], f"{len(failures)} of {runs} runs failed")

    def supports(self, flag: str) -> bool:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.cpp"
            source.write_text("int main() { return 0; }\n")
            binary = Path(directory) / "empty"
            built = subprocess.run(compile_command(binary, [source], flag),
                                   capture_output=True)
            return built.returncode == 0 and subprocess.run(
                [str(binary)], capture_output=True).returncode == 0

    def test_tracks_waits_and_frees_in_order(self) -> None:
        self.build_and_run()

    def test_scenarios_are_clean_under_sanitizers(self) -> None:
        for sanitizer in SANITIZERS:
            flag = f"-fsanitize={sanitizer}"
            with self.subTest(sanitizer=sanitizer):
                self.require(self.supports(flag),
                             f"{COMPILER} cannot build or run {flag} binaries")
                self.build_and_run(flag)


if __name__ == "__main__":
    unittest.main()

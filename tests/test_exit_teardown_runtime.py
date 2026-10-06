"""Runs `tests/native/exit_teardown_runtime_test.cpp` against a built runtime.

Needs the macOS runtime of this host under `bin/` (`tools/build.py build
--target macos-arm64`) and skips without it. The test binary writes its own
12 MB model, so nothing is downloaded.

On Linux, where teardown does not run at exit, it only checks that a context
left alive at exit does not need it: set `SD_EXIT_TEARDOWN_TARGET` to a built
Linux target.

    SD_REQUIRE_RUNTIME=1            fail instead of skipping
    SD_REQUIRE_DART=1               fail when no Dart SDK is on PATH; the
                                    Dart VM harness skips without one
    SD_EXIT_TEARDOWN_SANITIZER=address
                                    build the runtime with AddressSanitizer
                                    under build/ and test that one instead
    SD_EXIT_TEARDOWN_RUNS=5         runs per scenario and backend
    SD_EXIT_TEARDOWN_MODEL=<file>   also run the scenarios on this model
    SD_EXIT_TEARDOWN_MODEL_SIZE=256 the image size that model needs
    SD_EXIT_TEARDOWN_LLAMADART=<libllamadart.dylib>
                                    also load llamadart-native's library into
                                    the same process; it needs a build with
                                    that repository's own exit teardown
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build  # noqa: E402

TARGET = build.TARGETS["macos-arm64" if platform.machine() == "arm64" else "macos-x64"]
SOURCE = REPO_ROOT / "tests" / "native" / "exit_teardown_runtime_test.cpp"
DART_PROBE = REPO_ROOT / "tests" / "native" / "exit_teardown_dart_probe.cpp"
DART_HARNESS = REPO_ROOT / "tests" / "dart" / "exit_teardown.dart"
DART_SCENARIOS = ("idle", "leaked", "killed-in-load", "killed-in-run", "exit-in-run")
COMPILER = os.environ.get("CXX") or shutil.which("c++")
DART = shutil.which("dart")
TWO_LIBRARIES = REPO_ROOT / "tests" / "native" / "exit_teardown_two_libraries_test.cpp"
LLAMADART = os.environ.get("SD_EXIT_TEARDOWN_LLAMADART", "")
LINUX_TARGET = os.environ.get("SD_EXIT_TEARDOWN_TARGET", "")
SANITIZER = os.environ.get("SD_EXIT_TEARDOWN_SANITIZER", "")
RUNS = int(os.environ.get("SD_EXIT_TEARDOWN_RUNS", "1"))
REAL_MODEL = os.environ.get("SD_EXIT_TEARDOWN_MODEL", "")
REAL_MODEL_SIZE = os.environ.get("SD_EXIT_TEARDOWN_MODEL_SIZE", "256")
SCENARIOS = ("idle", "dispose", "cancel", "generate-wait", "load-wait", "late-load")
# `default` lets the runtime pick its device, which is Metal where there is one.
BACKENDS = ("default", "cpu")
METAL_ABORT = "[rsets->data count] == 0"
SCENARIO_ENV = {"ASAN_OPTIONS": "detect_leaks=0"}


def sanitized_library(sanitizer: str) -> Path:
    """Builds the runtime as `tools/build.py` does, plus the sanitizer."""
    work_dir = build.BUILD_ROOT / f"{TARGET.name}-{sanitizer}"
    work_dir.mkdir(parents=True, exist_ok=True)
    flags = f"-fsanitize={sanitizer} -fno-omit-frame-pointer -g"
    args = [f"{arg} {flags}" if arg.startswith("-DCMAKE_SHARED_LINKER_FLAGS=") else arg
            for arg in build.configure_args(TARGET, work_dir)]
    args += [f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}"]
    build.run(["cmake", "-S", str(REPO_ROOT), "-B", str(work_dir), *args])
    build.run(["cmake", "--build", str(work_dir), "-j", str(os.cpu_count() or 4)])
    return build.find_library(work_dir, TARGET.library)


def build_test(library: Path, output: Path, *flags: str, source: Path = SOURCE) -> None:
    subprocess.run(
        [COMPILER, "-std=c++17", "-O1", "-g", "-pthread", *flags, f"-I{REPO_ROOT / 'src'}",
         f"-I{build.HEADER.parent}", str(source), f"-L{library.parent}",
         "-lstable-diffusion", f"-Wl,-rpath,{library.parent}", "-o", str(output)],
        check=True)


@unittest.skipUnless(sys.platform == "linux" and LINUX_TARGET,
                     "set SD_EXIT_TEARDOWN_TARGET to a built Linux target")
class LiveContextAtExitTest(unittest.TestCase):
    """Nothing frees a context at exit here, so exiting with one must be harmless.

    Checked on the CPU and, for a Vulkan target, on its first Vulkan device.
    What happens on Vulkan is reported, not failed: it is a property of the
    driver at hand, which CI only has in software (Mesa lavapipe).
    """

    def test_exits_cleanly_with_a_context_alive(self) -> None:
        target = build.TARGETS[LINUX_TARGET]
        library = build.BIN_ROOT / target.name / "lib" / target.library
        self.assertTrue(library.is_file(), f"no runtime at {library}")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "exit_teardown_runtime_test"
            model = Path(directory) / "model.safetensors"
            build_test(library, binary)
            subprocess.run([str(binary), "make-model", str(model)], check=True, timeout=600)
            backends = ["cpu"] + (["Vulkan0"] if "vulkan" in target.accelerators else [])
            for backend in backends:
                with self.subTest(backend=backend):
                    result = subprocess.run(
                        [str(binary), "idle-untracked", str(model), backend],
                        capture_output=True, text=True, timeout=600, errors="replace")
                    generated = f"generated on {backend}" in result.stdout
                    lines = result.stderr.strip().splitlines()
                    outcome = (f"{target.name} on {backend}: "
                               + ("exit with a live context" if generated
                                  else "no context to exit with; the run")
                               + f" returned {result.returncode}"
                               + (f": {lines[-1]}" if result.returncode != 0 and lines else ""))
                    level = "notice" if generated and result.returncode == 0 else "warning"
                    print(f"::{level} title=exit teardown::{outcome}"
                          if os.environ.get("GITHUB_ACTIONS") else f"{level}: {outcome}", flush=True)
                    if backend == "cpu":
                        self.assertTrue(generated, result.stderr)
                        self.assertEqual(0, result.returncode, result.stderr)


@unittest.skipUnless(sys.platform == "darwin", "exit teardown runs at exit on Apple platforms only")
class ExitTeardownRuntimeTest(unittest.TestCase):
    binary: Path
    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        library = build.BIN_ROOT / TARGET.name / "lib" / TARGET.library
        if SANITIZER:
            library = sanitized_library(SANITIZER)
        elif not library.is_file():
            reason = f"no runtime at {library}; build {TARGET.name} first"
            if os.environ.get("SD_REQUIRE_RUNTIME"):
                raise AssertionError(reason)
            raise unittest.SkipTest(reason)
        cls.library = library
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "exit_teardown_runtime_test"
        cls.model = Path(cls.directory.name) / "model.safetensors"
        cls.compile(SOURCE, cls.binary)
        subprocess.run([str(cls.binary), "make-model", str(cls.model)], check=True, timeout=600,
                       env={**os.environ, **SCENARIO_ENV})

    @classmethod
    def compile(cls, source: Path, output: Path, *flags: str) -> None:
        sanitize = [f"-fsanitize={SANITIZER}"] if SANITIZER else []
        build_test(cls.library, output, *sanitize, *flags, source=source)

    def run_scenario(self, scenario: str, backend: str, model: Path | str | None = None,
                     size: str | None = None) -> subprocess.CompletedProcess:
        command = [str(self.binary), scenario, str(model or self.model), backend]
        return subprocess.run(command + ([size] if size else []), capture_output=True, text=True,
                              timeout=600, errors="replace", env={**os.environ, **SCENARIO_ENV})

    def check_scenarios(self, model: Path | str | None = None, size: str | None = None) -> None:
        for scenario in SCENARIOS:
            for backend in BACKENDS:
                for run in range(RUNS):
                    with self.subTest(scenario=scenario, backend=backend, run=run):
                        result = self.run_scenario(scenario, backend, model, size)
                        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_frees_tracked_contexts_at_exit(self) -> None:
        self.check_scenarios()

    @unittest.skipUnless(REAL_MODEL, "set SD_EXIT_TEARDOWN_MODEL to a model file")
    def test_frees_tracked_contexts_of_a_real_model_at_exit(self) -> None:
        self.check_scenarios(REAL_MODEL, REAL_MODEL_SIZE)

    def test_untracked_context_shows_whether_metal_aborts(self) -> None:
        """The control: a context from new_sd_ctx() is left alone, as before.

        ggml-metal then aborts at exit where Metal residency sets are live. A
        clean exit is not a failure of this repository, but it means the
        scenarios above showed the frees through the allocator only.
        """
        self.assertEqual(0, self.run_scenario("idle-untracked", "cpu").returncode)
        result = self.run_scenario("idle-untracked", "default")
        if result.returncode == 0:
            message = ("an untracked context exited cleanly: no Metal residency set was live, "
                       "so the Metal abort at exit was not exercised on this machine")
            print(f"::warning title=exit teardown::{message}" if os.environ.get("GITHUB_ACTIONS")
                  else f"note: {message}", flush=True)
            return
        self.assertEqual(-signal.SIGABRT, result.returncode, result.stdout + result.stderr)
        self.assertIn(METAL_ABORT, result.stderr)
        print("note: an untracked context aborted in ggml-metal at exit, as before", flush=True)

    def test_dart_vm_exits_cleanly_with_contexts_no_dart_code_frees(self) -> None:
        """`tests/dart/exit_teardown.dart`: what a Flutter quit or hot restart leaves."""
        if SANITIZER:
            self.skipTest("the Dart VM cannot load a sanitized runtime")
        if DART is None:
            if os.environ.get("SD_REQUIRE_DART"):
                self.fail("needs a Dart SDK on PATH")
            self.skipTest("needs a Dart SDK on PATH")
        probe = Path(self.directory.name) / "libexit_teardown_dart_probe.dylib"
        self.compile(DART_PROBE, probe, "-shared", "-fPIC")

        def run(scenario: str, api: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [DART, str(DART_HARNESS), str(self.library), str(probe), str(self.model),
                 scenario, api], capture_output=True, text=True, timeout=600, errors="replace")

        for scenario in DART_SCENARIOS:
            for attempt in range(RUNS):
                with self.subTest(scenario=scenario, run=attempt):
                    result = run(scenario, "tracked")
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        # The control: with upstream's functions the same exits abort where
        # Metal residency sets are live, and nowhere else.
        aborted = [scenario for scenario in DART_SCENARIOS
                   if METAL_ABORT in run(scenario, "raw").stderr]
        print(f"note: with new_sd_ctx and generate_image, {len(aborted)} of "
              f"{len(DART_SCENARIOS)} Dart scenarios aborted in ggml-metal", flush=True)

    @unittest.skipUnless(LLAMADART, "set SD_EXIT_TEARDOWN_LLAMADART to a libllamadart.dylib")
    def test_shares_a_process_with_libllamadart(self) -> None:
        """Each library tears down its own registry, whichever was loaded first."""
        binary = Path(self.directory.name) / "exit_teardown_two_libraries_test"
        subprocess.run([COMPILER, "-std=c++17", "-O1", "-g", f"-I{REPO_ROOT / 'src'}",
                        f"-I{build.HEADER.parent}", str(TWO_LIBRARIES), "-o", str(binary)],
                       check=True)
        for first in ("sd", "llama"):
            for mode in ("exit", "sd", "llama"):
                for backend in BACKENDS:
                    with self.subTest(first=first, mode=mode, backend=backend):
                        result = subprocess.run(
                            [str(binary), str(self.library), LLAMADART, first, mode,
                             str(self.model), backend],
                            capture_output=True, text=True, timeout=600, errors="replace")
                        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                        self.assertCountEqual(
                            ["freed by libstable-diffusion", "freed by libllamadart"],
                            result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()

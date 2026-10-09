"""Runs `tests/native/exit_teardown_runtime_test.cpp` against a built runtime.

Needs the macOS runtime of this host under `bin/` (`tools/build.py build
--target macos-arm64`) and skips without it. The test binary writes its own
12 MB model, so nothing is downloaded.

On Linux, where exit() frees nothing and ends the process when a call is in
flight, it runs the scenarios of that against a built Linux target: set
`SD_EXIT_TEARDOWN_TARGET` to its name. They have to pass on the CPU. What a
Vulkan target does on its first Vulkan device is reported and not failed.

    SD_REQUIRE_RUNTIME=1            fail instead of skipping
    SD_REQUIRE_DART=1               fail when no Dart SDK is on PATH; the
                                    Dart VM harness skips without one
    SD_REQUIRE_METAL_GENERATION=1   fail when the default device cannot run a
                                    generation, instead of only loading
                                    models on it; set it on a Mac
    SD_EXIT_TEARDOWN_SANITIZER=address
                                    build the runtime with AddressSanitizer
                                    under build/ and test that one instead,
                                    on macOS or Linux
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
DART_SCENARIOS = ("idle", "leaked", "killed-in-load", "killed-in-run", "exit-in-run",
                  "exit-in-free")
# How long after telling a worker to free its context `exit-in-free` exits.
DART_FREE_DELAYS_US = (0, 2000, 6000)
COMPILER = os.environ.get("CXX") or shutil.which("c++")
DART = shutil.which("dart")
TWO_LIBRARIES = REPO_ROOT / "tests" / "native" / "exit_teardown_two_libraries_test.cpp"
LLAMADART = os.environ.get("SD_EXIT_TEARDOWN_LLAMADART", "")
LINUX_TARGET = os.environ.get("SD_EXIT_TEARDOWN_TARGET", "")
SANITIZER = os.environ.get("SD_EXIT_TEARDOWN_SANITIZER", "")
RUNS = int(os.environ.get("SD_EXIT_TEARDOWN_RUNS", "1"))
REAL_MODEL = os.environ.get("SD_EXIT_TEARDOWN_MODEL", "")
REAL_MODEL_SIZE = os.environ.get("SD_EXIT_TEARDOWN_MODEL_SIZE", "256")
SCENARIOS = ("idle", "dispose", "free-quit", "cancel", "generate-wait", "load-wait", "late-load",
             "log", "query-quit")
# What the others do on a device that cannot compute is load, and free.
GENERATING_SCENARIOS = ("cancel", "generate-wait")
# `default` lets the runtime pick its device, which is Metal where there is one.
BACKENDS = ("default", "cpu")
# On Linux exit() frees nothing and does not wait, so the scenarios of an exit
# that does are left out. Added: a context that upstream's own functions
# created and use, an exit with a call in flight, and a host whose exit handler
# joins its worker.
LINUX_SCENARIOS = tuple(
    scenario for scenario in SCENARIOS
    if scenario not in ("generate-wait", "load-wait", "free-quit")
) + ("idle-untracked", "exit-in-load", "exit-in-generation", "exit-status",
     "generate-through-exit", "join-idle")
# The status a scenario passes to exit(), where it is not 0, and what it
# leaves in stdio's buffer when it exits with a call in flight.
EXIT_STATUS = {"exit-status": 37}
BUFFERED_OUTPUT = {"exit-in-generation": "generating on", "exit-status": "generating on",
                   "exit-in-load": "loading on"}
METAL_ABORT = "[rsets->data count] == 0"
SCENARIO_ENV = {"ASAN_OPTIONS": "detect_leaks=0"}


def sanitized_library(sanitizer: str, target: build.Target = TARGET) -> Path:
    """Builds the runtime as `tools/build.py` does, plus the sanitizer."""
    work_dir = build.BUILD_ROOT / f"{target.name}-{sanitizer}"
    work_dir.mkdir(parents=True, exist_ok=True)
    flags = f"-fsanitize={sanitizer} -fno-omit-frame-pointer -g"
    args = [f"{arg} {flags}" if arg.startswith("-DCMAKE_SHARED_LINKER_FLAGS=") else arg
            for arg in build.configure_args(target, work_dir)]
    args += [f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}"]
    build.run(["cmake", "-S", str(REPO_ROOT), "-B", str(work_dir), *args])
    build.run(["cmake", "--build", str(work_dir), "-j", str(os.cpu_count() or 4)])
    return build.find_library(work_dir, target.library)


def build_test(library: Path, output: Path, *flags: str, source: Path = SOURCE) -> None:
    subprocess.run(
        [COMPILER, "-std=c++17", "-O1", "-g", "-pthread", *flags, f"-I{REPO_ROOT / 'src'}",
         f"-I{build.HEADER.parent}", str(source), f"-L{library.parent}",
         "-lstable-diffusion", f"-Wl,-rpath,{library.parent}", "-o", str(output)],
        check=True)


def annotate(level: str, message: str) -> None:
    print(f"::{level} title=exit teardown::{message}" if os.environ.get("GITHUB_ACTIONS")
          else f"{level}: {message}", flush=True)


@unittest.skipUnless(sys.platform == "linux" and LINUX_TARGET,
                     "set SD_EXIT_TEARDOWN_TARGET to a built Linux target")
class ExitWaitRuntimeTest(unittest.TestCase):
    """On Linux exit() frees nothing and ends the process under a call in flight.

    The scenarios have to pass on the CPU. For a Vulkan target they also run
    on its first Vulkan device, where the outcome is reported, not failed: it
    is a property of the driver at hand, which CI only has in software (Mesa
    lavapipe).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.target = build.TARGETS[LINUX_TARGET]
        library = build.BIN_ROOT / cls.target.name / "lib" / cls.target.library
        if SANITIZER:
            library = sanitized_library(SANITIZER, cls.target)
        elif not library.is_file():
            raise AssertionError(f"no runtime at {library}")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "exit_teardown_runtime_test"
        cls.model = Path(cls.directory.name) / "model.safetensors"
        build_test(library, cls.binary, *([f"-fsanitize={SANITIZER}"] if SANITIZER else []))
        subprocess.run([str(cls.binary), "make-model", str(cls.model)], check=True, timeout=600,
                       env={**os.environ, **SCENARIO_ENV})
        cls.backends = ["cpu"] + (["Vulkan0"] if "vulkan" in cls.target.accelerators else [])

    def outcome(self, scenario: str, backend: str) -> tuple[str, str]:
        """How one run ended, and its last line on stderr."""
        try:
            result = subprocess.run(
                [str(self.binary), scenario, str(self.model), backend], capture_output=True,
                text=True, timeout=600, errors="replace", env={**os.environ, **SCENARIO_ENV})
        except subprocess.TimeoutExpired:
            return "hung", ""
        lines = result.stderr.strip().splitlines()
        if result.returncode == EXIT_STATUS.get(scenario, 0):
            if BUFFERED_OUTPUT.get(scenario, "") in result.stdout:
                return "clean", ""
            return "lost its buffered output", ""
        ended = (signal.Signals(-result.returncode).name if result.returncode < 0
                 else f"exit code {result.returncode}")
        return ended, lines[-1][:200] if lines else ""

    def test_exit_frees_nothing_and_ends_the_process_under_a_call(self) -> None:
        for backend in self.backends:
            for scenario in LINUX_SCENARIOS:
                outcomes: dict[str, int] = {}
                detail = ""
                for _ in range(RUNS):
                    ended, last_line = self.outcome(scenario, backend)
                    outcomes[ended] = outcomes.get(ended, 0) + 1
                    detail = last_line or detail
                summary = ", ".join(f"{count} {ended}" for ended, count in sorted(outcomes.items()))
                message = (f"{self.target.name} on {backend}: {scenario}, {RUNS} runs: {summary}"
                           + (f"; last line: {detail}" if detail else ""))
                if backend == "cpu":
                    with self.subTest(scenario=scenario):
                        self.assertEqual({"clean"}, set(outcomes), message)
                elif set(outcomes) != {"clean"}:
                    annotate("warning", message)
            if backend != "cpu":
                annotate("notice", f"{self.target.name} ran the exit scenarios on {backend}")


@unittest.skipUnless(sys.platform == "darwin", "exit frees the tracked objects on Apple platforms only")
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
        # GitHub's macOS runners load a model on their virtual GPU but crash in
        # ggml-metal as soon as it computes, whoever created the context.
        cls.load_only = {
            backend for backend in BACKENDS
            if subprocess.run([str(cls.binary), "generate", str(cls.model), backend],
                              capture_output=True, timeout=600,
                              env={**os.environ, **SCENARIO_ENV}).returncode != 0}
        if "cpu" in cls.load_only:
            raise AssertionError("the runtime cannot generate an image on the CPU")
        if cls.load_only and os.environ.get("SD_REQUIRE_METAL_GENERATION"):
            raise AssertionError("the default device cannot run a generation on this machine")
        if cls.load_only:
            message = ("the default device cannot run a generation on this machine: "
                       "there the scenarios only load a model, and those that need a "
                       "generation run on the CPU alone")
            print(f"::warning title=exit teardown::{message}" if os.environ.get("GITHUB_ACTIONS")
                  else f"note: {message}", flush=True)

    @classmethod
    def compile(cls, source: Path, output: Path, *flags: str) -> None:
        sanitize = [f"-fsanitize={SANITIZER}"] if SANITIZER else []
        build_test(cls.library, output, *sanitize, *flags, source=source)

    def run_scenario(self, scenario: str, backend: str, model: Path | str | None = None,
                     size: str | None = None) -> subprocess.CompletedProcess:
        command = [str(self.binary), scenario, str(model or self.model), backend]
        load_only = {"SD_EXIT_TEARDOWN_LOAD_ONLY": "1"} if backend in self.load_only else {}
        return subprocess.run(command + ([size] if size else []), capture_output=True, text=True,
                              timeout=600, errors="replace",
                              env={**os.environ, **SCENARIO_ENV, **load_only})

    def check_scenarios(self, model: Path | str | None = None, size: str | None = None) -> None:
        for scenario in SCENARIOS:
            for backend in BACKENDS:
                if scenario in GENERATING_SCENARIOS and backend in self.load_only:
                    continue
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

        backend = "cpu" if self.load_only else "default"
        models = [(str(self.model), "64")] + ([(REAL_MODEL, REAL_MODEL_SIZE)] if REAL_MODEL else [])

        def run(scenario: str, api: str, model: str, size: str,
                attempt: int = 0) -> subprocess.CompletedProcess:
            delay = DART_FREE_DELAYS_US[attempt % len(DART_FREE_DELAYS_US)]
            return subprocess.run(
                [DART, str(DART_HARNESS), str(self.library), str(probe), model, scenario, api,
                 backend, size, str(delay)],
                capture_output=True, text=True, timeout=600, errors="replace")

        for model, size in models:
            for scenario in DART_SCENARIOS:
                for attempt in range(RUNS):
                    with self.subTest(model=Path(model).name, scenario=scenario, run=attempt):
                        result = run(scenario, "tracked", model, size, attempt)
                        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            # The control: with upstream's functions the same exits abort where
            # Metal residency sets are live, and nowhere else.
            aborted = [scenario for scenario in DART_SCENARIOS
                       if METAL_ABORT in run(scenario, "raw", model, size).stderr]
            print(f"note: with upstream's functions on {backend}, {len(aborted)} of "
                  f"{len(DART_SCENARIOS)} Dart scenarios aborted in ggml-metal "
                  f"({Path(model).name})", flush=True)

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

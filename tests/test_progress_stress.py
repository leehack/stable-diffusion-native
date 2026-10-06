"""Runs the `tests/dart` harnesses, which need a real Dart VM.

`progress_stress.dart` dies of an unhandled Dart error while a worker isolate
is inside a native call that reports progress, with other isolates
allocating, and must still exit with that error. `progress_images.dart`
polls the reports of a batch the way a Dart binding would and must see all of
them.

The default run is short. `SD_PROGRESS_STRESS_RUNS` sets the runs per
configuration (CI uses 25), and `SD_REQUIRE_DART=1` fails instead of skipping
when no Dart SDK is on PATH. To run one configuration, controls included:

    python3 tests/test_progress_stress.py run locked main --allocating 2 --runs 25
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tests" / "dart" / "progress_stress.dart"
IMAGES_HARNESS = REPO_ROOT / "tests" / "dart" / "progress_images.dart"
SOURCES = [
    REPO_ROOT / "src" / "sd_dart_wrapper.cpp",
    REPO_ROOT / "tests" / "native" / "progress_stress.cpp",
]
INCLUDE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "third_party" / "stable-diffusion.cpp" / "include",
]
COMPILER = os.environ.get("CXX") or shutil.which("c++")
DART = shutil.which("dart")
DESIGNS = ("poll", "raw", "locked")
OWNERS = ("main", "background", "killed")
REPORT_MS = 1500
DIE_AFTER_MS = 600
# A report gap at which a 1 ms poller must lose nothing, while the reports of
# one run still go around the library's ring several times.
LOSSLESS_GAP_US = 20
HANG_SECONDS = 20
UNHANDLED_ERROR_EXIT = 255
# How the VM dies when native code calls a callback it can no longer run.
ABORT_MESSAGES = (
    "GetFfiCallbackMetadata called after shutdown",
    "Callback invoked after it has been deleted",
    "===== CRASH =====",
)


def build_library(directory: Path) -> Path:
    library = directory / ("libprogress_stress.dylib" if sys.platform == "darwin"
                           else "libprogress_stress.so")
    includes = [f"-I{path}" for path in INCLUDE_DIRS]
    subprocess.run([COMPILER, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
                    *includes, *map(str, SOURCES), "-o", str(library)], check=True)
    return library


def run_once(library: Path, design: str, owner: str, allocating: int, gap_us: int,
             sample: Path | None = None, lossless: bool = False) -> str:
    """Returns `clean`, `abort`, `hang` or `exit <code>: <last stderr line>`."""
    process = subprocess.Popen(
        [DART, str(HARNESS), str(library), design, owner, str(REPORT_MS), str(gap_us),
         str(DIE_AFTER_MS), str(allocating), *(["lossless"] if lossless else [])],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace",
        start_new_session=True)
    try:
        _, stderr = process.communicate(timeout=HANG_SECONDS)
    except subprocess.TimeoutExpired:
        if sample is not None and shutil.which("sample"):
            subprocess.run(["sample", str(process.pid), "1", "-file", str(sample)],
                           capture_output=True)
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return "hang"
    if any(message in stderr for message in ABORT_MESSAGES):
        return "abort"
    if process.returncode == UNHANDLED_ERROR_EXIT and "stress: dying after" in stderr:
        return "clean"
    lines = stderr.strip().splitlines()
    return f"exit {process.returncode}: {lines[-1] if lines else ''}"


def run_configuration(library: Path, design: str, owner: str, allocating: int,
                      gap_us: int, runs: int, sample: Path | None = None,
                      lossless: bool = False) -> collections.Counter:
    outcomes: collections.Counter = collections.Counter()
    for _ in range(runs):
        outcome = run_once(library, design, owner, allocating, gap_us,
                           None if outcomes["hang"] else sample, lossless)
        outcomes[outcome] += 1
    return outcomes


def run_images(library: Path, consumer: str, images: int, steps: int, step_us: int,
               tiles: int = 0, tile_us: int = 0, poll_ms: int = 50) -> subprocess.CompletedProcess:
    return subprocess.run(
        [DART, str(IMAGES_HARNESS), str(library), consumer, str(images), str(steps),
         str(step_us), str(tiles), str(tile_us), str(poll_ms)],
        capture_output=True, text=True, timeout=120)


@unittest.skipIf(sys.platform == "win32", "uses a GCC or Clang command line")
class ProgressStressTest(unittest.TestCase):
    runs = int(os.environ.get("SD_PROGRESS_STRESS_RUNS", "2"))

    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name, found in (
            ("a Dart SDK on PATH", DART),
            ("a C++ compiler", COMPILER),
            ("third_party/stable-diffusion.cpp", (INCLUDE_DIRS[1] / "stable-diffusion.h").is_file()),
        ) if not found]
        if missing:
            message = f"needs {', '.join(missing)}"
            if os.environ.get("SD_REQUIRE_DART"):
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = build_library(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_polling_never_aborts_or_hangs_when_the_owner_goes_away(self) -> None:
        for owner in OWNERS:
            for allocating in (2, 6):
                with self.subTest(owner=owner, allocating=allocating):
                    outcomes = run_configuration(self.library, "poll", owner, allocating,
                                                 gap_us=0, runs=self.runs)
                    self.assertEqual({"clean": self.runs}, dict(outcomes))

    def test_polling_loses_no_report_the_library_still_keeps(self) -> None:
        for owner in OWNERS:
            with self.subTest(owner=owner):
                outcomes = run_configuration(self.library, "poll", owner, allocating=2,
                                             gap_us=LOSSLESS_GAP_US, runs=self.runs,
                                             lossless=True)
                self.assertEqual({"clean": self.runs}, dict(outcomes))

    def test_polling_sees_every_report_of_a_batch(self) -> None:
        # images, steps, microseconds per step, tiles, microseconds per tile
        for batch in ((3, 4, 20000, 0, 0), (2, 1, 30000, 0, 0), (8, 1, 2000, 0, 0),
                      (2, 4, 5000, 961, 100)):
            with self.subTest(batch=batch):
                result = run_images(self.library, "batch", *batch)
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_a_consumer_of_only_the_latest_report_misses_some(self) -> None:
        # Without this the test above could pass by reporting too slowly.
        result = run_images(self.library, "latest", 3, 4, 20000)
        self.assertEqual(6, result.returncode, result.stdout + result.stderr)

    def test_harness_sees_a_dart_callback_abort(self) -> None:
        # Without this the test above could pass by not exercising shutdown.
        outcomes = run_configuration(self.library, "raw", "main", allocating=0,
                                     gap_us=0, runs=self.runs)
        self.assertGreater(outcomes["abort"], 0, dict(outcomes))
        self.assertNotIn("clean", outcomes)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        unittest.main()
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("design", choices=DESIGNS)
    parser.add_argument("owner", choices=OWNERS)
    parser.add_argument("--allocating", type=int, default=2,
                        help="Isolates that allocate in a loop.")
    parser.add_argument("--gap-us", type=int, default=0,
                        help="Microseconds between progress reports; 0 is a tight loop.")
    parser.add_argument("--lossless", action="store_true",
                        help="Fail a `poll` run that misses any report.")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--sample", type=Path,
                        help="Where to write a macOS `sample` of the first hang.")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        outcomes = run_configuration(build_library(Path(directory)), args.design, args.owner,
                                     args.allocating, args.gap_us, args.runs, args.sample,
                                     args.lossless)
    summary = ", ".join(f"{count} {outcome}" for outcome, count in sorted(outcomes.items()))
    print(f"{args.design} {args.owner} allocating={args.allocating} gap={args.gap_us}us "
          f"runs={args.runs}: {summary}")


if __name__ == "__main__":
    main()

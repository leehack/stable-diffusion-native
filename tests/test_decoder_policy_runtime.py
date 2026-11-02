"""Opt-in real SDXS/source-refresh regression; uses existing local model files.

No download or physical-device claim. CPU always runs; set
SD_DECODER_POLICY_BACKENDS=cpu,MTL0 for a local Mac with Metal.
"""
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build  # noqa: E402


class DecoderPolicyRuntimeTest(unittest.TestCase):
    def test_intrinsic_policy_survives_real_source_refresh(self):
        tiny = Path(os.environ.get("SD_INTRINSIC_TINY_MODEL", ""))
        full = Path(os.environ.get("SD_FULL_VAE_MODEL", ""))
        if not tiny.is_file() or not full.is_file():
            self.skipTest("set SD_INTRINSIC_TINY_MODEL and SD_FULL_VAE_MODEL to existing SDXS and SD-Turbo GGUFs")
        target_name = os.environ.get("SD_DECODER_POLICY_TARGET")
        if not target_name and sys.platform == "darwin":
            target_name = "macos-arm64" if platform.machine() == "arm64" else "macos-x64"
        if not target_name:
            self.fail("set SD_DECODER_POLICY_TARGET to a built target")
        target = build.TARGETS[target_name]
        library = build.BIN_ROOT / target.name / "lib" / target.library
        self.assertTrue(library.is_file(), f"no built library at {library}")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            binary = directory / "decoder_policy"
            subprocess.run([
                os.environ.get("CXX") or shutil.which("c++"), "-std=c++17", "-O1",
                f"-I{ROOT / 'src'}", f"-I{build.HEADER.parent}",
                str(ROOT / "tests/native/decoder_policy_runtime_test.cpp"),
                f"-L{library.parent}", "-lstable-diffusion", f"-Wl,-rpath,{library.parent}",
                "-o", str(binary),
            ], check=True)
            for backend in os.environ.get("SD_DECODER_POLICY_BACKENDS", "cpu").split(","):
                for mode in ("unchanged", "refresh-tiny", "refuse-reload", "unrestricted-reload",
                             "refuse-full-load", "refuse-external-load"):
                    with self.subTest(backend=backend, mode=mode):
                        live = directory / "live.gguf"
                        replacement = directory / "replacement.gguf"
                        for dest in (live, replacement):
                            dest.unlink(missing_ok=True)
                        def copy(source, destination):
                            if sys.platform == "darwin":
                                subprocess.run(["cp", "-c", str(source), str(destination)], check=True)
                            else:
                                shutil.copyfile(source, destination)
                        copy(full if mode == "refuse-full-load" else tiny, live)
                        copy(tiny if mode in ("refresh-tiny", "unchanged") else full, replacement)
                        if mode == "refresh-tiny":
                            stat = live.stat()
                            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
                        result = subprocess.run([str(binary), mode, backend, str(live), str(replacement)],
                                                capture_output=True, text=True, timeout=180)
                        print(f"{backend} {mode}: {result.returncode} {result.stdout.strip()}", flush=True)
                        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

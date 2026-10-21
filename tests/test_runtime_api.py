"""Exercise the changed upstream ABI against a built runtime without a model or GPU."""

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
from smoke_test import write_safetensors  # noqa: E402


class RuntimeApiTest(unittest.TestCase):
    def test_context_defaults_and_component_conversion(self):
        target_name = os.environ.get("SD_RUNTIME_API_TARGET", "")
        if not target_name and sys.platform == "darwin":
            target_name = "macos-arm64" if platform.machine() == "arm64" else "macos-x64"
        if not target_name or sys.platform == "win32":
            self.skipTest("set SD_RUNTIME_API_TARGET to a built macOS or Linux target")
        target = build.TARGETS[target_name]
        library = build.BIN_ROOT / target.name / "lib" / target.library
        if not library.is_file():
            self.skipTest(f"no built runtime for {target.name}")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "probe.safetensors"
            output = directory / "probe.gguf"
            write_safetensors(source)
            binary = directory / "runtime_api_test"
            subprocess.run([
                os.environ.get("CXX") or shutil.which("c++"), "-std=c++17", "-O1",
                f"-I{ROOT / 'src'}", f"-I{build.HEADER.parent}",
                str(ROOT / "tests/native/runtime_api_test.cpp"),
                f"-L{library.parent}", "-lstable-diffusion", f"-Wl,-rpath,{library.parent}",
                "-o", str(binary),
            ], check=True)
            result = subprocess.run([str(binary), str(source), str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual(b"GGUF", output.read_bytes()[:4])


if __name__ == "__main__":
    unittest.main()

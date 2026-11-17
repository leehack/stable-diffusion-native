"""Model-free ABI and version-selection controls for the fixed 1104 refresh.

Historical allocator backports remain executable controls. These checks prove
source selection and layouts, never iOS footprint or retirement.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party/stable-diffusion.cpp"
GGML = UPSTREAM / "ggml"
OLD_SD = "be486498eadc29dc3cd234d944c2180ed42b1415"
OLD_GGML = "a5915eb0e47adef21c3f25b889a008925ea838c2"
NEW_GGML = "eeda9ec48fc84a3df0bb9dcc19ae9053075a2539"
FILES = ("src/ggml.c", "src/ggml-metal/ggml-metal-device.m")


class UpstreamRefreshTest(unittest.TestCase):
    def selection(self, revision, source):
        script = source / "select.cmake"
        script.write_text(f'include("{ROOT}/cmake/AppleAllocatorSelection.cmake")\n'
                          f'sd_native_apple_allocator_selection("{revision}" "{source}" result)\n'
                          'message(STATUS "SELECTED=${result}")\n')
        return subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)

    def test_allocator_selection_is_explicit_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in FILES:
                dest = source / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(GGML / name, dest)
            for revision, expected in ((OLD_GGML, "backport"), (NEW_GGML, "upstream-experimental")):
                result = self.selection(revision, source)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("SELECTED=" + expected, result.stdout)
            self.assertNotEqual(0, self.selection("unreviewed", source).returncode)
            for name in FILES:
                path = source / name
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(0, self.selection(NEW_GGML, source).returncode,
                             "Windows CRLF checkout must select the same reviewed sources")
            for name in FILES:
                path = source / name
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            for name in FILES:
                path = source / name
                original = path.read_bytes()
                # Removing the actual deallocation must close admission, even
                # when the git revision still claims the reviewed candidate.
                self.assertIn(b"vm_deallocate", original)
                path.write_bytes(original.replace(b"vm_deallocate((vm_map_t)", b"free((vm_map_t)", 1))
                self.assertNotEqual(0, self.selection(NEW_GGML, source).returncode)
                path.write_bytes(original)

    def test_historical_allocator_backports_preserve_original_negative_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name, patch in zip(FILES, ("ggml_apple_vm_allocate.patch", "ggml_metal_apple_vm_allocate.patch")):
                old = subprocess.check_output(["git", "-C", str(GGML), "show", f"{OLD_GGML}:{name}"])
                path = source / "ggml" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(old)
                self.assertIn(b"TARGET_OS_OSX", old)
                subprocess.run(["git", "apply", str(ROOT / "patches" / patch)], cwd=source,
                               env={**os.environ, "GIT_CEILING_DIRECTORIES": directory}, check=True)
                patched = path.read_bytes()
                self.assertNotEqual(old, patched)
                if name.endswith("ggml.c"):
                    self.assertEqual(2, patched.count(b"#elif defined(__APPLE__) && !defined(GGML_TSAN_ENABLED)"))
                    self.assertIn(b"vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size)", patched)
                else:
                    begin = patched.index(b"static void * ggml_metal_host_malloc")
                    alloc = patched[begin:patched.index(b"\n}", begin)]
                    self.assertNotIn(b"posix_memalign", alloc)
                    self.assertIn(b"if (buf->is_shared && buf->owned) {\n        vm_deallocate", patched)
                # A per-carry omission really restores the old source. This is
                # a source negative control, not a substitute for original iOS.
                subprocess.run(["git", "apply", "-R", str(ROOT / "patches" / patch)], cwd=source,
                               env={**os.environ, "GIT_CEILING_DIRECTORIES": directory}, check=True)
                self.assertEqual(old, path.read_bytes())

    @unittest.skipUnless(shutil.which("c++"), "requires C++ compiler")
    def test_compiled_old_and_candidate_generation_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            old = subprocess.check_output(["git", "-C", str(UPSTREAM), "show",
                                           f"{OLD_SD}:include/stable-diffusion.h"])
            for label, header, expected in (("old", old, "336 552 592"),
                                             ("candidate", (UPSTREAM / "include/stable-diffusion.h").read_bytes(),
                                              "336 560 592")):
                include = work / label
                include.mkdir()
                (include / "stable-diffusion.h").write_bytes(header)
                source = include / "layout.cpp"
                source.write_text('#include "stable-diffusion.h"\n#include <cstdio>\n'
                                  'int main() { std::printf("%zu %zu %zu", sizeof(sd_ctx_params_t), '
                                  'sizeof(sd_img_gen_params_t), sizeof(sd_vid_gen_params_t)); }\n')
                binary = include / "layout"
                subprocess.run(["c++", "-std=c++17", str(source), "-o", str(binary)], check=True)
                self.assertEqual(expected, subprocess.check_output([str(binary)], text=True))

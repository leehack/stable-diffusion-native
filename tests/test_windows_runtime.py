import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import package_release
import validate_artifacts
import windows_runtime

SECTION_RVA = 0x1000
SECTION_OFFSET = 0x400
MACHINES = {'x64': 0x8664, 'arm64': 0xAA64}


def make_pe(*, arch='x64', linker=(14, 44), imports=None, delay_imports=None,
            exports=(), file_version=None):
    """Builds a PE32+ DLL whose one section holds only the requested tables."""
    section = bytearray()

    def add(data, align=8):
        section.extend(b'\0' * (-len(section) % align))
        rva = SECTION_RVA + len(section)
        section.extend(data)
        return rva

    def thunks(symbols):
        names = [add(struct.pack('<H', 0) + symbol.encode() + b'\0', 2) for symbol in symbols]
        return add(b''.join(struct.pack('<Q', rva) for rva in names) + bytes(8))

    directories = [(0, 0)] * 16
    descriptors = b''
    for dll, symbols in (imports or {}).items():
        table = thunks(symbols)
        descriptors += struct.pack('<IIIII', table, 0, 0, add(dll.encode() + b'\0', 2), table)
    if descriptors:
        directories[1] = (add(descriptors + bytes(20), 4), len(descriptors) + 20)
    descriptors = b''
    for dll, symbols in (delay_imports or {}).items():
        table = thunks(symbols)
        name = add(dll.encode() + b'\0', 2)
        descriptors += struct.pack('<IIIIIIII', 1, name, add(bytes(8)), table, table, 0, 0, 0)
    if descriptors:
        directories[13] = (add(descriptors + bytes(32), 4), len(descriptors) + 32)
    if exports:
        names = [add(symbol.encode() + b'\0', 2) for symbol in exports]
        functions = add(b''.join(struct.pack('<I', SECTION_RVA) for _ in exports), 4)
        name_table = add(b''.join(struct.pack('<I', rva) for rva in names), 4)
        ordinals = add(b''.join(struct.pack('<H', index) for index in range(len(exports))), 2)
        directory = struct.pack('<IIHHIIIIIII', 0, 0, 0, 0, names[0], 1, len(exports), len(exports),
                                functions, name_table, ordinals)
        directories[0] = (add(directory, 4), 40)
    if file_version:
        major, minor, build_number, revision = file_version
        blob = add(b'VS_VERSION_INFO\0' + struct.pack('<IIII', 0xFEEF04BD, 0x10000,
                                                      major << 16 | minor, build_number << 16 | revision), 4)
        root = SECTION_RVA + len(section) + (-len(section) % 4)
        level = struct.pack('<IIHHHH', 0, 0, 0, 0, 0, 1)
        tree = (level + struct.pack('<II', 16, 0x80000000 | 24)
                + level + struct.pack('<II', 1, 0x80000000 | 48)
                + level + struct.pack('<II', 0x409, 72)
                + struct.pack('<IIII', blob, 36, 0, 0))
        directories[2] = (add(tree, 4), len(tree))
        assert directories[2][0] == root

    header = bytearray(SECTION_OFFSET)
    header[0:2] = b'MZ'
    struct.pack_into('<I', header, 0x3C, 0x40)
    header[0x40:0x44] = b'PE\0\0'
    struct.pack_into('<HHIIIHH', header, 0x44, MACHINES[arch], 1, 0, 0, 0, 240, 0x2022)
    optional = 0x58
    struct.pack_into('<HBB', header, optional, 0x20B, *linker)
    struct.pack_into('<Q', header, optional + 24, 0x180000000)
    struct.pack_into('<I', header, optional + 108, 16)
    for index, (rva, size) in enumerate(directories):
        struct.pack_into('<II', header, optional + 112 + 8 * index, rva, size)
    struct.pack_into('<8sIIII', header, optional + 240, b'.rdata', len(section), SECTION_RVA,
                     len(section), SECTION_OFFSET)
    return bytes(header) + bytes(section)



RUNTIME_IMPORTS = {
    "msvcp140.dll": ["_Thrd_yield"],
    "msvcp140_codecvt_ids.dll": ["codecvt_id"],
    "vcruntime140.dll": ["memcpy"],
    "vcruntime140_1.dll": ["__CxxFrameHandler4"],
}
VERSION = (14, 44, 35211, 0)


def windows_libraries(*, vulkan=False):
    imports = {
        **RUNTIME_IMPORTS,
        "KERNEL32.dll": ["GetLastError"],
        "api-ms-win-crt-heap-l1-1-0.dll": ["free"],
    }
    if vulkan:
        imports["vulkan-1.dll"] = ["vkCreateInstance"]
    return {
        "stable-diffusion.dll": make_pe(imports=imports),
        **{
            name: make_pe(exports=symbols, file_version=VERSION)
            for name, symbols in RUNTIME_IMPORTS.items()
        },
    }


class WindowsRuntimeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.bin = Path(temp.name)
        self.target = validate_artifacts.TARGETS["windows-x64"]
        root = self.bin / self.target.name
        self.lib = root / "lib"
        self.lib.mkdir(parents=True)
        (root / "build-info.json").write_text(json.dumps({"target": self.target.name}))
        self.libraries = windows_libraries()
        (self.lib / self.target.library).write_bytes(self.libraries[self.target.library])

    def validate(self):
        with patch.multiple(validate_artifacts, BIN_ROOT=self.bin, HEADERS=[]):
            return validate_artifacts.validate(self.target)

    def test_validate_requires_all_four_app_local_dlls(self):
        problems = self.validate()
        self.assertEqual(len(problems), len(RUNTIME_IMPORTS))
        for name in RUNTIME_IMPORTS:
            self.assertTrue(any(f"imports {name}," in problem for problem in problems), name)
        for name, payload in self.libraries.items():
            (self.lib / name).write_bytes(payload)
        self.assertEqual(self.validate(), [])
        (self.lib / "msvcp140_codecvt_ids.dll").unlink()
        self.assertEqual(len(self.validate()), 1)


    def test_stage_follows_transitive_imports_from_the_desktop_redist(self):
        redist = self.bin / "redist"
        desktop = redist / "x64/Microsoft.VC144.CRT"
        desktop.mkdir(parents=True)
        (self.lib / self.target.library).write_bytes(make_pe(imports={"msvcp140.dll": ["_Thrd_yield"]}))
        for name, symbols in RUNTIME_IMPORTS.items():
            imports = {"vcruntime140.dll": ["memcpy"]} if name == "msvcp140.dll" else {}
            (desktop / name).write_bytes(make_pe(imports=imports, exports=symbols, file_version=VERSION))
        with patch.dict("os.environ", {"VCToolsRedistDir": str(redist)}):
            self.assertEqual(windows_runtime.vc_redist_dir(), redist)
        staged = windows_runtime.stage_vc_runtime(self.lib, "x64", redist, allow_vulkan=False)
        self.assertEqual(staged, ["msvcp140.dll", "vcruntime140.dll"])
        self.assertEqual(self.validate(), [])

    def test_staging_has_no_onecore_or_debug_fallback(self):
        redist = self.bin / "redist"
        for path in ("debug_nonredist/x64/Microsoft.VC144.CRT", "onecore/x64/Microsoft.VC144.CRT"):
            directory = redist / path
            directory.mkdir(parents=True)
            (directory / "msvcp140.dll").write_bytes(self.libraries["msvcp140.dll"])
        with self.assertRaisesRegex(RuntimeError, "imports msvcp140.dll"):
            windows_runtime.stage_vc_runtime(self.lib, "x64", redist, allow_vulkan=False)
        self.assertFalse((self.lib / "msvcp140.dll").exists())

    def test_archive_packaging_checks_closure_and_lists_runtime_libraries(self):
        dist = self.bin / "dist"
        root = self.lib.parent
        (root / "build-info.json").write_text(json.dumps({"accelerators": ["cpu"]}))
        (root / "symbols").mkdir()
        (root / "symbols" / self.target.library).write_bytes(self.libraries[self.target.library])
        with patch.multiple(package_release, BIN_ROOT=self.bin, HEADERS=[]), patch.object(
            sys, "argv", ["package_release.py", "--tag", "v0.4.0-1", "--dist", str(dist)]
        ):
            with self.assertRaisesRegex(windows_runtime.PeError, "imports msvcp140_codecvt_ids.dll"):
                package_release.main()
            for name, payload in self.libraries.items():
                (self.lib / name).write_bytes(payload)
            (self.lib / "stable-diffusion.lib").write_bytes(b"import library")
            package_release.main()
        manifest = json.loads((dist / "manifest.json").read_text())
        runtime = next(item for item in manifest["artifacts"] if item["kind"] == "runtime")
        self.assertEqual(runtime["library"], self.target.library)
        self.assertEqual(runtime["libraries"], sorted(self.libraries))

    def test_validation_rejects_disallowed_bundled_dlls(self):
        for name in ("ucrtbase.dll", "kernel32.dll", "vulkan-1.dll", "vcruntime140d.dll",
                     "msvcp140d.dll", "ucrtbased.dll", "libomp140.x86_64.dll"):
            with self.subTest(name=name):
                path = self.lib / name
                path.write_bytes(make_pe())
                try:
                    self.assertTrue(any(problem.startswith(name + ":") for problem in self.validate()))
                finally:
                    path.unlink()

    def test_loader_exception_is_only_for_vulkan(self):
        for name, payload in windows_libraries(vulkan=True).items():
            (self.lib / name).write_bytes(payload)
        self.assertIn("imports vulkan-1.dll", self.validate()[0])
        images = windows_runtime.read_bundle(self.lib)
        self.assertEqual(windows_runtime.bundle_errors(images, "x64", allow_vulkan=True), [])

    def test_rejects_mismatched_arch_exports_and_runtime_version(self):
        for name, payload in self.libraries.items():
            (self.lib / name).write_bytes(payload)
        vcruntime = self.lib / "vcruntime140.dll"
        vcruntime.write_bytes(make_pe(arch="arm64", file_version=(14, 40, 1, 0)))
        problems = self.validate()
        for expected in ("arm64 image", "does not export memcpy", "different redistributables"):
            self.assertTrue(any(expected in problem for problem in problems), problems)
        vcruntime.write_bytes(self.libraries["vcruntime140.dll"])
        (self.lib / self.target.library).write_bytes(make_pe(linker=(14, 51), imports=RUNTIME_IMPORTS))
        self.assertIn("linked by MSVC 14.51, newer than", self.validate()[0])

    def test_delay_loads_are_checked(self):
        (self.lib / self.target.library).write_bytes(make_pe(
            delay_imports={"MSVCP140_CODECVT_IDS.dll": ["codecvt_id"]}
        ))
        image = windows_runtime.read_pe(self.lib / self.target.library)
        self.assertEqual(image.imports, {})
        self.assertEqual(image.delay_imports, {"msvcp140_codecvt_ids.dll": frozenset({"codecvt_id"})})
        self.assertIn("imports msvcp140_codecvt_ids.dll", self.validate()[0])

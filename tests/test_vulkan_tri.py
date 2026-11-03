"""Compiled triangle zero-store, source staging and packaged-byte regressions."""
from __future__ import annotations

from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from vulkan_tri import GENERATED_MODULE, build_record, validate_artifact, validate_module, validate_artifact_bytes

FIXTURE = ROOT / 'tests/fixtures/vulkan/tri_f16.spv'
GGML = ROOT / 'third_party/stable-diffusion.cpp/ggml'
TRI = 'src/ggml-vulkan/vulkan-shaders/tri.comp'


def mutate_constant(value: int) -> bytes:
    data = bytearray(FIXTURE.read_bytes())
    at = 20
    while at < len(data):
        word = struct.unpack_from('<I', data, at)[0]
        if word & 65535 == 43:
            type_id, result_id = struct.unpack_from('<II', data, at + 4)
            if result_id == 466:
                struct.pack_into('<I', data, at + 12, value)
                return bytes(data)
        at += (word >> 16) * 4
    raise AssertionError('zero constant not found')


class VulkanTriangleTest(unittest.TestCase):
    def test_positive_zero_fixture(self):
        validate_module(FIXTURE.read_bytes())

    def test_nan_and_finite_nonzero_and_negative_zero_are_rejected(self):
        for value in [0xfffffe00, 0x3f800000, 0x80000000]:
            with self.subTest(value=hex(value)), self.assertRaisesRegex(ValueError, 'positive zero'):
                validate_module(mutate_constant(value))

    def test_missing_conversion_use_is_rejected(self):
        data = bytearray(FIXTURE.read_bytes())
        at = 20
        while at < len(data):
            word = struct.unpack_from('<I', data, at)[0]
            if word & 65535 == 115 and struct.unpack_from('<I', data, at + 12)[0] == 466:
                struct.pack_into('<I', data, at + 12, 447)
                break
            at += (word >> 16) * 4
        with self.assertRaisesRegex(ValueError, 'output branches'):
            validate_module(bytes(data))

    def test_malformed_or_truncated_module_is_rejected(self):
        for data in [b'', FIXTURE.read_bytes()[:-4], b'bad!' + FIXTURE.read_bytes()[4:]]:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                validate_module(data)

    def test_compiled_module_must_be_unique_in_packaged_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / GENERATED_MODULE).parent.mkdir(parents=True)
            shutil.copy2(FIXTURE, root / GENERATED_MODULE)
            record = build_record(root)
            library = root / 'runtime'
            library.write_bytes(b'prefix' + FIXTURE.read_bytes() + b'suffix')
            validate_artifact(library, record)
            for data in [b'none', FIXTURE.read_bytes() * 2, mutate_constant(0xfffffe00)]:
                library.write_bytes(data)
                with self.assertRaises(ValueError):
                    validate_artifact(library, record)
            library.write_bytes(FIXTURE.read_bytes())
            record['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'digest'):
                validate_artifact(library, record)

    def test_only_configured_generated_module_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(FIXTURE, root / 'tri_f16.spv')
            with self.assertRaisesRegex(ValueError, 'generated module is missing'):
                build_record(root)
            (root / GENERATED_MODULE).parent.mkdir(parents=True)
            shutil.copy2(FIXTURE, root / GENERATED_MODULE)
            build_record(root)

    def test_present_bad_metadata_is_not_legacy(self):
        data = FIXTURE.read_bytes()
        validate_artifact_bytes(data, None, allow_legacy=True)
        for record in [{}, {'schemaVersion': 2}, {'schemaVersion': 1, 'sha256': '0' * 64, 'spirv_hex': data.hex()}]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_artifact_bytes(data, record, allow_legacy=True)

    def test_legacy_still_rejects_bad_or_ambiguous_actual_module(self):
        for data in [b'none', mutate_constant(0xfffffe00), mutate_constant(0x3f800000),
                     FIXTURE.read_bytes() + mutate_constant(0xfffffe00)]:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                validate_artifact_bytes(data, None, allow_legacy=True)

    def test_changed_layout_requires_review(self):
        data = bytearray(FIXTURE.read_bytes())
        # Generator ID does not change the flow proof; layout admission remains explicit.
        struct.pack_into('<I', data, 8, 0xabcdef)
        with self.assertRaisesRegex(ValueError, 'unrecognized module layout'):
            validate_module(bytes(data))
        with self.assertRaisesRegex(ValueError, 'unrecognized'):
            validate_artifact_bytes(bytes(data), None, allow_legacy=True)


if __name__ == '__main__':
    unittest.main()

"""The real build, package and replay entry points enforce the byte gate."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import build
import package_release
import validate_artifacts
from vulkan_tri import GENERATED_MODULE
import test_prepared_release as prepared_tests

FIXTURE = ROOT / 'tests/fixtures/vulkan/tri_f16.spv'
TARGET = build.TARGETS['windows-x64-vulkan']


def module_record(data):
    return {'schemaVersion': 1, 'sha256': hashlib.sha256(data).hexdigest(), 'spirv_hex': data.hex()}


def bad_module():
    data = bytearray(FIXTURE.read_bytes())
    struct.pack_into('<I', data, 2028, 0xfffffe00)
    return bytes(data)


class VulkanEntryPointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / 'bin'
        self.target = self.bin / TARGET.name
        (self.target / 'lib').mkdir(parents=True)
        self.library = self.target / 'lib' / TARGET.library
        self.library.write_bytes(FIXTURE.read_bytes())
        (self.target / 'symbols').mkdir()
        shutil.copy2(self.library, self.target / 'symbols' / TARGET.library)
        self.info = {'target': TARGET.name, 'upstream': {}, 'accelerators': ['cpu', 'vulkan'],
                     'vulkanTriF16': module_record(FIXTURE.read_bytes())}
        self.write_info()

    def write_info(self):
        (self.target / 'build-info.json').write_text(json.dumps(self.info))

    def package(self):
        with patch.object(package_release, 'BIN_ROOT', self.bin), \
             patch.object(package_release, 'TARGETS', {TARGET.name: TARGET}), \
             patch.object(package_release, 'HEADERS', []), \
             patch.object(package_release, 'LICENSES', {}), \
             patch.object(package_release, 'upstream_info', return_value={}), \
             patch.object(sys, 'argv', ['package', '--tag', 'v0.3.0-2', '--dist', str(self.root / 'dist')]):
            package_release.main()

    def test_package_requires_metadata_and_correct_actual_bytes(self):
        self.package()
        del self.info['vulkanTriF16']
        self.write_info()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.package()
        self.info['vulkanTriF16'] = module_record(FIXTURE.read_bytes())
        self.write_info()
        self.library.write_bytes(bad_module())
        with self.assertRaisesRegex(ValueError, 'positive zero'):
            self.package()

    def test_archive_writer_rejects_path_aliases(self):
        with self.assertRaisesRegex(ValueError, 'noncanonical'):
            package_release.write_archive(self.root / 'alias.tar.gz', {
                'lib/' + TARGET.library: self.library,
                'lib/./' + TARGET.library: self.library,
            })

    def test_artifact_validator_requires_metadata_and_correct_actual_bytes(self):
        with patch.object(validate_artifacts, 'BIN_ROOT', self.bin), \
             patch.object(validate_artifacts, 'HEADERS', []), \
             patch.object(validate_artifacts, 'api_symbols', return_value=[]), \
             patch.object(validate_artifacts, 'exported_symbols', return_value=set()):
            self.assertEqual(validate_artifacts.validate(TARGET), [])
            self.library.write_bytes(bad_module())
            self.assertTrue(any('positive zero' in e for e in validate_artifacts.validate(TARGET)))
            self.library.write_bytes(FIXTURE.read_bytes())
            del self.info['vulkanTriF16']
            self.write_info()
            self.assertTrue(any('provenance' in e for e in validate_artifacts.validate(TARGET)))

    def test_real_build_rejects_bad_generated_module(self):
        build_root = self.root / 'build'
        source = self.root / 'source.dll'
        source.write_bytes(FIXTURE.read_bytes())
        generated = build_root / TARGET.name / GENERATED_MODULE
        generated.parent.mkdir(parents=True)
        generated.write_bytes(FIXTURE.read_bytes())
        with patch.object(build, 'BUILD_ROOT', build_root), patch.object(build, 'BIN_ROOT', self.bin), \
             patch.object(build, 'HEADER', FIXTURE), patch.object(build, 'HEADERS', []), \
             patch.object(build, 'configure_args', return_value=[]), patch.object(build, 'run'), \
             patch.object(build, 'check_wrapper_definitions'), patch.object(build, 'find_library', return_value=source), \
             patch.object(build, 'strip', side_effect=lambda t, s, d: shutil.copy2(s, d)), \
             patch.object(build, 'upstream_info', return_value={}):
            build.build(TARGET, 1)
            generated.write_bytes(bad_module())
            with self.assertRaisesRegex(ValueError, 'positive zero'):
                build.build(TARGET, 1)


class VulkanReplayTest(unittest.TestCase):
    setUp = prepared_tests.DistributionTest.setUp
    add_artifact = prepared_tests.DistributionTest.add_artifact
    write_manifest = prepared_tests.DistributionTest.write_manifest
    verify = prepared_tests.DistributionTest.verify

    def rewrite(self, mutate):
        artifact = next(a for a in self.manifest['artifacts']
                        if a['target'] == TARGET.name and a['kind'] == 'runtime')
        path = self.dist / artifact['file']
        with tarfile.open(path, 'r:gz') as archive:
            contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        info = json.loads(contents['build-info.json'])
        mutate(info, contents)
        contents['build-info.json'] = json.dumps(info).encode()
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        self.write_manifest()

    def test_legacy_auxiliary_absence_proves_same_actual_module(self):
        self.rewrite(lambda info, contents: info.pop('vulkanTriF16'))
        self.verify()
        self.rewrite(lambda info, contents: contents.update({'lib/' + TARGET.library: bad_module()}))
        with self.assertRaisesRegex(ValueError, 'positive zero'):
            self.verify()

    def test_canonical_directory_entries_remain_supported(self):
        artifact = next(a for a in self.manifest['artifacts']
                        if a['target'] == TARGET.name and a['kind'] == 'runtime')
        path = self.dist / artifact['file']
        with tarfile.open(path, 'r:gz') as archive:
            contents = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()]
        with tarfile.open(path, 'w:gz') as archive:
            directory = tarfile.TarInfo('lib/')
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            for name, data in contents:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        self.write_manifest()
        self.verify()

    def test_replay_rejects_extraction_aliases_even_with_updated_hashes(self):
        artifact = next(a for a in self.manifest['artifacts']
                        if a['target'] == TARGET.name and a['kind'] == 'runtime')
        path = self.dist / artifact['file']
        original = path.read_bytes()
        aliases = ['lib/./' + TARGET.library, 'lib/../lib/' + TARGET.library,
                   'lib\\' + TARGET.library, 'lib/' + TARGET.library.upper(),
                   'lib/' + TARGET.library + '.', 'lib//' + TARGET.library]
        for alias in aliases:
            with self.subTest(alias=alias):
                path.write_bytes(original)
                with tarfile.open(path, 'r:gz') as archive:
                    contents = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()]
                with tarfile.open(path, 'w:gz') as archive:
                    for name, data in contents + [(alias, bad_module())]:
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
                # A real ordinary extraction overwrites the good DLL for the dot
                # alias, demonstrating why choosing an exact member is insufficient.
                if '/./' in alias:
                    extracted = Path(self.tmp.name) / 'alias-extracted'
                    with tarfile.open(path, 'r:gz') as archive:
                        archive.extractall(extracted, filter='data')
                    self.assertEqual((extracted / 'lib' / TARGET.library).read_bytes(), bad_module())
                artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'noncanonical|normalized'):
                    self.verify()
        path.write_bytes(original)

    def test_replay_rejects_archive_links_and_duplicate_members(self):
        artifact = next(a for a in self.manifest['artifacts']
                        if a['target'] == TARGET.name and a['kind'] == 'runtime')
        path = self.dist / artifact['file']
        original = path.read_bytes()
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.REGTYPE]:
            with self.subTest(kind=kind):
                path.write_bytes(original)
                with tarfile.open(path, 'r:gz') as archive:
                    contents = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()]
                with tarfile.open(path, 'w:gz') as archive:
                    for name, data in contents:
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
                    member = tarfile.TarInfo('lib/' + TARGET.library if kind == tarfile.REGTYPE else 'lib/alias')
                    member.type, member.linkname = kind, TARGET.library
                    member.size = len(bad_module()) if kind == tarfile.REGTYPE else 0
                    archive.addfile(member, io.BytesIO(bad_module()) if member.size else None)
                artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'nonregular|normalized'):
                    self.verify()
        path.write_bytes(original)

    def test_replay_rejects_bad_metadata_and_updated_hashes(self):
        for bad in [None, {}, {'schemaVersion': 2}, module_record(bad_module())]:
            with self.subTest(metadata=bad):
                self.rewrite(lambda info, contents, bad=bad: info.update(vulkanTriF16=bad))
                with self.assertRaises(ValueError):
                    self.verify()

    def test_replay_rejects_missing_ambiguous_and_unknown_actual_modules(self):
        for payload in [b'none', bad_module(), FIXTURE.read_bytes() * 2,
                        FIXTURE.read_bytes() + bad_module()]:
            with self.subTest(size=len(payload)):
                self.rewrite(lambda info, contents, payload=payload:
                             contents.update({'lib/' + TARGET.library: payload}))
                with self.assertRaises(ValueError):
                    self.verify()

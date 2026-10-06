from __future__ import annotations

import plistlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import apple_privacy_manifest as validator  # noqa: E402
import apple_xcframework as packager  # noqa: E402

VALIDATOR = ROOT / "tools" / "apple_privacy_manifest.py"
SHIPPED_MANIFEST = packager.PRIVACY_MANIFEST.read_bytes()
XCFRAMEWORK = packager.XCFRAMEWORK
FRAMEWORK = f"{packager.FRAMEWORK}.framework"
IOS = "ios-arm64"
MACOS = "macos-arm64_x86_64"
IOS_BINARY = f"{XCFRAMEWORK}/{IOS}/{FRAMEWORK}/{packager.FRAMEWORK}"
MACOS_BINARY = f"{XCFRAMEWORK}/{MACOS}/{FRAMEWORK}/Versions/A/{packager.FRAMEWORK}"
IOS_MANIFEST = f"{XCFRAMEWORK}/{IOS}/{FRAMEWORK}/PrivacyInfo.xcprivacy"
MACOS_MANIFEST = (
    f"{XCFRAMEWORK}/{MACOS}/{FRAMEWORK}/Versions/A/Resources/PrivacyInfo.xcprivacy"
)


def manifest(**overrides: object) -> bytes:
    content = plistlib.loads(SHIPPED_MANIFEST)
    content.update(overrides)
    return plistlib.dumps(content)


def accessed(category: object, *reasons: object) -> dict[str, object]:
    return {
        "NSPrivacyAccessedAPIType": category,
        "NSPrivacyAccessedAPITypeReasons": list(reasons),
    }


def write_archive(directory: str, members: dict[str, bytes | None]) -> Path:
    """Write a two-slice XCFramework zip; a None value drops that member."""
    content: dict[str, bytes | None] = {
        f"{XCFRAMEWORK}/Info.plist": plistlib.dumps(
            {
                "AvailableLibraries": [
                    {
                        "LibraryIdentifier": IOS,
                        "LibraryPath": FRAMEWORK,
                        "BinaryPath": f"{FRAMEWORK}/{packager.FRAMEWORK}",
                    },
                    {
                        "LibraryIdentifier": MACOS,
                        "LibraryPath": FRAMEWORK,
                        "BinaryPath": f"{FRAMEWORK}/Versions/A/{packager.FRAMEWORK}",
                    },
                ]
            }
        ),
        IOS_BINARY: b"ios",
        IOS_MANIFEST: SHIPPED_MANIFEST,
        MACOS_BINARY: b"macos",
        MACOS_MANIFEST: SHIPPED_MANIFEST,
    }
    content.update(members)
    archive = Path(directory) / "xcframework.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in content.items():
            if data is not None:
                bundle.writestr(name, data)
    return archive


def quiet_run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


class ShippedManifestTests(unittest.TestCase):
    def test_declares_only_the_audited_file_timestamp_reasons(self) -> None:
        errors, declared = validator.validate_manifest(SHIPPED_MANIFEST)

        self.assertEqual([], errors)
        self.assertEqual({validator.FILE_TIMESTAMP}, declared)
        self.assertEqual(
            [accessed(validator.FILE_TIMESTAMP, "C617.1", "3B52.1")],
            plistlib.loads(SHIPPED_MANIFEST)["NSPrivacyAccessedAPITypes"],
        )


class PackagerTests(unittest.TestCase):
    def test_packaged_slices_carry_the_manifest_where_the_validator_expects_it(
        self,
    ) -> None:
        ios, _, macos = packager.SLICES
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "libstable-diffusion.dylib"
            source.write_bytes(b"binary")
            header = work / packager.HEADERS[0].name
            header.write_bytes(b"")
            xcframework = work / XCFRAMEWORK
            with patch.object(packager, "run"), patch.object(
                packager, "HEADERS", (header,)
            ), patch.object(packager, "slice_min_version", return_value="1.0"):
                frameworks = {
                    slice_: packager.make_framework(slice_, source, xcframework, "0.0.0")
                    for slice_ in (ios, macos)
                }
            with (xcframework / "Info.plist").open("wb") as file:
                plistlib.dump(
                    {
                        "AvailableLibraries": [
                            {
                                "LibraryIdentifier": slice_.identifier,
                                "LibraryPath": framework.name,
                            }
                            for slice_, framework in frameworks.items()
                        ]
                    },
                    file,
                )
            archive = work / "xcframework.zip"
            packager.write_zip(xcframework, archive)

            self.assertEqual(
                SHIPPED_MANIFEST,
                (frameworks[ios] / "PrivacyInfo.xcprivacy").read_bytes(),
            )
            self.assertEqual(
                SHIPPED_MANIFEST,
                (
                    frameworks[macos] / "Versions/A/Resources/PrivacyInfo.xcprivacy"
                ).read_bytes(),
            )
            self.assertFalse((frameworks[macos] / "PrivacyInfo.xcprivacy").exists())
            self.assertEqual([], validator.validate_archive(archive))


class ValidateArchiveTests(unittest.TestCase):
    def errors(self, members: dict[str, bytes | None]) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            return validator.validate_archive(write_archive(directory, members))

    def test_accepts_a_manifest_in_every_slice(self) -> None:
        self.assertEqual([], self.errors({}))

    def test_rejects_a_slice_without_a_manifest(self) -> None:
        self.assertEqual(
            [f"{IOS}: missing privacy manifest at {IOS_MANIFEST}"],
            self.errors({IOS_MANIFEST: None}),
        )

    def test_rejects_a_manifest_outside_the_versioned_resources_directory(self) -> None:
        stray = f"{XCFRAMEWORK}/{MACOS}/{FRAMEWORK}/PrivacyInfo.xcprivacy"

        self.assertEqual(
            [
                f"{MACOS}: unexpected privacy manifest at {stray}",
                f"{MACOS}: missing privacy manifest at {MACOS_MANIFEST}",
            ],
            self.errors({MACOS_MANIFEST: None, stray: SHIPPED_MANIFEST}),
        )

    def test_rejects_a_manifest_that_is_not_a_property_list(self) -> None:
        errors = self.errors({MACOS_MANIFEST: b"<plist><dict>"})

        self.assertEqual(1, len(errors))
        self.assertIn(f"{MACOS_MANIFEST} is not a valid property list", errors[0])

    def test_rejects_inaccurate_declarations(self) -> None:
        cases = {
            "NSPrivacyTracking must be false": manifest(NSPrivacyTracking=True),
            "NSPrivacyCollectedDataTypes must be an empty array": manifest(
                NSPrivacyCollectedDataTypes=[{"NSPrivacyCollectedDataType": "x"}]
            ),
            "NSPrivacyAccessedAPITypes must be an array": manifest(
                NSPrivacyAccessedAPITypes="C617.1"
            ),
            "unknown required-reason API category: 'Timestamp'": manifest(
                NSPrivacyAccessedAPITypes=[accessed("Timestamp", "C617.1")]
            ),
            f"{validator.FILE_TIMESTAMP} declares unapproved reason '35F9.1'": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP, "35F9.1")]
            ),
            f"{validator.FILE_TIMESTAMP} must declare at least one reason": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP)]
            ),
            "unknown required-reason API category: ['Timestamp']": manifest(
                NSPrivacyAccessedAPITypes=[accessed(["Timestamp"], "C617.1")]
            ),
            f"{validator.FILE_TIMESTAMP} declares unapproved reason ['C617.1']": manifest(
                NSPrivacyAccessedAPITypes=[accessed(validator.FILE_TIMESTAMP, ["C617.1"])]
            ),
        }
        for expected, data in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    [f"{IOS}: {IOS_MANIFEST} {expected}"],
                    self.errors({IOS_MANIFEST: data}),
                )

    def test_import_audit_requires_declarations_to_match_each_binary(self) -> None:
        references = {
            b"ios": ({"_stat", "_fstat", "_mach_absolute_time"}, set()),
            b"macos": ({"_abort"}, set()),
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            validator,
            "binary_api_references",
            side_effect=lambda binary: references[binary.read_bytes()],
        ):
            errors = validator.validate_archive(
                write_archive(directory, {}), audit_imports=True
            )

        self.assertEqual(
            [
                f"{IOS}: uses {validator.SYSTEM_BOOT_TIME} via _mach_absolute_time "
                "but the manifest does not declare it",
                f"{MACOS}: manifest declares {validator.FILE_TIMESTAMP} but the "
                "binary uses none of its APIs",
            ],
            errors,
        )

    def test_cli_fails_on_an_unreadable_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "xcframework.zip"
            archive.write_bytes(b"not a zip")
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(1, result.returncode)
        self.assertIn("error:", result.stderr)


@unittest.skipUnless(
    sys.platform == "darwin",
    "needs the Apple toolchain to build and inspect Mach-O binaries",
)
class MachOImportAuditTests(unittest.TestCase):
    PLAIN = "int plain(void) { return 1; }\n"
    REQUIRED_REASON = (
        "#import <Foundation/Foundation.h>\n"
        "#include <mach/mach_time.h>\n"
        "#include <sys/stat.h>\n"
        "double required_reason(int fd) {\n"
        "  struct stat info;\n"
        '  fstat(fd, &info);\n'
        '  stat("/", &info);\n'
        "  return [[NSProcessInfo processInfo] systemUptime] + mach_absolute_time();\n"
        "}\n"
    )

    def fat_dylib(self, work: Path, sources: dict[str, str]) -> bytes:
        thin = []
        for arch, source in sources.items():
            source_file = work / f"{arch}.m"
            source_file.write_text(source, encoding="utf-8")
            thin.append(work / f"{arch}.dylib")
            subprocess.run(
                [
                    "xcrun", "clang", "-dynamiclib", "-arch", arch,
                    "-framework", "Foundation", str(source_file), "-o", str(thin[-1]),
                ],
                check=True,
                capture_output=True,
            )
        fat = work / "fat.dylib"
        subprocess.run(
            ["xcrun", "lipo", "-create", *map(str, thin), "-output", str(fat)],
            check=True,
            capture_output=True,
        )
        return fat.read_bytes()

    def test_audits_every_architecture_of_a_fat_slice(self) -> None:
        architectures = ("arm64", "x86_64")
        for importing in architectures:
            with self.subTest(importing=importing), tempfile.TemporaryDirectory() as directory:
                binary = self.fat_dylib(
                    Path(directory),
                    {
                        arch: self.REQUIRED_REASON if arch == importing else self.PLAIN
                        for arch in architectures
                    },
                )
                archive = write_archive(
                    directory,
                    {IOS_BINARY: binary, MACOS_BINARY: binary},
                )

                self.assertEqual(
                    [
                        f"{identifier}: uses {validator.SYSTEM_BOOT_TIME} via "
                        "_mach_absolute_time, systemUptime but the manifest does "
                        "not declare it"
                        for identifier in (IOS, MACOS)
                    ],
                    validator.validate_archive(archive, audit_imports=True),
                )


@unittest.skipUnless(sys.platform == "darwin", "validate_zip unpacks with ditto")
class XcframeworkValidationTests(unittest.TestCase):
    def test_validate_reports_slices_without_a_manifest(self) -> None:
        header = f"{XCFRAMEWORK}/{IOS}/{FRAMEWORK}/Headers/{packager.HEADERS[0].name}"
        with tempfile.TemporaryDirectory() as directory:
            archive = write_archive(
                directory, {IOS_MANIFEST: None, MACOS_MANIFEST: None, header: b""}
            )
            with patch.object(packager, "run", quiet_run):
                problems = packager.validate_zip(archive)

        for identifier, member in ((IOS, IOS_MANIFEST), (MACOS, MACOS_MANIFEST)):
            self.assertIn(f"{identifier}: missing privacy manifest at {member}", problems)


class WorkflowTests(unittest.TestCase):
    AUDIT = "python3 tools/apple_privacy_manifest.py --audit-imports"

    def test_release_audits_the_packaged_xcframework_before_upload(self) -> None:
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text()

        self.assertTrue(
            -1
            < workflow.find("python3 tools/package_release.py")
            < workflow.find(self.AUDIT)
            < workflow.find("softprops/action-gh-release")
        )

    def test_pull_requests_run_the_mach_o_tests_and_the_audit_on_macos(self) -> None:
        workflow = (ROOT / ".github/workflows/validate.yml").read_text()

        self.assertIn("os: [ubuntu-latest, macos-latest]", workflow)
        self.assertTrue(
            -1
            < workflow.find("python3 tools/apple_xcframework.py build")
            < workflow.find(self.AUDIT)
        )


class RequiredCategoryTests(unittest.TestCase):
    def test_maps_symbol_variants_classes_and_selectors_to_categories(self) -> None:
        self.assertEqual(
            {
                validator.FILE_TIMESTAMP: ["_fstat", "_stat"],
                validator.SYSTEM_BOOT_TIME: ["systemUptime"],
                validator.DISK_SPACE: ["_statfs"],
                validator.USER_DEFAULTS: ["_OBJC_CLASS_$_NSUserDefaults"],
            },
            validator.required_categories(
                {
                    "_stat$INODE64",
                    "_fstat",
                    "_statfs$INODE64",
                    "_OBJC_CLASS_$_NSUserDefaults",
                    "_clock_gettime",
                },
                {"systemUptime", "fileURLWithPath:"},
            ),
        )


if __name__ == "__main__":
    unittest.main()

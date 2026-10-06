#!/usr/bin/env python3
"""Packages the Apple runtime slices as a SwiftPM binary-target XCFramework.

`build` wraps the `bin/` Apple slices (see `SLICES`) into
`stable_diffusion.xcframework` and writes a zip whose SHA-256 is the SwiftPM
`binaryTarget` checksum. `validate` checks a zip as a consumer would see it,
including each slice's privacy manifest (see `apple_privacy_manifest.py`);
`consumer` links it into `tests/swiftpm_consumer`, runs the macOS probe and
builds the iOS device and simulator slices.

Each framework's Info.plist minimum OS is read from the slice's
`LC_BUILD_VERSION`, so the two cannot disagree; App Store validation rejects
frameworks whose `MinimumOSVersion` differs from the binary's.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import apple_privacy_manifest
from build import (
    APPLE_IOS_MIN,
    APPLE_MACOS_MIN,
    BIN_ROOT,
    BUILD_ROOT,
    HEADERS,
    REPO_ROOT,
    TARGETS,
)
from sd_api import api_symbols
from validate_artifacts import (
    APPLE_STATIC_DESTRUCTOR_IMPORT,
    APPLE_SYSTEM_DEPENDENCIES,
    imports_symbol,
)

FRAMEWORK = "stable_diffusion"
XCFRAMEWORK = f"{FRAMEWORK}.xcframework"
BUNDLE_IDENTIFIER = "dev.leehack.stable-diffusion-native"
ARCHIVE_PREFIX = "stable-diffusion-native-apple-xcframework"
PRIVACY_MANIFEST = REPO_ROOT / "tools" / "apple" / apple_privacy_manifest.MANIFEST_NAME
# Fixed so that rebuilding the same slices yields the same SwiftPM checksum.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
SYSTEM_DEPENDENCY = re.compile(f"^({APPLE_SYSTEM_DEPENDENCIES})$")


@dataclass(frozen=True)
class Slice:
    identifier: str
    targets: tuple[str, ...]
    platform: str
    build_platform: str
    plist_min_key: str
    min_version: str
    archs: tuple[str, ...]
    macos_layout: bool = False

    @property
    def binary_path(self) -> str:
        if self.macos_layout:
            return f"{FRAMEWORK}.framework/Versions/A/{FRAMEWORK}"
        return f"{FRAMEWORK}.framework/{FRAMEWORK}"

    @property
    def install_name(self) -> str:
        return f"@rpath/{self.binary_path}"

    @property
    def info_plist_path(self) -> str:
        if self.macos_layout:
            return f"{FRAMEWORK}.framework/Versions/A/Resources/Info.plist"
        return f"{FRAMEWORK}.framework/Info.plist"

    @property
    def privacy_manifest_path(self) -> str:
        # Versioned bundles keep resources out of the framework root; a
        # manifest there is unsealed content that fails code signing.
        if self.macos_layout:
            return f"{FRAMEWORK}.framework/Versions/A/Resources/{PRIVACY_MANIFEST.name}"
        return f"{FRAMEWORK}.framework/{PRIVACY_MANIFEST.name}"


SLICES = (
    Slice("ios-arm64", ("ios-arm64",), "iPhoneOS", "IOS",
          "MinimumOSVersion", APPLE_IOS_MIN, ("arm64",)),
    Slice("ios-arm64_x86_64-simulator", ("ios-arm64-sim", "ios-x64-sim"), "iPhoneSimulator",
          "IOSSIMULATOR", "MinimumOSVersion", APPLE_IOS_MIN, ("arm64", "x86_64")),
    Slice("macos-arm64_x86_64", ("macos-arm64", "macos-x64"), "MacOSX", "MACOS",
          "LSMinimumSystemVersion", APPLE_MACOS_MIN, ("arm64", "x86_64"),
          macos_layout=True),
)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def archive_name(tag: str) -> str:
    return f"{ARCHIVE_PREFIX}-{tag}.zip"


def bundle_version(tag: str) -> str:
    match = re.match(r"^v(\d+\.\d+\.\d+)", tag)
    if not match:
        raise ValueError(f"tag {tag!r} has no MAJOR.MINOR.PATCH version")
    return match.group(1)


def module_map() -> str:
    headers = "".join(f'  header "{header.name}"\n' for header in HEADERS)
    return f"framework module {FRAMEWORK} {{\n{headers}  export *\n}}\n"


def build_versions(binary: Path) -> dict[str, set[str]]:
    """Maps each LC_BUILD_VERSION platform in `binary` to its minos values."""
    versions: dict[str, set[str]] = {}
    platform = None
    for line in output(["xcrun", "vtool", "-show-build", str(binary)]).splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "platform":
            platform = fields[1]
        elif len(fields) == 2 and fields[0] == "minos" and platform:
            versions.setdefault(platform, set()).add(fields[1])
    return versions


def slice_min_version(slice_: Slice, binary: Path) -> str:
    versions = build_versions(binary)
    if set(versions) != {slice_.build_platform} or len(versions[slice_.build_platform]) != 1:
        fail(f"{binary} has build versions {versions}, expected one "
             f"{slice_.build_platform} minos")
    return next(iter(versions[slice_.build_platform]))


def info_plist(slice_: Slice, version: str, min_version: str) -> dict[str, object]:
    return {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": FRAMEWORK,
        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": FRAMEWORK,
        "CFBundlePackageType": "FMWK",
        "CFBundleShortVersionString": version,
        "CFBundleSupportedPlatforms": [slice_.platform],
        "CFBundleVersion": version,
        slice_.plist_min_key: min_version,
    }


def slice_binary(slice_: Slice, work_dir: Path) -> Path:
    libraries = []
    for name in slice_.targets:
        library = BIN_ROOT / name / "lib" / TARGETS[name].library
        if not library.is_file():
            fail(f"missing {library}; build {name} first")
        libraries.append(library)
    if len(libraries) == 1:
        return libraries[0]
    universal = work_dir / slice_.identifier / TARGETS[slice_.targets[0]].library
    universal.parent.mkdir(parents=True, exist_ok=True)
    run(["xcrun", "lipo", "-create", *map(str, libraries), "-output", str(universal)])
    return universal


def make_framework(slice_: Slice, source: Path, root: Path, version: str) -> Path:
    framework = root / slice_.identifier / f"{FRAMEWORK}.framework"
    content = framework / "Versions" / "A" if slice_.macos_layout else framework
    (content / "Headers").mkdir(parents=True)
    (content / "Modules").mkdir()
    binary = content / FRAMEWORK
    shutil.copy2(source, binary)
    binary.chmod(0o755)
    run(["install_name_tool", "-id", slice_.install_name, str(binary)])
    for header in HEADERS:
        shutil.copy2(header, content / "Headers" / header.name)
    (content / "Modules" / "module.modulemap").write_text(module_map())

    plist = root / slice_.identifier / slice_.info_plist_path
    plist.parent.mkdir(parents=True, exist_ok=True)
    min_version = slice_min_version(slice_, binary)
    with plist.open("wb") as handle:
        plistlib.dump(info_plist(slice_, version, min_version), handle)
    shutil.copy2(PRIVACY_MANIFEST, root / slice_.identifier / slice_.privacy_manifest_path)

    if slice_.macos_layout:
        (framework / "Versions" / "Current").symlink_to("A")
        for name in ("Headers", "Modules", "Resources", FRAMEWORK):
            (framework / name).symlink_to(Path("Versions") / "Current" / name)
    return framework


def write_zip(xcframework: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(xcframework.rglob("*")):
            name = path.relative_to(xcframework.parent).as_posix()
            if path.is_symlink():
                info = zipfile.ZipInfo(name, ZIP_DATE_TIME)
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, os.readlink(path))
            elif path.is_dir():
                info = zipfile.ZipInfo(f"{name}/", ZIP_DATE_TIME)
                info.create_system = 3
                info.external_attr = (0o40755 << 16) | 0x10
                archive.writestr(info, b"")
            else:
                info = zipfile.ZipInfo(name, ZIP_DATE_TIME)
                info.create_system = 3
                mode = 0o755 if os.access(path, os.X_OK) else 0o644
                info.external_attr = (0o100000 | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())


def build_xcframework(tag: str, dist: Path) -> Path:
    version = bundle_version(tag)
    work_dir = BUILD_ROOT / "apple-xcframework"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    frameworks = [
        make_framework(slice_, slice_binary(slice_, work_dir / "universal"),
                       work_dir / "frameworks", version)
        for slice_ in SLICES
    ]
    xcframework = work_dir / XCFRAMEWORK
    command = ["xcodebuild", "-create-xcframework"]
    for framework in frameworks:
        command += ["-framework", str(framework)]
    run([*command, "-output", str(xcframework)])
    # xcodebuild lists the slices in a varying order.
    plist_path = xcframework / "Info.plist"
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    plist["AvailableLibraries"].sort(key=lambda entry: entry["LibraryIdentifier"])
    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)

    dist.mkdir(parents=True, exist_ok=True)
    dest = dist / archive_name(tag)
    write_zip(xcframework, dest)
    problems = validate_zip(dest)
    if problems:
        dest.unlink()
        fail("built XCFramework is invalid:\n  - " + "\n  - ".join(problems))
    return dest


def validate_slice(slice_: Slice, root: Path, entry: dict, expected_symbols: set[str]) -> list[str]:
    problems: list[str] = []
    where = slice_.identifier
    if entry.get("BinaryPath") != slice_.binary_path:
        problems.append(f"{where}: BinaryPath is {entry.get('BinaryPath')!r}")
    if sorted(entry.get("SupportedArchitectures", [])) != sorted(slice_.archs):
        problems.append(f"{where}: architectures {entry.get('SupportedArchitectures')}")

    binary = root / slice_.binary_path
    plist_path = root / slice_.info_plist_path
    if not binary.is_file() or not plist_path.is_file():
        return problems + [f"{where}: missing binary or Info.plist"]
    if not (root / f"{FRAMEWORK}.framework" / "Modules" / "module.modulemap").is_file():
        problems.append(f"{where}: missing module map")
    for header in HEADERS:
        if not (root / f"{FRAMEWORK}.framework" / "Headers" / header.name).is_file():
            problems.append(f"{where}: missing {header.name}")
    if slice_.macos_layout:
        current = root / f"{FRAMEWORK}.framework" / "Versions" / "Current"
        if not current.is_symlink() or os.readlink(current) != "A":
            problems.append(f"{where}: Versions/Current must link to A")

    versions = build_versions(binary)
    if versions != {slice_.build_platform: {slice_.min_version}}:
        problems.append(f"{where}: build versions {versions}, expected "
                        f"{slice_.build_platform} minos {slice_.min_version}")

    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    if plist.get("CFBundleExecutable") != FRAMEWORK or plist.get("CFBundlePackageType") != "FMWK":
        problems.append(f"{where}: Info.plist does not describe the {FRAMEWORK} framework")
    if plist.get(slice_.plist_min_key) != slice_.min_version:
        problems.append(f"{where}: {slice_.plist_min_key} is "
                        f"{plist.get(slice_.plist_min_key)!r}, binary minos is "
                        f"{slice_.min_version}")

    for arch in slice_.archs:
        install_name = output(["otool", "-arch", arch, "-D", str(binary)]).splitlines()[-1]
        if install_name.strip() != slice_.install_name:
            problems.append(f"{where} {arch}: install name {install_name.strip()}")
        lines = output(["otool", "-arch", arch, "-L", str(binary)]).splitlines()[1:]
        for dep in (line.split(" (")[0].strip() for line in lines):
            if dep != slice_.install_name and not SYSTEM_DEPENDENCY.match(dep):
                problems.append(f"{where} {arch}: unexpected dependency {dep}")
        if imports_symbol(binary, APPLE_STATIC_DESTRUCTOR_IMPORT, arch):
            problems.append(f"{where} {arch}: imports {APPLE_STATIC_DESTRUCTOR_IMPORT}; "
                            "static destructors bypass exit teardown")
        names = output(["nm", "-gUj", "-arch", arch, str(binary)]).split()
        exported = {n[1:] if n.startswith("_") else n for n in names}
        if exported != expected_symbols:
            missing = sorted(expected_symbols - exported)
            leaked = sorted(exported - expected_symbols)
            problems.append(f"{where} {arch}: exports differ from SD_API; "
                            f"missing {missing[:5]}, extra {leaked[:5]}")
    return problems


def validate_zip(archive: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        # Python's zipfile drops symlinks; ditto restores them as SwiftPM does.
        run(["ditto", "-x", "-k", str(archive), tmp])
        tops = sorted(p.name for p in Path(tmp).iterdir())
        if tops != [XCFRAMEWORK]:
            return [f"zip must contain only {XCFRAMEWORK}, found {tops}"]
        xcframework = Path(tmp) / XCFRAMEWORK
        with (xcframework / "Info.plist").open("rb") as handle:
            libraries = {
                entry["LibraryIdentifier"]: entry
                for entry in plistlib.load(handle)["AvailableLibraries"]
            }
        problems: list[str] = []
        expected = {s.identifier for s in SLICES}
        if set(libraries) != expected:
            problems.append(f"slices {sorted(libraries)}, expected {sorted(expected)}")
        shipped = xcframework / "ios-arm64" / f"{FRAMEWORK}.framework" / "Headers"
        symbols = set(api_symbols(*(shipped / header.name for header in HEADERS
                                    if (shipped / header.name).is_file())))
        for slice_ in SLICES:
            if slice_.identifier in libraries:
                problems += validate_slice(slice_, xcframework / slice_.identifier,
                                           libraries[slice_.identifier], symbols)
        return problems + apple_privacy_manifest.validate_archive(archive, audit_imports=True)


def check_consumer(archive: Path) -> None:
    package = REPO_ROOT / "tests" / "swiftpm_consumer"
    work_dir = BUILD_ROOT / "swiftpm-consumer"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copy2(archive, package / "stable_diffusion.zip")
    scratch = work_dir / "spm"
    run(["swift", "build", "-c", "release", "--package-path", str(package),
         "--scratch-path", str(scratch), "--product", "Probe"])
    run([str(scratch / "release" / "Probe")])
    for destination in ("generic/platform=iOS", "generic/platform=iOS Simulator"):
        subprocess.run(
            ["xcodebuild", "build", "-quiet", "-scheme", "Companion",
             "-destination", destination, "-derivedDataPath", str(work_dir / "xcode"),
             "CODE_SIGNING_ALLOWED=NO"],
            cwd=package, check=True,
        )
        print(f"linked Companion for {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Build the XCFramework zip.")
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--dist", type=Path, default=Path("dist"))
    validate_parser = sub.add_parser("validate", help="Validate an XCFramework zip.")
    validate_parser.add_argument("archive", type=Path)
    consumer_parser = sub.add_parser("consumer", help="Link a zip into a SwiftPM consumer.")
    consumer_parser.add_argument("archive", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        print(f"wrote {build_xcframework(args.tag, args.dist)}")
        return
    if args.command == "consumer":
        check_consumer(args.archive.resolve())
        return
    problems = validate_zip(args.archive)
    print(f"{args.archive.name}: {'ok' if not problems else 'FAILED'}")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Packages `bin/<target>/` payloads into release archives.

Writes `dist/stable-diffusion-native-runtime-<target>-<tag>.tar.gz` (stripped
runtime, header, licenses, build info), a matching `-symbols` archive with the
unstripped library for crash symbolication, `manifest.json` and `SHA256SUMS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

from build import BIN_ROOT, REPO_ROOT, TARGETS, UPSTREAM_DIR, upstream_commit

PACKAGE = "stable-diffusion-native"
TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+(-[1-9]\d*)?$")
LICENSES = {
    "LICENSE": REPO_ROOT / "LICENSE",
    "licenses/stable-diffusion.cpp.LICENSE": UPSTREAM_DIR / "LICENSE",
    "licenses/ggml.LICENSE": UPSTREAM_DIR / "ggml" / "LICENSE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def write_archive(dest: Path, entries: dict[str, Path]) -> None:
    with tarfile.open(dest, "w:gz") as archive:
        for name in sorted(entries):
            archive.add(entries[name], arcname=name, filter=_reset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args()
    if not TAG_PATTERN.match(args.tag):
        print(f"error: tag {args.tag!r} must look like v0.1.0 or v0.1.0-2", file=sys.stderr)
        sys.exit(1)

    if args.dist.exists():
        shutil.rmtree(args.dist)
    args.dist.mkdir(parents=True)

    artifacts = []
    for name, target in TARGETS.items():
        root = BIN_ROOT / name
        if not root.is_dir():
            continue
        runtime = args.dist / f"{PACKAGE}-runtime-{name}-{args.tag}.tar.gz"
        entries = {
            f"lib/{p.name}": p for p in sorted((root / "lib").iterdir())
        }
        entries["include/stable-diffusion.h"] = root / "include" / "stable-diffusion.h"
        entries["build-info.json"] = root / "build-info.json"
        entries.update(LICENSES)
        write_archive(runtime, entries)

        symbols = args.dist / f"{PACKAGE}-symbols-{name}-{args.tag}.tar.gz"
        write_archive(symbols, {f"symbols/{target.library}": root / "symbols" / target.library})

        info = json.loads((root / "build-info.json").read_text())
        for kind, path in (("runtime", runtime), ("symbols", symbols)):
            artifacts.append({
                "target": name,
                "kind": kind,
                "file": path.name,
                "sha256": sha256(path),
                "size": path.stat().st_size,
                "library": target.library,
                "accelerators": info["accelerators"],
            })

    if not artifacts:
        print("error: nothing to package under bin/", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "schemaVersion": 1,
        "package": PACKAGE,
        "tag": args.tag,
        "upstream": {
            "repository": "leejet/stable-diffusion.cpp",
            "commit": upstream_commit(),
        },
        "artifacts": artifacts,
    }
    (args.dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sums = "".join(f"{a['sha256']}  {a['file']}\n" for a in artifacts)
    (args.dist / "SHA256SUMS").write_text(sums)
    for artifact in artifacts:
        print(f"{artifact['file']}  {artifact['size'] >> 20} MB")


if __name__ == "__main__":
    main()

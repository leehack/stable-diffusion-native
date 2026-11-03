#!/usr/bin/env python3
"""Verify a prepared release, or publish its verified bytes from the release job."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build import TARGETS
from package_release import PACKAGE, TAG_PATTERN, sha256

REPOSITORY = "leehack/stable-diffusion-native"
WORKFLOW = ".github/workflows/native_release.yml"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def api(path: str, *, method: str = "GET", data: dict | None = None,
        missing_ok: bool = False) -> dict | None:
    command = ["gh", "api", f"repos/{REPOSITORY}/{path}", "--method", method]
    if data is not None:
        command += ["--input", "-"]
    result = subprocess.run(command, input=json.dumps(data) if data is not None else None,
                            text=True, capture_output=True)
    if result.returncode:
        if missing_ok and "(HTTP 404)" in result.stderr:
            return None
        raise ValueError(f"GitHub API failed: {method} {path.split('?')[0]}")
    return json.loads(result.stdout)


def validate_inputs(tag: str, source: str, digest: str) -> None:
    require(TAG_PATTERN.fullmatch(tag) is not None, "invalid release tag")
    require(SHA_PATTERN.fullmatch(source) is not None, "expected source must be a full SHA")
    require(DIGEST_PATTERN.fullmatch(digest) is not None, "expected manifest SHA256 required")


def provenance(run_id: str, source: str, tag: str) -> tuple[dict, str]:
    require(re.fullmatch(r"[1-9][0-9]*", run_id) is not None, "invalid prepared run ID")
    run = api(f"actions/runs/{run_id}")
    workflow = api("actions/workflows/native_release.yml")
    require(run["id"] == int(run_id) and run["workflow_id"] == workflow["id"]
            and run["path"] == WORKFLOW, "prepared run is not the release workflow")
    require(run["event"] == "workflow_dispatch" and run["status"] == "completed"
            and run["conclusion"] == "success", "prepared run did not succeed")
    require(run["repository"]["full_name"] == REPOSITORY
            and run["head_repository"]["full_name"] == REPOSITORY
            and run["head_branch"] == "main" and run["head_sha"] == source,
            "prepared source repository, branch or SHA mismatch")
    comparison = api(f"compare/{source}...main")
    require(comparison["status"] in ("ahead", "identical"),
            "prepared source is not on main")
    upstream = api(f"contents/third_party/stable-diffusion.cpp?ref={source}")
    require(upstream["submodule_git_url"] ==
            "https://github.com/leejet/stable-diffusion.cpp.git",
            "unexpected upstream repository")
    artifacts = []
    page = 1
    while True:
        batch = api(f"actions/runs/{run_id}/artifacts?per_page=100&page={page}")["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    matches = [a for a in artifacts if a["name"] == f"{PACKAGE}-{tag}"]
    require(len(matches) == 1, "prepared artifact missing or ambiguous")
    artifact = matches[0]
    require(not artifact["expired"] and datetime.fromisoformat(
        artifact["expires_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc),
        "prepared artifact expired")
    origin = artifact["workflow_run"]
    require(origin["id"] == int(run_id) and origin["head_sha"] == source
            and origin["repository_id"] == run["repository"]["id"]
            and origin["head_repository_id"] == run["repository"]["id"],
            "artifact provenance mismatch")
    return artifact, upstream["sha"]


def unpack(archive: Path, dest: Path) -> None:
    require(not dest.exists(), "destination must not exist")
    with zipfile.ZipFile(archive) as bundle:
        names = set()
        for entry in bundle.infolist():
            name = entry.filename
            mode = entry.external_attr >> 16
            require(name not in names and name not in ("", ".", "..")
                    and "/" not in name and "\\" not in name
                    and not entry.is_dir() and not stat.S_ISLNK(mode)
                    and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                    "artifact contains duplicate, nested or non-regular files")
            names.add(name)
        dest.mkdir(parents=True)
        for entry in bundle.infolist():
            with bundle.open(entry) as src, (dest / entry.filename).open("xb") as out:
                shutil.copyfileobj(src, out)


def verify_dist(dist: Path, tag: str, source: str, digest: str, upstream: str) -> dict:
    validate_inputs(tag, source, digest)
    require(dist.is_dir() and not dist.is_symlink(), "invalid distribution directory")
    files = list(dist.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in files),
            "distribution contains non-regular files")
    require(sha256(dist / "manifest.json") == digest, "manifest SHA256 mismatch")
    manifest = json.loads((dist / "manifest.json").read_text())
    require(manifest["schemaVersion"] == 1 and manifest["package"] == PACKAGE
            and manifest["tag"] == tag, "manifest package, schema or tag mismatch")
    require(manifest["upstream"]["repository"] == "leejet/stable-diffusion.cpp"
            and manifest["upstream"]["commit"] == upstream, "manifest upstream mismatch")
    # Legacy prepared runs bind the owner SHA through run provenance; new packages
    # record it too. The manifest digest is always an explicit replay input.
    if "source" in manifest:
        require(manifest["source"] == {"repository": REPOSITORY, "commit": source},
                "manifest owner source mismatch")
    expected = {(name, kind): f"{PACKAGE}-{kind}-{name}-{tag}.tar.gz"
                for name in TARGETS for kind in ("runtime", "symbols")}
    expected[("apple", "xcframework")] = f"{PACKAGE}-apple-xcframework-{tag}.zip"
    seen = set()
    sums = {}
    for artifact in manifest["artifacts"]:
        key = (artifact["target"], artifact["kind"])
        require(key in expected and key not in seen and artifact["file"] == expected[key],
                "manifest archive inventory mismatch")
        seen.add(key)
        path = dist / expected[key]
        require(type(artifact["size"]) is int and path.stat().st_size == artifact["size"]
                and DIGEST_PATTERN.fullmatch(artifact["sha256"]) is not None
                and sha256(path) == artifact["sha256"], "archive size or SHA256 mismatch")
        sums[path.name] = artifact["sha256"]
        if key[1] == "runtime":
            with tarfile.open(path, "r:gz") as archive:
                entries = [e for e in archive.getmembers() if e.name == "build-info.json"]
                require(len(entries) == 1 and entries[0].isfile(), "missing build information")
                info = json.load(archive.extractfile(entries[0]))
                require(info["target"] == key[0] and info["upstream"] == manifest["upstream"],
                        "archive build source mismatch")
    require(seen == set(expected), "manifest is missing release targets")
    require({p.name for p in files} == set(sums) | {"manifest.json", "SHA256SUMS"},
            "distribution has extra or missing files")
    lines = (dist / "SHA256SUMS").read_text().splitlines()
    require(len(lines) == len(sums) and set(lines) == {f"{v}  {k}" for k, v in sums.items()},
            "SHA256SUMS does not match manifest")
    return manifest


def check_absent(tag: str) -> None:
    require(api(f"releases/tags/{tag}", missing_ok=True) is None,
            "release already exists; refusing to overwrite")
    require(api(f"git/ref/tags/{tag}", missing_ok=True) is None,
            "tag already exists; refusing to reuse")


def replay(args: argparse.Namespace) -> None:
    validate_inputs(args.tag, args.source, args.manifest_sha256)
    artifact, upstream = provenance(args.run_id, args.source, args.tag)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "artifact.zip"
        with archive.open("wb") as output:
            subprocess.run(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/"
                            f"{artifact['id']}/zip"], stdout=output, check=True)
        require(artifact.get("digest") == "sha256:" + sha256(archive),
                "downloaded artifact digest mismatch")
        unpack(archive, args.dist)
    verify_dist(args.dist, args.tag, args.source, args.manifest_sha256, upstream)
    check_absent(args.tag)
    print(f"Verified {args.tag}: source {args.source}, manifest {args.manifest_sha256}")


def publish(args: argparse.Namespace) -> None:
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY
            and os.environ.get("GITHUB_WORKFLOW") == "Native Build & Release"
            and os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("GITHUB_REF") == "refs/heads/main",
            "publish only through release workflow on main")
    validate_inputs(args.tag, args.source, args.manifest_sha256)
    upstream = api(f"contents/third_party/stable-diffusion.cpp?ref={args.source}")["sha"]
    verify_dist(args.dist, args.tag, args.source, args.manifest_sha256, upstream)
    check_absent(args.tag)
    # Creating the ref is atomic: another publisher winning the race fails this
    # request, before release creation or any asset upload. Existing tags are held.
    api("git/refs", method="POST", data={"ref": f"refs/tags/{args.tag}", "sha": args.source})
    command = ["gh", "release", "create", args.tag, "--repo", REPOSITORY,
               "--verify-tag", "--title", args.tag, "--notes",
               f"Prepared source: {args.source}\nManifest SHA256: {args.manifest_sha256}"]
    if args.prerelease:
        command.append("--prerelease")
    command.extend(str(p) for p in sorted(args.dist.iterdir()))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"tag {args.tag} was created, but publication did not finish; "
            "inspect the partial tag/release state before recovery; no overwrite attempted"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "verify", "publish"):
        command = sub.add_parser(name)
        command.add_argument("--tag", required=True)
        command.add_argument("--source", required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--dist", type=Path, default=Path("dist"))
        if name == "replay":
            command.add_argument("--run-id", required=True)
        if name == "verify":
            command.add_argument("--upstream", required=True)
        if name == "publish":
            command.add_argument("--prerelease", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "replay":
            replay(args)
        elif args.command == "publish":
            publish(args)
        else:
            verify_dist(args.dist, args.tag, args.source, args.manifest_sha256, args.upstream)
            check_absent(args.tag)
    except (ValueError, KeyError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

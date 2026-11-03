"""Prepared archives stay bound to the qualified run, source, tag and bytes."""

import argparse
import copy
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import prepared_release as release

TAG = "v0.3.0-2"
SOURCE = "1" * 40
UPSTREAM = "2" * 40


class DistributionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dist = Path(self.tmp.name) / "dist"
        self.dist.mkdir()
        self.manifest = {"schemaVersion": 1, "package": release.PACKAGE, "tag": TAG,
                         "upstream": {"repository": "leejet/stable-diffusion.cpp",
                                      "commit": UPSTREAM, "patches": []}, "artifacts": []}
        for target in release.TARGETS:
            for kind in ("runtime", "symbols"):
                name = f"{release.PACKAGE}-{kind}-{target}-{TAG}.tar.gz"
                path = self.dist / name
                with tarfile.open(path, "w:gz") as archive:
                    info = json.dumps({"target": target, "upstream": self.manifest["upstream"]}).encode()
                    entry = tarfile.TarInfo("build-info.json")
                    entry.size = len(info)
                    archive.addfile(entry, io.BytesIO(info))
                self.add_artifact(target, kind, path)
        path = self.dist / f"{release.PACKAGE}-apple-xcframework-{TAG}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("example", b"xcframework")
        self.add_artifact("apple", "xcframework", path)
        self.write_manifest()

    def add_artifact(self, target, kind, path):
        self.manifest["artifacts"].append({"target": target, "kind": kind,
            "file": path.name, "sha256": release.sha256(path), "size": path.stat().st_size})

    def write_manifest(self):
        (self.dist / "manifest.json").write_text(json.dumps(self.manifest))
        (self.dist / "SHA256SUMS").write_text("".join(
            f"{a['sha256']}  {a['file']}\n" for a in self.manifest["artifacts"]))
        self.digest = release.sha256(self.dist / "manifest.json")

    def verify(self):
        return release.verify_dist(self.dist, TAG, SOURCE, self.digest, UPSTREAM)

    def test_legacy_and_owner_source_manifests(self):
        self.verify()
        self.manifest["source"] = {"repository": release.REPOSITORY, "commit": SOURCE}
        self.write_manifest()
        self.verify()
        self.manifest["source"]["commit"] = "3" * 40
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "owner source"):
            self.verify()

    def test_mutations_fail_closed(self):
        for mutation in ("tag", "upstream", "duplicate", "missing", "extra", "hash", "sums", "digest", "symlink"):
            with self.subTest(mutation=mutation):
                backup = copy.deepcopy(self.manifest)
                if mutation == "tag": self.manifest["tag"] = "v0.3.0-1"
                if mutation == "upstream": self.manifest["upstream"]["commit"] = "3" * 40
                if mutation == "duplicate": self.manifest["artifacts"].append(self.manifest["artifacts"][0])
                if mutation == "missing": self.manifest["artifacts"].pop()
                if mutation == "hash": self.manifest["artifacts"][0]["sha256"] = "0" * 64
                self.write_manifest()
                extra = self.dist / "extra"
                if mutation == "extra": extra.write_text("extra")
                if mutation == "symlink": extra.symlink_to(self.dist / self.manifest["artifacts"][0]["file"])
                if mutation == "sums": (self.dist / "SHA256SUMS").write_text("")
                if mutation == "digest": self.digest = "0" * 64
                with self.assertRaises(ValueError): self.verify()
                if extra.exists(): extra.unlink()
                self.manifest = backup
                self.write_manifest()

    def test_archive_build_source_is_checked_even_with_updated_hashes(self):
        artifact = self.manifest["artifacts"][0]
        path = self.dist / artifact["file"]
        with tarfile.open(path, "w:gz") as archive:
            data = json.dumps({"target": artifact["target"], "upstream": {"commit": "3" * 40}}).encode()
            info = tarfile.TarInfo("build-info.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        artifact.update(sha256=release.sha256(path), size=path.stat().st_size)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "build source"):
            self.verify()

    def test_download_envelope_rejects_unsafe_members(self):
        for name, mode in (("../escape", 0), ("nested/file", 0), ("file", stat.S_IFLNK | 0o777),
                           ("file", stat.S_IFIFO | 0o600), ("file\\escape", 0)):
            with self.subTest(name=name, mode=mode):
                bundle = Path(self.tmp.name) / "bundle.zip"
                with zipfile.ZipFile(bundle, "w") as archive:
                    info = zipfile.ZipInfo(name)
                    info.external_attr = mode << 16
                    archive.writestr(info, "payload")
                with self.assertRaises(ValueError):
                    release.unpack(bundle, Path(self.tmp.name) / "unpacked")

    def test_publish_verifies_again_and_uses_atomic_ref_creation(self):
        args = argparse.Namespace(tag=TAG, source=SOURCE, manifest_sha256=self.digest,
                                  dist=self.dist, prerelease=True)
        env = {"GITHUB_REPOSITORY": release.REPOSITORY, "GITHUB_WORKFLOW": "Native Build & Release",
               "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}
        calls = []
        def fake_api(path, **kwargs):
            calls.append((path, kwargs))
            if path.startswith("contents/"): return {"sha": UPSTREAM}
            return None
        with patch.dict(os.environ, env), patch.object(release, "api", side_effect=fake_api), \
                patch.object(release.subprocess, "run") as run:
            release.publish(args)
            self.assertEqual(calls[-1], ("git/refs", {"method": "POST", "data": {
                "ref": f"refs/tags/{TAG}", "sha": SOURCE}}))
            command = run.call_args.args[0]
            self.assertIn("--verify-tag", command)
            self.assertNotIn("--clobber", command)
            self.assertEqual(len([x for x in command if str(self.dist) in x]), 27)
        (self.dist / "SHA256SUMS").write_text("corrupt")
        with patch.dict(os.environ, env), patch.object(release, "api", side_effect=fake_api), \
                patch.object(release.subprocess, "run") as run:
            with self.assertRaises(ValueError): release.publish(args)
            run.assert_not_called()

    def test_replay_preserves_every_prepared_byte(self):
        bundle = Path(self.tmp.name) / "prepared.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            for path in self.dist.iterdir():
                archive.write(path, path.name)
        dest = Path(self.tmp.name) / "replayed"
        args = argparse.Namespace(tag=TAG, source=SOURCE, manifest_sha256=self.digest,
                                  run_id="123", dist=dest)
        artifact = {"id": 789, "digest": "sha256:" + release.sha256(bundle)}
        def download(command, **kwargs):
            self.assertEqual(command[-1], f"repos/{release.REPOSITORY}/actions/artifacts/789/zip")
            kwargs["stdout"].write(bundle.read_bytes())
        with patch.object(release, "provenance", return_value=(artifact, UPSTREAM)), \
                patch.object(release.subprocess, "run", side_effect=download), \
                patch.object(release, "check_absent"):
            release.replay(args)
        self.assertEqual({p.name: p.read_bytes() for p in self.dist.iterdir()},
                         {p.name: p.read_bytes() for p in dest.iterdir()})

    def test_partial_publication_reports_the_created_tag(self):
        args = argparse.Namespace(tag=TAG, source=SOURCE, manifest_sha256=self.digest,
                                  dist=self.dist, prerelease=True)
        env = {"GITHUB_REPOSITORY": release.REPOSITORY, "GITHUB_WORKFLOW": "Native Build & Release",
               "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}
        with patch.dict(os.environ, env), patch.object(release, "api", return_value={"sha": UPSTREAM}), \
                patch.object(release, "check_absent"), \
                patch.object(release.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "gh")):
            with self.assertRaisesRegex(ValueError, "tag .* was created.*partial tag/release"):
                release.publish(args)

    def test_tag_race_never_creates_or_uploads_release(self):
        args = argparse.Namespace(tag=TAG, source=SOURCE, manifest_sha256=self.digest,
                                  dist=self.dist, prerelease=True)
        env = {"GITHUB_REPOSITORY": release.REPOSITORY, "GITHUB_WORKFLOW": "Native Build & Release",
               "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}
        def fake_api(path, **kwargs):
            if path.startswith("contents/"): return {"sha": UPSTREAM}
            if path == "git/refs": raise ValueError("tag won by other publisher")
            return None
        with patch.dict(os.environ, env), patch.object(release, "api", side_effect=fake_api), \
                patch.object(release.subprocess, "run") as run:
            with self.assertRaises(ValueError): release.publish(args)
            run.assert_not_called()


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.run = {"id": 123, "workflow_id": 456, "path": release.WORKFLOW,
                    "event": "workflow_dispatch", "status": "completed", "conclusion": "success",
                    "repository": {"full_name": release.REPOSITORY, "id": 1},
                    "head_repository": {"full_name": release.REPOSITORY},
                    "head_branch": "main", "head_sha": SOURCE}
        self.artifact = {"id": 789, "name": f"{release.PACKAGE}-{TAG}", "expired": False,
                         "expires_at": "2999-01-01T00:00:00Z", "workflow_run": {
                             "id": 123, "head_sha": SOURCE, "repository_id": 1, "head_repository_id": 1}}
        self.artifacts = [self.artifact]
        self.status = "ahead"

    def api(self, path, **kwargs):
        if path == "actions/runs/123": return self.run
        if path == "actions/workflows/native_release.yml": return {"id": 456}
        if path.startswith("compare/"): return {"status": self.status}
        if path.startswith("contents/"): return {"sha": UPSTREAM, "submodule_git_url":
            "https://github.com/leejet/stable-diffusion.cpp.git"}
        if "/artifacts?" in path: return {"artifacts": self.artifacts}
        raise AssertionError(path)

    def test_exact_successful_owner_run(self):
        with patch.object(release, "api", side_effect=self.api):
            self.assertEqual(release.provenance("123", SOURCE, TAG), (self.artifact, UPSTREAM))

    def test_untrusted_runs_fail_closed(self):
        for key, value in (("workflow_id", 999), ("path", "other.yml"), ("event", "pull_request"),
                           ("status", "in_progress"), ("conclusion", "failure"), ("head_sha", "3" * 40),
                           ("head_branch", "feature"), ("head_repository", {"full_name": "fork/repo"})):
            with self.subTest(key=key):
                original = self.run[key]
                self.run[key] = value
                with patch.object(release, "api", side_effect=self.api), self.assertRaises(ValueError):
                    release.provenance("123", SOURCE, TAG)
                self.run[key] = original

    def test_expired_missing_ambiguous_and_foreign_artifacts(self):
        for mutation in ("expired", "expiry", "missing", "duplicate", "foreign", "not-main"):
            with self.subTest(mutation=mutation):
                saved = copy.deepcopy(self.artifact)
                if mutation == "expired": self.artifact["expired"] = True
                if mutation == "expiry": self.artifact["expires_at"] = "2000-01-01T00:00:00Z"
                if mutation == "missing": self.artifacts = []
                if mutation == "duplicate": self.artifacts = [self.artifact, self.artifact]
                if mutation == "foreign": self.artifact["workflow_run"]["head_repository_id"] = 2
                if mutation == "not-main": self.status = "diverged"
                with patch.object(release, "api", side_effect=self.api), self.assertRaises(ValueError):
                    release.provenance("123", SOURCE, TAG)
                self.artifact = saved
                self.artifacts = [self.artifact]
                self.status = "ahead"

    def test_existing_tag_or_release_blocks_publication(self):
        for results in (({},), (None, {})):
            with patch.object(release, "api", side_effect=results), self.assertRaises(ValueError):
                release.check_absent(TAG)


if __name__ == "__main__":
    unittest.main()

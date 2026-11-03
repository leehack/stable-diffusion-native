"""Keep preparation runs unable to enter the release-writing job."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ReleaseWorkflowTest(unittest.TestCase):
    def test_preparation_has_no_release_write_path(self):
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text()
        preparation, marker, release = workflow.partition("\n  release:\n")
        self.assertTrue(marker, "publication must remain a separate job")
        self.assertIn("permissions:\n  contents: read\n", preparation)
        self.assertNotIn("contents: write", preparation)
        self.assertNotIn("action-gh-release", preparation)
        self.assertIn("name: stable-diffusion-native-${{ inputs.tag }}", preparation)
        self.assertIn("retention-days: 14", preparation)
        job_header = release.split("    steps:", 1)[0]
        self.assertIn("    needs: verify\n", job_header)
        self.assertIn("inputs.publish && github.ref == 'refs/heads/main'", job_header)
        self.assertIn("      contents: write\n", job_header)
        self.assertNotIn("action-gh-release@", workflow)
        self.assertIn("tools/prepared_release.py publish", release)
        self.assertIn("tools/prepared_release.py replay", preparation)
        self.assertIn("if: ${{ !inputs.prepared_run_id }}", preparation)
        self.assertIn("group: native-release-${{ inputs.tag }}", preparation)
        self.assertIn("cancel-in-progress: false", preparation)

    def test_replay_publication_ignores_skipped_builds_but_requires_verification(self):
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text()
        release = workflow.split("\n  release:\n", 1)[1]
        expression = release.split("if: ${{", 1)[1].split("}}", 1)[0].strip()
        # Actions implicitly adds success() only when there is no status check.
        # Replay intentionally skips build/package, so that implicit gate is false.
        if not re.search(r"\b(always|cancelled|success|failure)\(", expression):
            expression = f"success() && ({expression})"
        for verified, cancelled, expected in (
            ("success", False, True),
            ("failure", False, False),
            ("cancelled", False, False),
            ("skipped", False, False),
            ("success", True, False),
        ):
            with self.subTest(verified=verified, cancelled=cancelled):
                condition = expression.replace("needs.verify.result", repr(verified))
                condition = condition.replace("inputs.publish", "True")
                condition = condition.replace("github.ref", repr("refs/heads/main"))
                condition = condition.replace("github.repository", repr("leehack/stable-diffusion-native"))
                condition = condition.replace("&&", " and ").replace("!", "not ")
                actual = eval(condition, {"__builtins__": {}}, {
                    "always": lambda: True,
                    "cancelled": lambda: cancelled,
                    "success": lambda: False,
                })
                self.assertEqual(actual, expected)

    def test_linux_runtime_builds_keep_the_qualified_ubuntu_baseline(self):
        # The floating label moved to Ubuntu 26.04 and a different software
        # Vulkan matrix path; qualification and release must use one baseline.
        for name in ("validate.yml", "native_release.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text()
            for target in ("linux-x64", "linux-x64-vulkan"):
                with self.subTest(workflow=name, target=target):
                    self.assertRegex(
                        workflow,
                        rf"target: {target}, runs_on: ubuntu-24\.04,",
                    )

    def test_publication_requires_explicit_opt_in(self):
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text()
        self.assertIn(
            "      publish:\n"
            "        description: Publish the prepared artifacts as a GitHub release\n"
            "        type: boolean\n"
            "        default: false\n",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()

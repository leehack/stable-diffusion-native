"""Keep preparation runs unable to enter the release-writing job."""

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
        self.assertIn("    needs: package\n", job_header)
        self.assertIn("    if: ${{ inputs.publish }}\n", job_header)
        self.assertIn("      contents: write\n", job_header)
        self.assertEqual(workflow.count("action-gh-release@"), 1)

    def test_normal_release_mode_remains_explicit_and_enabled(self):
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text()
        self.assertIn(
            "      publish:\n"
            "        description: Publish the prepared artifacts as a GitHub release\n"
            "        type: boolean\n"
            "        default: true\n",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()

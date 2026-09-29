import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build import TARGETS  # noqa: E402
from package_release import TAG_PATTERN  # noqa: E402
from sd_api import api_symbols  # noqa: E402

HEADER = """
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#endif
// SD_API void commented_out(void);
/* SD_API void block_commented(void); */
extern SD_API const char* sample_method_to_str[];
SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);
SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* params,
                           sd_image_t** images_out,
                           int* num_images_out);
static inline int not_exported(void) { return 0; }
"""


class ApiSymbolsTest(unittest.TestCase):
    def test_collects_functions_and_arrays_but_not_macros_or_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "stable-diffusion.h"
            header.write_text(HEADER)
            self.assertEqual(
                api_symbols(header),
                ["generate_image", "new_sd_ctx", "sample_method_to_str"],
            )


class ReleaseTagTest(unittest.TestCase):
    def test_accepts_semver_and_positive_rebuild_counters(self):
        for tag in ("v0.1.0", "v1.20.3", "v0.1.0-2"):
            self.assertRegex(tag, TAG_PATTERN)

    def test_rejects_unprefixed_zero_and_padded_counters(self):
        for tag in ("0.1.0", "v0.1", "v0.1.0-0", "v0.1.0-01", "latest"):
            self.assertNotRegex(tag, TAG_PATTERN)


class TargetsTest(unittest.TestCase):
    def test_native_hosts_declare_their_cpu(self):
        for target in TARGETS.values():
            if target.os in ("linux", "windows"):
                self.assertTrue(target.host_arch, target.name)


if __name__ == "__main__":
    unittest.main()

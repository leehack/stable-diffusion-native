import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import apple_xcframework  # noqa: E402
from build import HEADERS, TARGETS, WRAPPER_HEADER  # noqa: E402
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

    def test_merges_the_symbols_of_several_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            upstream = Path(tmp) / "stable-diffusion.h"
            upstream.write_text(HEADER)
            wrapper = Path(tmp) / "wrapper.h"
            wrapper.write_text("SD_API void sd_dart_added(void* token);\n")
            self.assertEqual(
                api_symbols(upstream, wrapper),
                ["generate_image", "new_sd_ctx", "sample_method_to_str", "sd_dart_added"],
            )

    def test_wrapper_header_adds_only_the_progress_routing_exports(self):
        self.assertEqual(
            api_symbols(WRAPPER_HEADER),
            ["sd_dart_clear_progress_callback", "sd_dart_set_progress_callback"],
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


class AppleXcframeworkTest(unittest.TestCase):
    def test_every_apple_target_lands_in_one_slice(self):
        packaged = [t for s in apple_xcframework.SLICES for t in s.targets]
        apple = [t.name for t in TARGETS.values() if t.os in ("ios", "macos")]
        self.assertCountEqual(packaged, apple)

    def test_module_map_exposes_every_shipped_header(self):
        module_map = apple_xcframework.module_map()
        for header in HEADERS:
            self.assertIn(f'header "{header.name}"', module_map)

    def test_bundle_version_drops_the_rebuild_counter(self):
        self.assertEqual(apple_xcframework.bundle_version("v0.2.0"), "0.2.0")
        self.assertEqual(apple_xcframework.bundle_version("v0.2.0-3"), "0.2.0")

    def test_info_plist_names_the_framework_and_its_minimum_os(self):
        ios, simulator, macos = apple_xcframework.SLICES
        plist = apple_xcframework.info_plist(ios, "0.2.0", "16.4")
        self.assertEqual(plist["CFBundleExecutable"], "stable_diffusion")
        self.assertEqual(plist["CFBundlePackageType"], "FMWK")
        self.assertEqual(plist["MinimumOSVersion"], "16.4")
        self.assertEqual(plist["CFBundleSupportedPlatforms"], ["iPhoneOS"])
        self.assertEqual(
            apple_xcframework.info_plist(simulator, "0.2.0", "16.4")
            ["CFBundleSupportedPlatforms"],
            ["iPhoneSimulator"],
        )
        macos_plist = apple_xcframework.info_plist(macos, "0.2.0", "13.3")
        self.assertEqual(macos_plist["LSMinimumSystemVersion"], "13.3")
        self.assertNotIn("MinimumOSVersion", macos_plist)

    def test_framework_binaries_use_rpath_install_names(self):
        ios, _, macos = apple_xcframework.SLICES
        self.assertEqual(ios.install_name,
                         "@rpath/stable_diffusion.framework/stable_diffusion")
        self.assertEqual(macos.install_name,
                         "@rpath/stable_diffusion.framework/Versions/A/stable_diffusion")


if __name__ == "__main__":
    unittest.main()

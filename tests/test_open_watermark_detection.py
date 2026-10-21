"""Actual encoded-payload regressions; no model weights or encoder dependency.

The immutable fixture was encoded by the unmodified invisible-watermark API.
Its provenance and hash are documented beside the file. Test variations only
change EXIF metadata, lossless orientation, or alpha; no guessed watermark is
injected into the test decoder.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unmarker_open_watermark_test", ROOT / "open_watermark_bridge.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
FIXTURE = ROOT / "tests" / "fixtures" / "open_watermark" / "sd1-dwtdct.png"
SD1 = "Stable Diffusion 1.x / 2.x"


class OpenWatermarkDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != (
                "964006325b0b0edc0bd33222a7fbbbfa1b0997654694412aeafbf1df6c8933bb"):
            raise AssertionError("The genuine upstream-encoded fixture was replaced")
        with Image.open(FIXTURE) as image:
            cls.positive = image.copy()
        rng = np.random.default_rng(1729)
        y, x = np.indices((512, 512))
        carrier_bgr = np.stack((
            120 + 24 * np.sin(x / 17.) + 17 * np.cos(y / 23.),
            126 + 21 * np.cos((x + y) / 27.),
            128 + 22 * np.sin(y / 19.) + 19 * np.cos(x / 31.)), axis=2)
        carrier_bgr = np.clip(carrier_bgr + rng.normal(0, 8, carrier_bgr.shape), 30, 224).astype(np.uint8)
        cls.negative = Image.fromarray(carrier_bgr[:, :, ::-1].copy())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unmarker-open-detection-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def save(self, image, name="input.png", orientation=None):
        options = {}
        if orientation is not None:
            exif = Image.Exif()
            exif[0x0112] = orientation
            options["exif"] = exif
        path = self.directory / name
        image.save(path, format="PNG", **options)
        return path

    def assert_sd1(self, result, basis, attempts=1):
        self.assertEqual(result["status"], "detected", result)
        self.assertFalse(result["synthid_supported"])
        self.assertEqual(result["scope"], "open_dwt_dct")
        self.assertEqual(result["basis"], basis)
        self.assertEqual(len(result["attempts"]), attempts)
        self.assertEqual(len(result["comparisons"]), 4)
        self.assertEqual([match["name"] for match in result["matches"]], [SD1])
        match = result["matches"][0]
        self.assertEqual((match["matched_bits"], match["total_bits"], match["threshold_bits"]), (129, 136, 126))
        json.dumps(result, allow_nan=False)

    def test_known_public_positive_decodes_without_orientation_or_models(self):
        result = bridge.scan_open_watermarks(FIXTURE)
        self.assert_sd1(result, "stored_rgb")
        self.assertEqual(result["exif_orientation"], 1)
        self.assertEqual(result["alpha_handling"], "opaque_rgb")

    def test_exif_only_changes_preserve_the_positive_stored_raster(self):
        for orientation in range(2, 9):
            with self.subTest(orientation=orientation):
                source = self.save(self.positive, orientation=orientation)
                with Image.open(source) as image:
                    self.assertTrue(np.array_equal(np.asarray(image), np.asarray(self.positive)))
                result = bridge.scan_open_watermarks(source)
                self.assert_sd1(result, "stored_rgb")
                self.assertEqual(result["exif_orientation"], orientation)

    def test_actual_exif_transform_can_recover_a_losslessly_rotated_raster(self):
        # These are the inverse raster transforms of EXIF 2 and 6, respectively.
        for orientation, transform in ((2, Image.Transpose.FLIP_LEFT_RIGHT), (6, Image.Transpose.ROTATE_90)):
            with self.subTest(orientation=orientation):
                source = self.save(self.positive.transpose(transform), orientation=orientation)
                with Image.open(source) as image:
                    self.assertTrue(np.array_equal(np.asarray(ImageOps.exif_transpose(image)), np.asarray(self.positive)))
                result = bridge.scan_open_watermarks(source)
                self.assert_sd1(result, "exif_oriented_rgb", attempts=2)
                self.assertEqual(result["attempts"][0]["status"], "not_detected")
                self.assertEqual(result["attempts"][0]["basis"], "stored_rgb")
                self.assertEqual(result["attempts"][1]["status"], "detected")

    def test_unmarked_carrier_stays_negative_with_every_valid_exif_orientation(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                result = bridge.scan_open_watermarks(self.save(self.negative, orientation=orientation))
                self.assertEqual(result["status"], "not_detected", result)
                self.assertEqual(result["matches"], [])
                self.assertEqual(result["basis"], "stored_rgb")
                self.assertEqual(len(result["attempts"]), 1 if orientation == 1 else 2)
                self.assertFalse(any(c["matched"] for a in result["attempts"] for c in a["comparisons"]))

    def test_identical_oriented_pixels_do_not_run_an_extra_decode(self):
        source = self.save(Image.new("RGB", (512, 512), (120, 120, 120)), orientation=6)
        result = bridge.scan_open_watermarks(source)
        self.assertEqual(result["status"], "not_detected")
        self.assertEqual(len(result["attempts"]), 1)

    def test_opaque_alpha_preserves_the_normal_positive_basis(self):
        result = bridge.scan_open_watermarks(self.save(self.positive.convert("RGBA")))
        self.assert_sd1(result, "stored_rgb")
        self.assertEqual(result["alpha_handling"], "opaque_rgb")

    def test_partial_alpha_reads_uncomposited_rgb_and_keeps_source_unchanged(self):
        rgba = self.positive.convert("RGBA")
        alpha = Image.new("L", rgba.size, 255)
        alpha.putpixel((0, 0), 0)
        alpha.putpixel((1, 0), 128)
        rgba.putalpha(alpha)
        self.assertTrue(np.array_equal(np.asarray(rgba.convert("RGB")), np.asarray(self.positive)))
        source = self.save(rgba, orientation=6)
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        result = bridge.scan_open_watermarks(source)
        self.assert_sd1(result, "stored_rgb_uncomposited")
        self.assertEqual(result["alpha_handling"], "uncomposited_rgb")
        self.assertEqual(original_hash, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_partial_alpha_and_orientation_fallback_record_both_bases(self):
        rgba = self.positive.transpose(Image.Transpose.ROTATE_90).convert("RGBA")
        alpha = Image.new("L", rgba.size, 128)
        rgba.putalpha(alpha)
        result = bridge.scan_open_watermarks(self.save(rgba, orientation=6))
        self.assert_sd1(result, "exif_oriented_rgb_uncomposited", attempts=2)
        self.assertEqual(result["attempts"][0]["basis"], "stored_rgb_uncomposited")
        self.assertEqual(result["alpha_handling"], "uncomposited_rgb")

    def test_partial_alpha_does_not_turn_unmarked_rgb_into_a_positive(self):
        rgba = self.negative.convert("RGBA")
        alpha = Image.fromarray(np.broadcast_to(np.arange(512, dtype=np.uint16) % 256, (512, 512)).astype(np.uint8))
        rgba.putalpha(alpha)
        result = bridge.scan_open_watermarks(self.save(rgba, orientation=6))
        self.assertEqual(result["status"], "not_detected", result)
        self.assertEqual(result["basis"], "stored_rgb_uncomposited")
        self.assertEqual(result["alpha_handling"], "uncomposited_rgb")
        self.assertFalse(any(c["matched"] for a in result["attempts"] for c in a["comparisons"]))

    def test_fully_transparent_pixels_do_not_claim_a_visible_image_verdict(self):
        for carrier in (self.positive, self.negative):
            with self.subTest(carrier=carrier is self.positive):
                rgba = carrier.convert("RGBA")
                rgba.putalpha(Image.new("L", rgba.size, 0))
                result = bridge.scan_open_watermarks(self.save(rgba))
                self.assertEqual(result["status"], "unsupported")
                self.assertEqual(result["alpha_handling"], "fully_transparent")
                self.assertEqual(result["attempts"], [])
                self.assertEqual(result["matches"], [])

    def test_decoder_failure_stays_an_error_with_attempt_evidence(self):
        with patch("remove_ai_watermarks.dwt_dct.decode_dwt_dct_lengths", side_effect=RuntimeError("decoder failed")):
            result = bridge.scan_open_watermarks(FIXTURE)
        self.assertEqual(result["status"], "error")
        self.assertIn("decoder failed", result["error"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["attempts"][0]["status"], "error")
        self.assertIn("decoder failed", result["attempts"][0]["error"])

    def test_public_scan_imports_no_generation_runtime(self):
        script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("public_scan_only", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.scan_open_watermarks(sys.argv[2])
assert result["status"] == "detected", result
heavy = {"torch", "torchvision", "diffusers", "transformers", "accelerate"}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
"""
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script,
            str(ROOT / "open_watermark_bridge.py"), str(FIXTURE)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

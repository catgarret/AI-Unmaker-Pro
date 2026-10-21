"""CPU-only numerical/output-review regressions; no trained inference tests."""
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_inference as q


class PixelPreservationTests(unittest.TestCase):
    def test_large_visible_change_requires_review_and_is_not_a_watermark_verdict(self):
        before = Image.new("RGB", (80, 80), (80, 120, 160))
        after = Image.new("RGB", before.size, (90, 130, 170))
        report = q._qwen_preservation_check(q._qwen_compare_pixels(before, after))
        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["reasons"], ["pixel_change_exceeds_budget"])
        self.assertFalse(report["watermark_verified"])
        self.assertFalse(report["visual_quality_verified"])
        self.assertEqual(report["reference"], "current_input")

    def test_small_change_stays_within_budget_without_claiming_perceptual_quality(self):
        before = Image.new("RGB", (80, 80), (80, 120, 160))
        after = Image.new("RGB", before.size, (81, 121, 161))
        report = q._qwen_preservation_check(q._qwen_compare_pixels(before, after))
        self.assertEqual(report["status"], "within_budget")
        self.assertFalse(report["visual_quality_verified"])
        self.assertFalse(report["watermark_verified"])

    def test_identity_is_finite_and_not_a_verified_removal(self):
        source = Image.new("RGB", (80, 80), (80, 120, 160))
        report = q._qwen_preservation_check(q._qwen_compare_pixels(source, source))
        self.assertEqual(report["status"], "within_budget")
        self.assertEqual(report["root_mean_squared_error"], 0)
        self.assertIsNone(report["psnr_db"])
        self.assertFalse(report["watermark_verified"])

    def test_invisible_rgb_does_not_trigger_a_visible_change_review(self):
        before = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
        after = Image.new("RGBA", before.size, (255, 255, 255, 0))
        report = q._qwen_preservation_check(q._qwen_compare_pixels(before, after))
        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(report["reasons"], ["no_visible_pixels"])

    def test_missing_and_invalid_measurements_cannot_silently_pass(self):
        for value in ({}, {"compared_pixels": 1},
                      {"compared_pixels": -1, "root_mean_squared_error": 0},
                      {"compared_pixels": 1, "root_mean_squared_error": math.nan},
                      {"compared_pixels": 1, "root_mean_squared_error": math.inf}):
            with self.subTest(value=value):
                self.assertEqual(q._qwen_preservation_check(value)["status"], "review_required")

    def test_stripped_output_matches_desynth_full_gaussian_reference(self):
        rng = np.random.default_rng(39)
        source = rng.integers(0, 256, (311, 199, 3), dtype=np.uint8)
        clean = rng.integers(0, 256, source.shape, dtype=np.uint8)
        first, second = source.astype(np.float32), clean.astype(np.float32)
        sigma = 1.95
        expected_float = np.clip(cv2.GaussianBlur(second, (0, 0), sigmaX=sigma, sigmaY=sigma)
                                + (first - cv2.GaussianBlur(first, (0, 0), sigmaX=sigma, sigmaY=sigma)), 0, 255)
        result = np.asarray(q._qwen_restore_detail(Image.fromarray(clean), Image.fromarray(source), sigma))
        # Independent full-frame reference vs bounded strips, including seams.
        self.assertTrue(np.array_equal(result, expected_float.astype(np.uint8)))
        bias = float(np.mean(result.astype(np.float32) - expected_float))
        self.assertLess(bias, -.40)
        truncated_bias = float(np.mean(expected_float.astype(np.uint8).astype(np.float32) - expected_float))
        self.assertLess(truncated_bias, -.40)

    def test_identical_restore_keeps_pixels_and_dimensions(self):
        rng = np.random.default_rng(41)
        source = Image.fromarray(rng.integers(0, 256, (311, 199, 3), dtype=np.uint8))
        restored = q._qwen_restore_detail(source, source, 1.95)
        self.assertEqual(restored.size, source.size)
        self.assertTrue(np.array_equal(np.asarray(source), np.asarray(restored)))


if __name__ == "__main__":
    unittest.main()

"""행 단위 처리의 경계·정수 변환·취소 검사. 제거 성능 검사가 아닙니다."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import detail_restore as detail
import qwen_inference as qwen


class DetailRestoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(173)
        self.source = Image.fromarray(rng.integers(0, 256, (517, 83, 3), dtype=np.uint8))
        self.clean = Image.fromarray(rng.integers(0, 256, (205, 107, 3), dtype=np.uint8))

    def test_strip_boundaries_match_whole_frame_for_all_modes(self):
        # 한 번에 전체 행을 처리한 결과와 비교해 필터 주변부 부족을 검출합니다.
        for mode in ("gaussian", "edge"):
            for sigma in (.1, 1.95, 4):
                for unsharp in (0, .2, 1):
                    with self.subTest(mode=mode, sigma=sigma, unsharp=unsharp):
                        options = dict(mode=mode, sigma=sigma, unsharp=unsharp)
                        expected = detail.restore_image_detail(self.clean, self.source, **options)
                        with patch.dict(detail.DETAIL_RESTORE_POLICY, strip_rows=2048):
                            whole = detail.restore_image_detail(self.clean, self.source, **options)
                        np.testing.assert_array_equal(expected, whole)

    def test_flat_image_has_no_false_edges_or_sharpening(self):
        for size in ((1, 1), (1, 513), (129, 1), (129, 257)):
            original = Image.new("RGB", size, (80, 100, 120))
            clean = Image.new("RGB", size, (60, 90, 130))
            expected = detail.restore_image_detail(clean, original, sigma=1.95)
            actual = detail.restore_image_detail(clean, original, sigma=1.95, mode="edge", unsharp=.2)
            np.testing.assert_array_equal(actual, expected)

    def test_options_change_pixels_without_mutating_inputs(self):
        before = self.source.tobytes(), self.clean.tobytes()
        base = detail.restore_image_detail(self.clean, self.source, sigma=1.95)
        for options in ({"mode": "edge"}, {"unsharp": .2}, {"mode": "edge", "unsharp": .2}):
            actual = detail.restore_image_detail(self.clean, self.source, sigma=1.95, **options)
            self.assertNotEqual(actual.tobytes(), base.tobytes())
            self.assertEqual(actual.size, self.source.size)
        self.assertEqual(before, (self.source.tobytes(), self.clean.tobytes()))

    def test_invalid_settings_fail_before_model_access(self):
        for options in ({"restore_mode": "unknown"}, {"unsharp": float("nan")},
                        {"unsharp": float("inf")}, {"unsharp": -1}, {"unsharp": 1.01},
                        {"restore": False, "restore_mode": "edge"}, {"restore": False, "unsharp": .2}):
            with self.subTest(options=options), self.assertRaises(qwen.ModelRuntimeError):
                qwen.regenerate_image("missing.png", "unused.png", options)

    def test_cancel_during_edge_processing(self):
        count = 0
        def cancelled():
            nonlocal count
            count += 1
            if count == 3:
                raise qwen.ModelCancelled("cancelled")
        with self.assertRaises(qwen.ModelCancelled):
            detail.restore_image_detail(self.clean, self.source, sigma=1.95, mode="edge", check_cancel=cancelled)
        self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()

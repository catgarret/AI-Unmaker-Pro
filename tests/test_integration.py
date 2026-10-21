"""Real metadata/pixel/save regressions; neural defaults have separate tests."""
from pathlib import Path
import errno
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image, ImageCms, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unmarker_test_app", ROOT / "main.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unmarker-integration-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def process(self, source, target, options=None, progress_callback=None, cancel_event=None):
        chosen = {"regenerate": False, "pixel_process": True}
        chosen.update(options or {})
        return app.app_process(source, target, chosen, progress_callback, cancel_event)

    def image(self, name="source.png", alpha=False, metadata=False):
        y, x = np.indices((288, 320))
        a = np.stack(((x * 3 + y) % 256, (x + y * 5) % 256, (x * 2 + y * 2) % 256), axis=2).astype(np.uint8)
        if alpha:
            layer = np.full(a.shape[:2], 255, np.uint8)
            layer[:, :80] = 0
            layer[:, 80:100] = 128
            a = np.dstack((a, layer))
        path = self.directory / name
        kwargs = {}
        if metadata:
            info = PngImagePlugin.PngInfo()
            info.add_text("Software", "test editor")
            info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta>test</x:xmpmeta>")
            exif = Image.Exif()
            exif[305] = "test editor"
            kwargs = {"pnginfo": info, "exif": exif,
                      "icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()}
        Image.fromarray(a).save(path, **kwargs)
        if metadata:
            # Requestable + label flags: label starts immediately after flags.
            descriptor = app._C2PA_UUID + bytes([3]) + b"c2pa\0"
            jumd = struct.pack(">I", 8 + len(descriptor)) + b"jumd" + descriptor
            jumb = struct.pack(">I", 8 + len(jumd)) + b"jumb" + jumd
            data = path.read_bytes()
            path.write_bytes(data[:33] + png_chunk(b"caBX", jumb) + data[33:])
        return path

    def test_explicit_pixel_pass_processes_unmarked_image_without_models(self):
        source = self.image(alpha=True, metadata=True)
        before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        target = self.directory / "processed.png"
        with patch.object(app, "prepare_models", side_effect=AssertionError("model download")), \
                patch.object(app, "_qwen_load_pipeline", side_effect=AssertionError("neural pipeline")):
            result = self.process(source, target)
        self.assertTrue(result["processing_completed"])
        self.assertTrue(result["pixel_processing"]["processed"])
        self.assertFalse(result["regeneration"]["processed"])
        self.assertFalse(result["validation"]["synthid_verified"])
        self.assertGreater(result["pixel_processing"]["changed_pixels"], 0)
        self.assertEqual(before_hash, hashlib.sha256(source.read_bytes()).hexdigest())
        with Image.open(source) as before, Image.open(target) as after:
            self.assertEqual(before.size, after.size)
            self.assertTrue(np.array_equal(np.asarray(before.getchannel("A")), np.asarray(after.getchannel("A"))))
            self.assertEqual(before.info["icc_profile"], after.info["icc_profile"])
            hidden = np.asarray(before.getchannel("A")) == 0
            self.assertTrue(np.array_equal(np.asarray(before)[hidden], np.asarray(after)[hidden]))
        for category in ("exif", "xmp", "text", "c2pa", "jumbf", "comment", "other"):
            self.assertFalse(result["output_scan"]["metadata"]["metadata"][category]["present"])
        measured = app.scan_frequency_signature(target)
        self.assertEqual(measured["peak_count"], result["output_scan"]["frequency"]["peak_count"])
        self.assertEqual(result["pixel_processing"]["output_path"], str(target))
        self.assertGreaterEqual(result["pixel_processing"]["pixel_comparison"]["psnr_db"], 40.)
        json.dumps(result, allow_nan=False)
        # Discovery of test_model_memory imports Torch in this parent process.
        # Check the explicit pixel-only path in a fresh interpreter instead of
        # confusing another test module's imports with this path's behavior.
        check = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("pixel_only_check", sys.argv[1])
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
result = app.app_process(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]),
                         {"regenerate": False, "pixel_process": True})
assert result["pixel_processing"]["processed"]
assert not result["regeneration"]["processed"]
heavy = {"torch", "torchvision", "diffusers", "transformers", "accelerate"}
loaded = sorted(name for name in sys.modules if name.split(".")[0] in heavy)
assert not loaded, loaded
"""
        fresh = subprocess.run(
            [sys.executable, "-I", "-B", "-c", check, str(ROOT / "main.py"),
             str(source), str(self.directory / "pixel-only-fresh.png")],
            capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)

    def test_metadata_only_keeps_pixels(self):
        source = self.image(alpha=True, metadata=True)
        target = self.directory / "metadata.png"
        result = self.process(source, target, {"pixel_process": False, "remove_visible": False})
        self.assertTrue(result["metadata"]["original_pixels_unchanged"])
        self.assertFalse(result["pixel_processing"]["processed"])

    def test_no_visible_detection_does_not_skip_pixel_pass(self):
        source = self.image()
        target = self.directory / "no-mark.png"
        result = self.process(source, target)
        self.assertEqual(result["input_scan"]["visible"]["status"], "not_detected")
        self.assertEqual(result["visible"]["status"], "no_watermark")
        self.assertGreater(result["pixel_processing"]["changed_pixels"], 0)

    def test_unsupported_visible_scan_keeps_file_with_explicit_report(self):
        source = self.image()
        target = self.directory / "pixel-only.png"
        with patch.object(app, "app_scan_visible", return_value={"status": "unavailable", "marks": [], "detail": "test"}):
            result = self.process(source, target)
        self.assertTrue(target.is_file())
        self.assertTrue(result["pixel_processing"]["processed"])
        self.assertFalse(result["visible"]["verified"])
        self.assertEqual(result["visible"]["status"], "unvalidated")

    def test_external_volume_copy_fallback(self):
        source = self.image(alpha=True)
        target = self.directory / "external.png"
        with patch.object(app.os, "link", side_effect=OSError(errno.EXDEV, "test cross-device")):
            result = self.process(source, target, {"remove_visible": False})
        self.assertTrue(result["processing_completed"])
        self.assertTrue(target.is_file())
        self.assertEqual(set(p.name for p in self.directory.iterdir()), {source.name, target.name})

    def test_original_existing_output_and_symlink_are_protected(self):
        source = self.image()
        old = source.read_bytes()
        with self.assertRaises(ValueError):
            self.process(source, source)
        self.assertEqual(source.read_bytes(), old)
        target = self.directory / "existing.png"
        target.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            self.process(source, target)
        self.assertEqual(target.read_bytes(), b"keep me")
        dangling = self.directory / "dangling.png"
        dangling.symlink_to(self.directory / "does-not-exist")
        with self.assertRaises(ValueError):
            self.process(source, dangling)
        self.assertTrue(dangling.is_symlink())

    def test_cancel_before_and_during_pipeline_leaves_no_output(self):
        source = self.image()
        target = self.directory / "cancelled.png"
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(app.ProcessingCancelled):
            self.process(source, target, cancel_event=cancel)
        self.assertFalse(target.exists())
        cancel.clear()
        def progress(payload, detail=""):
            if isinstance(payload, dict) and payload.get("phase") == "pixel_process":
                cancel.set()
        with self.assertRaises(app.ProcessingCancelled):
            self.process(source, target, progress_callback=progress, cancel_event=cancel)
        self.assertFalse(target.exists())

    def test_corrupt_input_does_not_create_output(self):
        source = self.directory / "broken.png"
        source.write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
        target = self.directory / "output.png"
        with self.assertRaises((ValueError, OSError)):
            self.process(source, target)
        self.assertFalse(target.exists())

    def test_decoder_errors_are_not_negative_results(self):
        source = self.image()
        with patch("remove_ai_watermarks.dwt_dct.decode_dwt_dct_lengths", side_effect=RuntimeError("broken decoder")):
            result = app.scan_open_watermarks(source)
        self.assertEqual(result["status"], "error")
        self.assertIn("broken decoder", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

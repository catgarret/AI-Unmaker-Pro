"""Real local child-process round trips; no pretrained inference is mocked as success."""
from pathlib import Path
import hashlib
import importlib.util
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from PIL import Image, PngImagePlugin

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("worker_test_app",ROOT/"main.py")
app=importlib.util.module_from_spec(spec); spec.loader.exec_module(app)

class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix="unmarker-worker-test-")
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.source=self.root/"원본 with spaces.png"
        info=PngImagePlugin.PngInfo(); info.add_text("Software","test")
        Image.new("RGBA",(120,100),(20,40,60,128)).save(self.source,pnginfo=info)

    def test_real_child_saves_metadata_only_result_and_exits(self):
        output=self.root/"result.png"
        digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        events=[]
        result=app.app_process_isolated(self.source,output,
            {"regenerate":False,"pixel_process":False,"remove_visible":False},events.append)
        self.assertTrue(result["processing_completed"])
        self.assertTrue(result["execution"]["model_memory_released"])
        self.assertEqual(result["execution"]["worker_exit_code"],0)
        self.assertEqual(digest,hashlib.sha256(self.source.read_bytes()).hexdigest())
        with Image.open(output) as image:
            self.assertEqual(image.getpixel((0,0)),(20,40,60,128))
            self.assertNotIn("Software",image.info)
        self.assertTrue(events)
        self.assertEqual(result["app"]["engine_id"],app.ENGINE_ID)

    def test_child_error_is_reported_without_replacing_destination(self):
        output=self.root/"bad.ext"
        with self.assertRaisesRegex(RuntimeError,"png"):
            app.app_process_isolated(self.source,output,{"regenerate":False})
        self.assertFalse(output.exists())

    def test_cancel_during_child_start_releases_process(self):
        event=threading.Event()
        timer=threading.Timer(.01,event.set); timer.start()
        self.addCleanup(timer.cancel)
        start=time.monotonic()
        with self.assertRaises(app.ProcessingCancelled):
            app.app_process_isolated(self.source,self.root/"cancelled.png",
                {"regenerate":False,"pixel_process":False},cancel_event=event)
        self.assertLess(time.monotonic()-start,6)
        self.assertTrue(self.source.exists())

    def test_default_reaches_model_even_when_visible_scan_is_negative(self):
        class ReachedModel(Exception): pass
        output=self.root/"model.png"
        with patch.object(app,"app_scan_visible",return_value={"status":"not_detected","marks":[]}), \
             patch.object(app,"regenerate_image",side_effect=ReachedModel) as model:
            with self.assertRaises(ReachedModel):
                app.app_process(self.source,output)
        self.assertEqual(model.call_count,1)
        self.assertFalse(output.exists())

    def test_legacy_cache_cleanup_follows_verified_output_and_cannot_break_it(self):
        # This is explicitly a plumbing double, not pretrained inference.
        import shutil
        output=self.root/"migration.png"
        def fake_model(source,target,*args):
            shutil.copyfile(source,target)
            return {"processed":True,"weights_loaded":True,"verified":False}
        def failed_cache_cleanup(*args):
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                image.load()
                self.assertEqual(image.getpixel((0,0)),(20,40,60,128))
                self.assertNotIn("Software",image.info)
            raise OSError("OFFLINE injected cache maintenance failure")
        with patch.object(app,"regenerate_image",side_effect=fake_model), \
             patch.object(app,"prune_legacy_model_cache",side_effect=failed_cache_cleanup) as cleanup:
            result=app.app_process(self.source,output,{"remove_visible":False})
        self.assertTrue(result["processing_completed"])
        self.assertEqual(result["legacy_model_cache_cleanup"]["status"],"deferred")
        self.assertEqual(cleanup.call_count,1)

    def test_metadata_only_never_prunes_model_cache(self):
        with patch.object(app,"prune_legacy_model_cache",side_effect=AssertionError("no model run")) as cleanup:
            result=app.app_process(self.source,self.root/"metadata.png",
                {"regenerate":False,"remove_visible":False,"pixel_process":False})
        self.assertTrue(result["processing_completed"])
        cleanup.assert_not_called()

if __name__=="__main__": unittest.main()

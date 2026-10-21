"""Offline regression tests; fake inference tests do NOT establish model quality.

One engine test executes real tiny randomly initialized Qwen/GGUF/LoRA/VAE
operators. Other tests explicitly replace the heavy model to verify file,
alpha, cancellation and lifecycle guarantees without loading trained weights.
"""
import contextlib
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_inference as q
import cv2
import gguf
import numpy as np
from PIL import Image, PngImagePlugin
import torch
from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter


class _OfflineFakePipeline:
    """Deterministic plumbing double; never a detection/removal model."""
    def __init__(self, calls, behavior="valid"):
        self.calls, self.behavior = calls, behavior
        self.default_sample_size, self.vae_scale_factor = 128, 8
        self.transformer = torch.nn.Module()
        self.transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.vae = torch.nn.Module()
        for name in ("encoder", "quant_conv", "post_quant_conv", "decoder"):
            self.vae.add_module(name, torch.nn.Identity())
        self._qwen_mmap_report = {"verified": False, "test_double": True}

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        width, height = kwargs.get("width", 1024), kwargs.get("height", 1024)
        for step in range(int(kwargs["strength"] * kwargs["num_inference_steps"])):
            self.transformer.transformer_blocks[0](torch.ones(1))
            kwargs["callback_on_step_end"](self, step, step, {"latents": torch.ones(1)})
        for child in self.vae.children():
            child(torch.ones(1))
        output = np.asarray(kwargs["image"].resize((width, height)), dtype=np.float32) / 255
        output = np.clip(output + .02, 0, 1)
        if self.behavior == "nan":
            output[0, 0, 0] = np.nan
        return types.SimpleNamespace(images=[output])


class QwenRuntimeTests(unittest.TestCase):
    def test_actual_tiny_engine_is_offline_and_not_a_detector(self):
        with mock.patch("requests.Session.request", side_effect=AssertionError("Network forbidden in tiny probe")), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in tiny probe")):
            report = q.model_engine_preflight()
        self.assertTrue(report["ready"], report)
        self.assertFalse(report["weights_loaded"])
        self.assertFalse(report["watermark_verified"])
        self.assertEqual(report["checked"], ["qwen", "gguf", "lightning", "vae", "embeddings", "detail_restore"])
        self.assertTrue(report["pipeline"]["mmap"]["verified"])
        self.assertGreater(report["pipeline"]["lora_scale_effect"], 0)
        self.assertGreaterEqual(report["pipeline"]["vae_staging_calls"], 4)
        self.assertEqual(report["pipeline"]["transformer_staging_calls"], 2)

    def test_mmap_reader_is_copy_on_write_and_pointer_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gguf"
            writer = gguf.GGUFWriter(str(path), "qwen_image")
            quant = gguf.quants.quantize(np.arange(64, dtype=np.float32).reshape(1, 64),
                                         gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor("weight", quant, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            readers = []
            state = q._qwen_read_mapped_checkpoint(torch, path, readers)
            layer = GGUFLinear(64, 1, compute_dtype=torch.float32)
            layer.weight = state["weight"]
            self.assertTrue(q._qwen_check_mmap_binding(layer, readers)["verified"])
            with torch.inference_mode():
                output = layer(torch.ones(1, 64))
                self.assertTrue(bool(torch.isfinite(output).all()))
                layer.weight.data = layer.weight.data.to("cpu:0")
            with self.assertRaises(q.ModelRuntimeError):
                q._qwen_check_mmap_binding(layer, readers)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_block_reference_restores_even_after_forward_failure(self):
        class FailingLinear(GGUFLinear):
            fail = False
            pointers = []

            def forward(self, value):
                self.pointers.append(self.weight.data_ptr())
                result = super().forward(value)
                if self.fail:
                    raise RuntimeError("Deliberate test failure")
                return result

        quant = gguf.quants.quantize(np.arange(32, dtype=np.float32).reshape(1, 32),
                                     gguf.GGMLQuantizationType.Q8_0)
        layer = FailingLinear(32, 1, compute_dtype=torch.float32)
        layer.weight = GGUFParameter(torch.from_numpy(quant), quant_type=gguf.GGMLQuantizationType.Q8_0)
        source_pointer, parameter_identity = layer.weight.data_ptr(), id(layer.weight)
        # cpu:0 intentionally forces a real storage copy on the CPU-only test
        # host. Production CPU execution uses unindexed cpu and does no copy.
        group = q._QwenResidentGroup(layer, "cpu:0")
        with torch.inference_mode():
            layer(torch.ones(1, 32))
            self.assertNotEqual(layer.pointers[-1], source_pointer)
            self.assertEqual(layer.weight.data_ptr(), source_pointer)
            layer.fail = True
            with self.assertRaisesRegex(RuntimeError, "Deliberate"):
                layer(torch.ones(1, 32))
        self.assertFalse(group.active)
        self.assertEqual(id(layer.weight), parameter_identity)
        self.assertEqual(layer.weight.data_ptr(), source_pointer)
        self.assertEqual(layer.weight.quant_type, gguf.GGMLQuantizationType.Q8_0)
        group.close()
        self.assertIsNone(group.module)
        self.assertFalse(layer._forward_hooks)
        self.assertFalse(layer._forward_pre_hooks)

    def test_embeddings_require_hash_and_consistent_mask_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeds.pt"
            cache = {"prompt_embeds": torch.ones(3, 32), "prompt_embeds_mask": torch.ones(3, dtype=torch.long),
                     "negative_prompt_embeds": torch.ones(4, 32),
                     "negative_prompt_embeds_mask": torch.ones(4, dtype=torch.long)}
            torch.save(cache, path)
            with mock.patch.object(torch, "load", wraps=torch.load) as load:
                result = q._qwen_load_embeddings(path, torch, torch.bfloat16, "cpu",
                                                 hidden_size=32, require_pinned=False)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(result["prompt_embeds"].shape, (1, 3, 32))
            self.assertEqual(result["prompt_embeds"].dtype, torch.bfloat16)
            self.assertEqual(result["prompt_embeds_mask"].dtype, torch.long)
            with mock.patch.object(torch, "load", side_effect=AssertionError("Hash must be checked first")):
                with self.assertRaises(q.ModelRuntimeError):
                    q._qwen_load_embeddings(path, torch, torch.bfloat16, "cpu", hidden_size=32)
            cache["prompt_embeds_mask"] = torch.zeros(3, dtype=torch.long)
            torch.save(cache, path)
            with self.assertRaises(q.ModelRuntimeError):
                q._qwen_load_embeddings(path, torch, torch.bfloat16, "cpu", hidden_size=32, require_pinned=False)

    def test_gaussian_strips_match_the_full_reference_formula(self):
        random = np.random.default_rng(39)
        original = random.integers(0, 256, (311, 199, 3), dtype=np.uint8)
        clean = random.integers(0, 256, (311, 199, 3), dtype=np.uint8)
        first, second = original.astype(np.float32), clean.astype(np.float32)
        sigma = 1.95
        expected = np.clip(cv2.GaussianBlur(second, (0, 0), sigmaX=sigma, sigmaY=sigma)
                           + (first - cv2.GaussianBlur(first, (0, 0), sigmaX=sigma, sigmaY=sigma)), 0, 255).astype(np.uint8)
        result = np.asarray(q._qwen_restore_detail(Image.fromarray(clean), Image.fromarray(original), sigma))
        self.assertTrue(np.array_equal(expected, result))

    def test_working_dimensions_preserve_aspect_and_stay_bounded(self):
        work, crop = q._qwen_prepare_image(Image.new("RGB", (1000, 333)), 512)
        self.assertEqual(crop, (512, 170))
        self.assertEqual(work.size, (512, 176))
        self.assertTrue(all(value % 16 == 0 for value in work.size))
        square, crop = q._qwen_prepare_image(Image.new("RGB", (1080, 1080)), 1024)
        self.assertEqual(square.size, (1024, 1024))
        self.assertEqual(crop, square.size)

    @contextlib.contextmanager
    def fake_runtime(self, behavior="valid"):
        calls = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(q, "model_runtime_status", return_value={"missing": []}))
            stack.enter_context(mock.patch.object(q, "prepare_models", return_value={"embeddings": "/never-read"}))
            stack.enter_context(mock.patch.object(q, "_qwen_compute_dtype", return_value=torch.bfloat16))
            stack.enter_context(mock.patch.object(q, "_qwen_memory_guard", return_value={}))
            stack.enter_context(mock.patch.object(q, "_qwen_load_embeddings", return_value={}))
            stack.enter_context(mock.patch.object(q, "_qwen_load_pipeline",
                                                   side_effect=lambda *args: _OfflineFakePipeline(calls, behavior)))
            yield calls

    def make_source(self, directory):
        source = Path(directory) / "original.png"
        random = np.random.default_rng(81)
        pixels = random.integers(0, 220, (71, 87, 4), dtype=np.uint8)
        pixels[:3, :, 3] = 0
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("comment", "file plumbing fixture")
        Image.fromarray(pixels).save(source, pnginfo=metadata, icc_profile=b"ICC-fixture-data")
        return source, pixels

    def test_offline_fake_single_pass_preserves_source_alpha_icc_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory, self.fake_runtime() as calls:
            source, pixels = self.make_source(directory)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = Path(directory) / "output.png"
            report = q.regenerate_image(source, target, {"device": "cpu", "max_working_side": 256})
            self.assertEqual(len(calls), 1, "No duplicate warm-up/sample pass")
            self.assertEqual(report["inference_calls"], 1)
            self.assertTrue(report["detail_restore"]["performed"])
            self.assertFalse(report["synthid"]["removal_verified"])
            self.assertEqual(report["synthid"]["status"], "not_tested")
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            with Image.open(target) as saved:
                self.assertEqual(saved.size, (87, 71))
                self.assertEqual(saved.info["icc_profile"], b"ICC-fixture-data")
                self.assertNotIn("comment", saved.info)
                actual = np.asarray(saved)
            self.assertTrue(np.array_equal(actual[:, :, 3], pixels[:, :, 3]))
            self.assertTrue(np.array_equal(actual[pixels[:, :, 3] == 0], pixels[pixels[:, :, 3] == 0]))

    def test_cancel_before_load_and_during_inference_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory, self.fake_runtime() as calls:
            source, _ = self.make_source(directory)
            cancel = threading.Event()
            cancel.set()
            target = Path(directory) / "cancelled.png"
            with self.assertRaises(q.ModelCancelled):
                q.regenerate_image(source, target, {"device": "cpu"}, cancel_event=cancel)
            self.assertFalse(calls)
            cancel.clear()

            def progress(payload, detail=""):
                if isinstance(payload, dict) and payload.get("phase") == "inference":
                    cancel.set()

            with self.assertRaises(q.ModelCancelled):
                q.regenerate_image(source, target, {"device": "cpu"}, progress, cancel)
            self.assertFalse(target.exists())

    def test_cpu_default_matches_reference_without_silent_downscale(self):
        with tempfile.TemporaryDirectory() as directory, self.fake_runtime() as calls:
            source = Path(directory) / "large.png"
            Image.new("RGB", (1080, 1080), (83, 117, 151)).save(source)
            automatic = q.regenerate_image(source, Path(directory) / "automatic.png", {"device": "cpu"})
            explicit = q.regenerate_image(source, Path(directory) / "explicit.png",
                                          {"device": "cpu", "max_working_side": 1024})
            self.assertNotIn("width", calls[0])
            self.assertNotIn("height", calls[0])
            self.assertEqual(calls[1]["width"], 1024)
            self.assertIsNone(automatic["parameters"]["max_working_side"])
            self.assertEqual(automatic["working_size"], [1024, 1024])
            self.assertEqual(explicit["parameters"]["max_working_side"], 1024)
            self.assertEqual(automatic["original_size"], [1080, 1080])

    def test_nonfinite_model_output_fails_without_saving(self):
        with tempfile.TemporaryDirectory() as directory, self.fake_runtime("nan"):
            source, _ = self.make_source(directory)
            target = Path(directory) / "bad.png"
            with self.assertRaises(q.ModelRuntimeError):
                q.regenerate_image(source, target, {"device": "cpu"})
            self.assertFalse(target.exists())

    def test_existing_output_and_original_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory, self.fake_runtime() as calls:
            source, _ = self.make_source(directory)
            with self.assertRaises(q.ModelRuntimeError):
                q.regenerate_image(source, source)
            target = Path(directory) / "existing.png"
            target.write_bytes(b"keep me")
            with self.assertRaises(FileExistsError):
                q.regenerate_image(source, target)
            self.assertEqual(target.read_bytes(), b"keep me")
            self.assertFalse(calls)

    def test_cgroup_reclaimable_pages_are_not_treated_as_heap(self):
        gib = 1024 ** 3
        fake_memory = types.SimpleNamespace(total=32 * gib, available=20 * gib)

        def read(path, *args, **kwargs):
            if str(path).endswith("memory.max"):
                return str(8 * gib)
            if str(path).endswith("memory.current"):
                return str(int(7.9 * gib))
            if str(path).endswith("memory.stat"):
                return f"file {6 * gib}\nshmem 0\nfile_dirty 0\nfile_writeback 0\n"
            raise OSError("not a fixture path")

        with mock.patch("psutil.virtual_memory", return_value=fake_memory), \
                mock.patch("pathlib.Path.read_text", autospec=True, side_effect=read):
            memory = q._qwen_memory_guard(torch, "cpu", loading=True)
        self.assertEqual(memory["limit_bytes"], 8 * gib)
        self.assertGreater(memory["available_bytes"], 6 * gib)
        with mock.patch.object(q, "_qwen_memory_snapshot", return_value={"available_bytes": gib // 2}):
            with self.assertRaises(q.ModelRuntimeError):
                q._qwen_memory_guard(torch, "cpu")

    def test_mps_worker_installs_hard_cap_even_when_inherited_env_disables_it(self):
        fake_mps = types.SimpleNamespace(set_per_process_memory_fraction=mock.Mock())
        fake_torch = types.SimpleNamespace(mps=fake_mps)
        with mock.patch.dict(q.os.environ, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}):
            q._qwen_apply_device_limits(fake_torch, "mps")
            fake_mps.set_per_process_memory_fraction.assert_called_once_with(.9)
            self.assertEqual(q.os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"], "0.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)

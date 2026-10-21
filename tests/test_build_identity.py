"""Regression checks for mixed releases, stale bundles, and local-model builds."""
from __future__ import annotations

import contextlib
import ast
import fnmatch
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import plistlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unmarker_build_identity_test", ROOT / "build.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def source_text(version=builder.APP_VERSION, engine=builder.ENGINE_ID, release=builder.RELEASE_ID, extra=""):
    return f'''APP_VERSION = {version!r}
ENGINE_ID = {engine!r}
RELEASE_ID = {release!r}
_RS_REQUIREMENTS = {{
    "pillow": ("Pillow>=10.4,<13", ">=10.4,<13"),
    "torch": ("torch>=2.6,<3", ">=2.6,<3"),
    "torchvision": ("torchvision>=0.21,<1", ">=0.21,<1"),
    "diffusers": ("diffusers==0.36.0", "==0.36.0"),
    "transformers": ("transformers>=4.51,<5", ">=4.51,<5"),
    "huggingface-hub": ("huggingface-hub>=0.34,<1", ">=0.34,<1"),
    "accelerate": ("accelerate>=1.2,<2", ">=1.2,<2"),
    "safetensors": ("safetensors>=0.4.5,<1", ">=0.4.5,<1"),
    "peft": ("peft==0.18.0", "==0.18.0"),
    "gguf": ("gguf==0.17.1", "==0.17.1"),
}}
MODEL_EXECUTION_POLICY = {{"max_working_side":1024, "batch_size":1, "threads":2,
    "cuda_memory_policy":"model_cpu_offload", "mps_memory_policy":"model_cpu_offload"}}
def app_process(*args, **kwargs):
    pass
def regenerate_image(*args, **kwargs):
    pass
def model_engine_preflight(*args, **kwargs):
    pass
{extra}
'''


class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="unmarker-build-test-")
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "main.py"
        self.source.write_text(source_text(), encoding="utf-8")
        self.identity = builder.source_build_identity(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def valid_result(self):
        return dict(self.identity, ready=True, model_preflight={
            "ready": True, "device": "cpu", "weights_loaded": False,
            "checked": ["qwen", "gguf", "lightning", "vae", "embeddings", "detail_restore"], "bounded_memory": True})

    def test_old_main_is_rejected_before_installer_or_build_directory(self):
        self.source.write_text('_RS_REQUIREMENTS = {"torch": ("torch>=2.6,<3", ">=2.6,<3")}\n', encoding="utf-8")
        stderr = io.StringIO()
        with mock.patch.object(builder, "prepare_build_environment", side_effect=AssertionError("installer must not run")) as install:
            with contextlib.redirect_stderr(stderr):
                code = builder.main(["--main", str(self.source)])
        self.assertEqual(code, 1)
        self.assertIn("APP_VERSION", stderr.getvalue())
        install.assert_not_called()
        self.assertFalse((self.root / "build").exists())

    def test_mixed_version_engine_and_release_are_rejected(self):
        for kwargs in ({"version": "1.0.16"}, {"engine": "local-sd15-controlnet-ipadapter-v1"}, {"release": "2026-10-20.4"}):
            with self.subTest(kwargs=kwargs):
                self.source.write_text(source_text(**kwargs), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    builder.source_build_identity(self.source)

    def test_required_local_implementation_and_no_api_paths(self):
        for text in (
            source_text().replace("def model_engine_preflight", "def missing_preflight"),
            source_text().replace("def regenerate_image", "def missing_regeneration"),
            source_text(extra="def check_openai_provenance():\n    pass\n"),
            source_text(extra="from openai import OpenAI"),
            source_text(extra="import importlib\ndef api():\n    importlib.import_module('openai')"),
            source_text(extra="from diffusers import ControlNetModel"),
            source_text(extra="from transformers import CLIPVisionModelWithProjection"),
            source_text(extra="import importlib\ndef legacy():\n    importlib.import_module('diffusers.loaders.ip_adapter')"),
        ):
            with self.subTest(text=text[-100:]):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    builder.source_build_identity(self.source)
        self.source.write_text(source_text(extra="import torch\nfrom diffusers import QwenImageTransformer2DModel"), encoding="utf-8")
        builder.source_build_identity(self.source)

    def test_qwen_dependencies_and_memory_limits_are_checked(self):
        cases = [
            source_text().replace('    "torch": ("torch>=2.6,<3", ">=2.6,<3"),\n', ''),
            source_text().replace('    "peft": ("peft==0.18.0", "==0.18.0"),\n', ''),
            source_text().replace('    "gguf": ("gguf==0.17.1", "==0.17.1"),\n', ''),
            source_text().replace('MODEL_EXECUTION_POLICY =', 'UNUSED_POLICY ='),
            source_text().replace('"max_working_side":1024', '"max_working_side":1080'),
            source_text().replace('"batch_size":1', '"batch_size":2'),
            source_text().replace('"threads":2', '"threads":64'),
            source_text().replace('"mps_memory_policy":"model_cpu_offload"', '"mps_memory_policy":""'),
        ]
        for text in cases:
            with self.subTest(text=text[-110:]):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    builder.source_build_identity(self.source)
        bundle = self.root / "bundle"
        library = bundle / "torch" / "lib" / "libtorch.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"test fixture")
        builder.audit_bundle(bundle)
        (bundle / "model.safetensors").write_bytes(b"weight fixture")
        with self.assertRaises(RuntimeError):
            builder.audit_bundle(bundle)
        (bundle / "model.safetensors").unlink()
        (bundle / "qwen.gguf").write_bytes(b"weight fixture")
        with self.assertRaises(RuntimeError):
            builder.audit_bundle(bundle)

    def test_source_digest_tracks_exact_packaged_bytes(self):
        self.assertEqual(self.identity["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.source.write_text(self.source.read_text() + "# source edited\n", encoding="utf-8")
        current = builder.source_build_identity(self.source)
        self.assertNotEqual(current["source_sha256"], self.identity["source_sha256"])
        with self.assertRaises(RuntimeError):
            builder.verify_engine_result(dict(self.valid_result(), **current), self.identity)

    def test_non_runtime_payloads_are_excluded_and_native_libraries_preserved(self):
        dropped = {
            "torch/lib/.libtorch_cpu.so.abC123": "temporary_native_library_copy",
            "torch/bin/test_api": "torch_test_executable",
            "torch/bin/HashStoreTest.exe": "torch_test_executable",
            "torch/test/cpp/test_jit": "torch_native_test_payload",
            "c2pa/libs/libc2pa_c.rlib": "c2pa_static_rust_archive",
            "c2pa/libs/libc2pa_c.d": "c2pa_build_dependency_manifest",
        }
        kept = [
            "torch/lib/libtorch_cpu.so", "torch/lib/libgomp.so.1",
            "torch/lib/libtorch_cpu.dylib", "torch/lib/c10.dll",
            "torch/bin/torch_shm_manager", "torch/testing/_internal/common_utils.py",
            "c2pa/libs/libc2pa_c.so", "c2pa/libs/libc2pa_c.dylib",
            "c2pa/libs/x86_64/c2pa_c.dll", "c2pa/LICENSE",
            "c2pa_python-0.38.0.dist-info/licenses/LICENSE",
            "diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py",
        ]
        # Exercise the generated spec's actual collection filter, without
        # invoking Analysis or importing the model/GUI dependencies.
        functions = [node for node in ast.parse(builder.SPEC_BODY).body
                     if isinstance(node, ast.FunctionDef) and node.name in {"exclusion_reason", "retain_runtime"}]
        namespace = {"Path": Path, "fnmatch": fnmatch, "MODEL_SUFFIXES": builder.WEIGHT_SUFFIXES,
                     "CACHE_PARTS": builder.CACHE_PARTS, "NON_RUNTIME_ASSETS": builder.NON_RUNTIME_ASSETS,
                     "excluded_entries": []}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "test-spec-filters", "exec"), namespace)
        for name in kept + list(dropped):
            expected = dropped.get(name)
            self.assertEqual(namespace["exclusion_reason"](name), expected)
            self.assertEqual(builder.bundle_exclusion_reason("_internal/" + name), expected)
            self.assertEqual(builder.bundle_exclusion_reason("Contents/Frameworks/" + name), expected)
        payload = self.root / "payload"
        payload.write_bytes(b"123456789")
        entries = [(name, str(payload), "DATA") for name in kept + list(dropped)]
        retained = namespace["retain_runtime"](entries)
        self.assertEqual([entry[0] for entry in retained], kept)
        report = namespace["excluded_entries"]
        self.assertEqual({entry["destination"]: entry["reason"] for entry in report}, dropped)
        self.assertTrue(all(entry["source"] == str(payload) and entry["bytes"] == 9 for entry in report))
        bundle = self.root / "clean-bundle"
        for name in kept:
            file = bundle / "_internal" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"runtime fixture")
        builder.audit_bundle(bundle)
        invalid = bundle / "_internal" / next(iter(dropped))
        invalid.write_bytes(b"temporary native fixture")
        with self.assertRaises(RuntimeError):
            builder.audit_bundle(bundle)

    def test_runtime_hook_embeds_selected_build_without_loading_application(self):
        fake_sys = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"sys": fake_sys}):
            exec(compile(builder.build_identity_hook(self.identity), "runtime-build-identity.py", "exec"), {})
        self.assertEqual(fake_sys._ai_unmarker_build, self.identity)

    def test_engine_result_requires_identity_and_actual_local_preflight(self):
        builder.verify_engine_result(self.valid_result(), self.identity)
        builder.verify_engine_result(dict(self.valid_result(), heavy_imports=["torch", "diffusers"]), self.identity)
        for override in (
            {"source_sha256": "0" * 64}, {"app_version": "1.0.15"},
            {"engine_id": "cpu-pixel-v2"}, {"release_id": "older"},
            {"ready": False}, {"model_preflight": None},
        ):
            with self.subTest(override=override):
                with self.assertRaises(RuntimeError):
                    builder.verify_engine_result(dict(self.valid_result(), **override), self.identity)
        for override in ({"ready": False}, {"device": "mps"}, {"weights_loaded": True},
                         {"checked": ["qwen", "gguf", "lightning", "vae", "embeddings"]}, {"bounded_memory": False}):
            with self.subTest(preflight_override=override):
                result = self.valid_result()
                result["model_preflight"].update(override)
                with self.assertRaises(RuntimeError):
                    builder.verify_engine_result(result, self.identity)

    def test_stale_probe_file_cannot_mask_child_that_produces_no_result(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "frozen-engine-probe.json").write_text(json.dumps(self.valid_result()), encoding="utf-8")
        child = self.root / "empty_child.py"
        child.write_text("pass\n", encoding="utf-8")
        with mock.patch.object(builder, "PROJECT_DIR", self.root):
            with self.assertRaises(RuntimeError):
                builder.verify_engine_runtime([sys.executable, str(child)], generated, self.identity, frozen=True)
        self.assertFalse((generated / "frozen-engine-probe.json").exists())

    def test_engine_probe_file_ipc_and_failed_exit(self):
        child = self.root / "probe_child.py"
        script = "import json,sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--result-file')+1]).write_text(json.dumps(" + repr(self.valid_result()) + "))\n"
        child.write_text(script, encoding="utf-8")
        with mock.patch.object(builder, "PROJECT_DIR", self.root):
            result = builder.verify_engine_runtime([sys.executable, str(child)], self.root / "generated", self.identity, frozen=False)
            self.assertEqual(result, self.valid_result())
            child.write_text(script + "raise SystemExit(3)\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                builder.verify_engine_runtime([sys.executable, str(child)], self.root / "generated", self.identity, frozen=False)

    def test_macos_bundle_version_and_identity_are_verified(self):
        app = self.root / "AI Unmarker Pro.app"
        contents = app / "Contents"
        contents.mkdir(parents=True)
        info = {"CFBundleName": "AI Unmarker Pro", "CFBundleDisplayName": "AI Unmarker Pro",
                "CFBundleShortVersionString": "1.0.14", "CFBundleVersion": "1.0.14"}
        info_file = contents / "Info.plist"
        info_file.write_bytes(plistlib.dumps(info))
        with self.assertRaises(RuntimeError):
            builder.verify_mac_bundle_identity(app, "AI Unmarker Pro", None)
        info.update(CFBundleShortVersionString=builder.APP_VERSION, CFBundleVersion=builder.APP_VERSION)
        info_file.write_bytes(plistlib.dumps(info))
        builder.verify_mac_bundle_identity(app, "AI Unmarker Pro", None)
        info["CFBundleName"] = "UNMARK"
        info_file.write_bytes(plistlib.dumps(info))
        with self.assertRaises(RuntimeError):
            builder.verify_mac_bundle_identity(app, "AI Unmarker Pro", None)

    def test_user_icon_wins_over_packaged_fallback(self):
        fallback = self.root / "assets" / "icons" / "icon.icns"
        fallback.parent.mkdir(parents=True)
        fallback.write_bytes(b"fallback fixture")
        args = types.SimpleNamespace(main=self.source, icon=None, font=[])
        with mock.patch.object(builder.sys, "platform", "darwin"):
            icon, assets = builder.asset_plan(args)
            self.assertEqual(icon, fallback)
            self.assertIn((str(fallback), "assets/icons"), assets)
            original = self.root / "icon.png"
            original.write_bytes(b"user fixture")
            self.assertEqual(builder.asset_plan(args)[0], original)
            self.assertFalse(any(dest == "assets/icons" for _src, dest in builder.asset_plan(args)[1]))
            preferred = self.root / "icons" / "icon.ico"
            preferred.parent.mkdir()
            preferred.write_bytes(b"user preferred fixture")
            self.assertEqual(builder.asset_plan(args)[0], preferred)
            explicit = self.root / "selected.png"
            explicit.write_bytes(b"explicit user fixture")
            args.icon = explicit
            self.assertEqual(builder.asset_plan(args)[0], explicit)

            original_font = self.root / "fonts" / "Pretendard-Regular.otf"
            original_font.parent.mkdir()
            original_font.write_bytes(b"user font fixture")
            old_font = self.root / "assets" / "fonts" / original_font.name
            old_font.parent.mkdir()
            old_font.write_bytes(b"old font fixture")
            license_file = old_font.parent / "Pretendard-LICENSE.txt"
            license_file.write_text("font license fixture")
            planned = builder.asset_plan(args)[1]
            self.assertIn((str(original_font), "fonts"), planned)
            self.assertNotIn((str(old_font), "assets/fonts"), planned)
            self.assertIn((str(license_file), "assets/fonts"), planned)

    def test_interrupted_rebuild_invalidates_previous_success_receipt(self):
        generated = self.root / "build" / "unmarker-generated"
        generated.mkdir(parents=True)
        receipt = generated / "build-receipt.json"
        marker = generated / "completed-build-layout.txt"
        receipt.write_text(json.dumps(self.identity), encoding="utf-8")
        marker.write_text("older completed build", encoding="utf-8")
        args = builder.argument_parser().parse_args(["--main", str(self.source)])
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(builder, "PROJECT_DIR", self.root))
            stack.enter_context(mock.patch.object(builder, "check_dependencies"))
            stack.enter_context(mock.patch.object(builder, "repair_native_files"))
            stack.enter_context(mock.patch.object(builder, "verify_source_runtime"))
            stack.enter_context(mock.patch.object(builder, "verify_engine_runtime", return_value=self.valid_result()))
            stack.enter_context(mock.patch.object(builder, "tk_runtime_binaries", return_value=[]))
            stack.enter_context(mock.patch.object(builder, "run_logged", side_effect=RuntimeError("packaging interrupted")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertEqual(builder._execute_build(args), 1)
        self.assertFalse(receipt.exists())
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

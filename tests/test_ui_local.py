"""Local UI contracts; no GUI, model download or inference is performed."""

import ast
import importlib.util
from pathlib import Path
import unittest


UI_PATH = Path(__file__).resolve().parents[1] / "desktop_ui.py"
_spec = importlib.util.spec_from_file_location("unmarker_local_ui_contract", UI_PATH)
_ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ui)
_tree = ast.parse(UI_PATH.read_text(encoding="utf-8"))
_app = next(node for node in ast.walk(_tree)
            if isinstance(node, ast.ClassDef) and node.name == "AIUnmarkerApp")


def method(name):
    return next(node for node in _app.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


class LocalUiContractTests(unittest.TestCase):
    def test_download_names_the_current_model_and_file(self):
        cases = (
            ("Frederic75/Qwen-Image-2512-GGUF", "Qwen-Image-2512-Q2_K_M.gguf", "구성 1/4 · Qwen 이미지 모델"),
            ("lightx2v/Qwen-Image-2512-Lightning", "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors", "구성 2/4 · Lightning 가속 모델"),
            ("Qwen/Qwen-Image-2512", "vae/diffusion_pytorch_model.safetensors", "구성 3/4 · 이미지 변환 모델"),
            ("0xROOTPLS/DeSynth", "embeds_cache.pt", "구성 4/4 · 처리 설정"),
        )
        for repo, filename, headline in cases:
            with self.subTest(repo=repo):
                labels = _ui._ui_local_model_progress({
                    "phase": "download", "title": "모델 다운로드", "detail": repo + " · " + filename,
                    "downloaded_bytes": 128, "total_bytes": 4096,
                })
                self.assertEqual(labels["headline"], headline)
                self.assertIn(filename, labels["activity"])
                self.assertIn("다운로드", labels["title"])

    def test_retry_and_verification_keep_the_file_identity(self):
        payload = {"title": "모델 다운로드 3/4",
                   "detail": "Qwen/Qwen-Image-2512 · vae/config.json"}
        retry = _ui._ui_local_model_progress(dict(payload, phase="retry"))
        verify = _ui._ui_local_model_progress(dict(payload, phase="verify"))
        self.assertEqual(retry["headline"], verify["headline"])
        self.assertIn("config.json", retry["activity"])
        self.assertIn("다시 시도", retry["activity"])
        self.assertIn("파일을 확인", verify["title"])
        self.assertIsNone(_ui._ui_local_model_progress({
            "phase": "dependencies", "title": "실행 준비 중", "detail": "필요한 구성요소를 설치합니다."}))

    def test_gui_processing_uses_the_isolated_worker(self):
        calls = [node.func.id for node in ast.walk(method("_process"))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("app_process_isolated", calls)
        self.assertNotIn("app_process", calls)
        options = next(node.value for node in ast.walk(method("_process"))
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == "options" for target in node.targets))
        self.assertEqual(ast.literal_eval(options)["regenerate"], True)
        self.assertEqual(ast.literal_eval(options)["pixel_process"], False)

    def test_download_progress_never_unlocks_the_app(self):
        progress_branch = next(node for node in ast.walk(method("_setup_event"))
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "kind"
            and any(isinstance(item, ast.Constant) and item.value == "progress"
                    for item in node.test.comparators))
        # The progress branch's body excludes the sibling final-ready branch.
        for statement in progress_branch.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Assign):
                    self.assertFalse(any(isinstance(target, ast.Attribute) and target.attr == "setup_ready"
                                         for target in node.targets))


if __name__ == "__main__":
    unittest.main()

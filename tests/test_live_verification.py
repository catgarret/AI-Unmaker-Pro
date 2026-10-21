"""검사 기록 조회가 실제 SynthID 검사로 둔갑하지 않도록 확인합니다."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_ui_test", ROOT / "desktop_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
tree = ast.parse((ROOT / "desktop_ui.py").read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_apply_live_verification")
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), "live-result", "exec"), namespace)
apply_result = namespace["_apply_live_verification"]


class LiveVerificationTests(unittest.TestCase):
    def result(self, **changes):
        return dict({"sha256": "a" * 64, "provider": "openai", "method": "official_api",
                     "checked_at": "2026-10-21", "signals": {"synthid": {"status": "detected"}}}, **changes)

    def test_past_positive_records_are_ignored(self):
        record = self.result(method="official_web_observation")
        scan = {"external_verification": {"sha256": record["sha256"], "records": [record]}}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "unverified")
        self.assertEqual(ui._ui_synthid_evidence(scan)["status"], "not_tested")

    def test_imported_web_observation_cannot_be_a_live_response(self):
        scan = {"live_verification": self.result(method="official_web_observation")}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "unverified")

    def test_frequency_or_declarations_cannot_replace_a_decoder(self):
        scan = {"frequency": {"detected": True, "score": 100},
                "synthid": {"status": "metadata_declared", "detected": True}}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "unverified")

    def test_api_error_is_not_negative(self):
        scan = {"live_verification": self.result(signals={"synthid": {"status": "error"}})}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "review")

    def test_missing_synthid_result_is_not_negative(self):
        scan = {"live_verification": self.result(signals={"c2pa": {"status": "not_detected"}})}
        self.assertEqual(ui._ui_synthid_evidence(scan)["status"], "not_tested")

    def test_live_response_only_attaches_to_uploaded_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            original, output = Path(folder)/"original.png", Path(folder)/"output.png"
            original.write_bytes(b"uploaded bytes")
            output.write_bytes(b"different output")
            view = SimpleNamespace(input_scan={"metadata": {}}, output_scan={"metadata": {}},
                input_path=str(original), result_path=str(output), phase="ready", _set_state=lambda _: None)
            response = self.result(sha256=hashlib.sha256(original.read_bytes()).hexdigest())
            apply_result(view, response)
            self.assertEqual(ui._ui_detection_summary(view.input_scan)["state"], "detected")
            self.assertEqual(ui._ui_detection_summary(view.output_scan)["state"], "unverified")


if __name__ == "__main__":
    unittest.main()

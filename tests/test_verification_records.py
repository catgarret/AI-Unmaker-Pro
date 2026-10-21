"""파일 혼동·잘못된 기록·화면 상태 덮어쓰기를 방지하는 회귀검사입니다."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


records = module("records_under_test", "verification_records.py")
ui = module("records_ui_under_test", "desktop_ui.py")


class VerificationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"original bytes")
        self.store = self.root / "records"
        self.row = {"sha256": records.verification_file_hash(self.image), "provider": "openai",
                    "method": "official_web_observation", "checked_at": "2026-10-21T09:00:00+00:00",
                    "signals": {"synthid": {"status": "detected"}, "c2pa": {"status": "not_detected"}}}

    def scan(self, path=None):
        return {"external_verification": records.file_verification_records(path or self.image, root=self.store)}

    def test_renamed_identical_file_matches_but_processed_bytes_do_not(self):
        records.save_verification_record(self.row, root=self.store)
        renamed = self.root / "renamed.png"
        renamed.write_bytes(self.image.read_bytes())
        self.assertEqual(ui._ui_detection_summary(self.scan(renamed))["state"], "detected")
        self.image.write_bytes(b"processed bytes")
        self.assertEqual(self.scan()["external_verification"]["records"], [])
        self.assertEqual(ui._ui_detection_summary(self.scan())["state"], "unverified")

    def test_import_rejects_different_image_before_writing(self):
        report = self.root / "report.json"
        report.write_text(json.dumps(dict(self.row, sha256="f" * 64)))
        with self.assertRaises(ValueError):
            records.import_file_verification(self.image, report, root=self.store)
        self.assertFalse(self.store.exists())

    def test_import_preserves_web_source_and_date(self):
        report = self.root / "report.json"
        report.write_text(json.dumps({"records": [self.row]}))
        external = records.import_file_verification(self.image, report, root=self.store)
        display = ui._ui_synthid_evidence({"external_verification": external})
        self.assertEqual(display["status"], "detected")
        self.assertIn("웹 검사 기록 · 2026-10-21", display["detail"])
        self.assertTrue(external["records"][0]["imported"])

    def test_errors_do_not_become_negative_and_provider_positive_is_preserved(self):
        records.save_verification_record(dict(self.row, provider="google",
            signals={"synthid": {"status": "error"}}), root=self.store)
        self.assertEqual(ui._ui_synthid_evidence(self.scan())["status"], "error")
        records.save_verification_record(self.row, root=self.store)
        evidence = ui._ui_synthid_evidence(self.scan())
        self.assertEqual(evidence["status"], "detected")
        self.assertEqual(evidence["positive_providers"], ["OpenAI"])
        self.assertIn("Google · 검사 오류", evidence["detail"])

    def test_only_latest_provider_record_is_used(self):
        records.save_verification_record(self.row, root=self.store)
        older = dict(self.row, checked_at="2026-10-20", signals={"synthid": {"status": "not_detected"}})
        records.save_verification_record(older, root=self.store)
        self.assertEqual(ui._ui_synthid_evidence(self.scan())["status"], "detected")
        newer = dict(older, checked_at="2026-10-22")
        records.save_verification_record(newer, root=self.store)
        self.assertEqual(ui._ui_detection_summary(self.scan())["state"], "checked")

    def test_malformed_cache_requires_review(self):
        records.save_verification_record(self.row, root=self.store)
        path = next(self.store.glob("*.json"))
        path.write_text("broken JSON")
        self.assertEqual(ui._ui_detection_summary(self.scan())["state"], "review")

    def test_no_records_are_not_a_negative_verdict(self):
        scan = {"frequency": {"status": "pattern_candidate", "detected": True, "score": 100},
                "synthid": {"status": "metadata_declared", "detected": True}}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "unverified")
        self.assertEqual(ui._ui_synthid_evidence(scan)["status"], "not_tested")

    def test_hash_mismatch_cannot_reach_gauge(self):
        scan = {"external_verification": {"sha256": "f" * 64, "records": [self.row]}}
        self.assertEqual(ui._ui_detection_summary(scan)["state"], "unverified")

    def test_unsafe_or_unsupported_record_fields_are_rejected(self):
        for replacement in ({"sha256": "../../outside"}, {"provider": []},
                            {"provider": "sightengine"}, {"method": "spectral_estimate"},
                            {"checked_at": None}, {"signals": {"genai": {"status": "detected"}}},
                            {"provider": "google", "method": "official_api"}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                records.normalise_verification_record(dict(self.row, **replacement))

    def test_oversized_report_is_not_loaded(self):
        report = self.root / "large.json"
        report.write_bytes(b" " * (records.VERIFICATION_RECORD_LIMIT + 1))
        with self.assertRaises(ValueError):
            records.read_verification_records(report)


if __name__ == "__main__":
    unittest.main()

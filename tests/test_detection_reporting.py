"""File evidence reaches the existing UI without becoming a false pixel result."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unmarker_detection_reporting", ROOT / "main.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
ui_spec = importlib.util.spec_from_file_location("unmarker_detection_ui", ROOT / "desktop_ui.py")
ui = importlib.util.module_from_spec(ui_spec)
ui_spec.loader.exec_module(ui)
ui_tree = ast.parse((ROOT / "desktop_ui.py").read_text(encoding="utf-8"))
ui_class = next(node for node in ast.walk(ui_tree)
                if isinstance(node, ast.ClassDef) and node.name == "AIUnmarkerApp")
refresh_node = next(node for node in ui_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_refresh_evidence")
state_node = next(node for node in ui_class.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_set_state")
notice_node = next(node for node in ui_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_result_notice")
refresh_tree = ast.Module(body=[refresh_node, state_node, notice_node], type_ignores=[])
namespace = {"re": re, "math": math, "Path": Path,
             "C": {"ink": "ink", "amber": "amber", "green": "green", "red": "red"},
             "_ui_metadata_evidence": ui._ui_metadata_evidence,
             "_ui_synthid_evidence": ui._ui_synthid_evidence,
             "_ui_detection_summary": ui._ui_detection_summary,
             "_ui_compact_evidence": ui._ui_compact_evidence}
exec(compile(refresh_tree, str(ROOT / "desktop_ui.py"), "exec"), namespace)
refresh_evidence = namespace["_refresh_evidence"]
set_state = namespace["_set_state"]
result_notice = namespace["_result_notice"]


class Widget:
    def __init__(self):
        self.values = {}
        self.visible = True

    def configure(self, **kwargs):
        self.values.update(kwargs)

    def grid(self, **kwargs):
        self.visible = True

    def grid_remove(self):
        self.visible = False

    def pack(self, **kwargs):
        self.visible = True

    def pack_forget(self):
        self.visible = False

    def start(self):
        self.values["running"] = True

    def stop(self):
        self.values["running"] = False


def report(*, c2pa="not_detected", metadata_present=(), public="not_detected"):
    fields = {key: {"present": key in metadata_present}
              for key in ("exif", "xmp", "text", "icc", "c2pa", "jumbf", "comment", "other")}
    fields["c2pa"]["status"] = c2pa
    return {"metadata": {"metadata": fields, "recorded_apps": []},
            "frequency": {"status": "measured", "peak_count": 0},
            "visible": {"status": "not_detected", "marks": []},
            "open_watermarks": {"status": public, "matches": []},
            "synthid": {"status": "not_tested", "declarations": []}}


def render(before, after=None):
    keys = ("synthid", "regeneration", "pixel", "metadata", "open_watermark", "visible")
    view = SimpleNamespace(
        tabs=SimpleNamespace(get=lambda: "처리 후" if after else "원본"),
        input_scan=before, output_scan=after, result=None, last_error=None,
        phase="done" if after else "ready", work_stage="",
        evidence_labels={key: (Widget(), Widget(), Widget()) for key in keys},
        evidence_cards={key: Widget() for key in keys},
        info_button=Widget(), evidence_title=Widget(),
    )
    refresh_evidence(view)
    return view


def render_state(scan, state="ready", *, result=None, after=None):
    view = render(scan, after)
    for key in ("state_title", "state_detail", "primary", "change_button", "preview_tools",
                "secondary", "error_details", "work_progress"):
        setattr(view, key, Widget())
    view.input_path = "/local/image.png" if scan else None
    view.saved_path = None
    view.result = result
    view._result_notice = lambda: result_notice(view)
    view._draw_icon = lambda icon: setattr(view, "drawn_icon", icon)
    view._refresh_evidence = lambda: refresh_evidence(view)
    view._reset_evidence_scroll = lambda: None
    view._set_small_status = lambda value: setattr(view, "small_status_value", value)
    view._sync_setup_gate = lambda: None
    view._sync_idle_view = lambda: None
    view._reset_and_pick = lambda: None
    view._cancel = lambda: None
    set_state(view, state)
    return view


def processed_result(after, *, preservation=None):
    regeneration = {"processed": True, "pixels_changed": True, "status": "processed",
                    "pixel_comparison": {"visual_quality_verified": False}}
    if preservation is not None:
        regeneration["preservation_check"] = preservation
    return {"regeneration": regeneration, "visible": {"status": "no_watermark"},
            "output_scan": after, "elapsed_seconds": 1.25}


class DetectionReportingTests(unittest.TestCase):
    def test_official_positive_survives_save_and_stays_on_its_own_tab(self):
        before, after = report(), report()
        before["live_verification"] = {
            "sha256": "a" * 64, "provider": "openai", "method": "official_api",
            "checked_at": "2026-10-21", "signals": {"synthid": {"status": "detected"}}}
        for state in ("ready", "done", "saved"):
            original = render_state(before, state, result=processed_result(after))
            self.assertEqual(original.drawn_icon, "detected")
            self.assertEqual(original.state_title.values["text"], "SynthID 탐지됨")
            self.assertEqual(original.evidence_labels["synthid"][1].values["text_color"], "red")
            self.assertNotIn("%", original.state_detail.values["text"])
        processed = render_state(before, "done", result=processed_result(after), after=after)
        self.assertEqual(processed.drawn_icon, "unverified")
        after["live_verification"] = before["live_verification"]
        retained = render_state(before, "saved", result=processed_result(after), after=after)
        self.assertEqual(retained.drawn_icon, "detected")

    def test_c2pa_cannot_hide_behind_other_metadata_count(self):
        scan = report(c2pa="detected", metadata_present=("exif", "xmp", "text", "c2pa", "jumbf"))
        view = render(scan)
        value = view.evidence_labels["metadata"][1].values["text"]
        self.assertTrue(value.startswith("C2PA"), value)

    def test_failed_credentials_are_never_reported_as_no_metadata(self):
        for status in ("error", "unavailable", "invalid", "external_reference"):
            with self.subTest(status=status):
                view = render(report(c2pa=status))
                value = view.evidence_labels["metadata"][1].values["text"]
                self.assertEqual(value, "확인하지 못했습니다")

    def test_public_decoder_failure_does_not_disappear_as_negative(self):
        for status in ("error", "unavailable", "unsupported", "insufficient_data"):
            with self.subTest(status=status):
                view = render(report(public=status))
                self.assertTrue(view.evidence_cards["open_watermark"].visible)
                self.assertEqual(view.evidence_labels["open_watermark"][1].values["text"],
                                 "검사 결과 확인 필요")

    def test_after_tab_uses_after_metadata(self):
        before = report(c2pa="detected", metadata_present=("c2pa", "exif", "text"))
        after = report()
        view = render(before, after)
        self.assertIn("C2PA", view.evidence_labels["metadata"][1].values["text"])
        self.assertIn("삭제 확인", view.evidence_labels["metadata"][1].values["text"])
        original_view = render(before)
        self.assertNotIn("삭제 확인", original_view.evidence_labels["metadata"][1].values["text"])

    def test_synthid_declaration_is_metadata_not_pixel_detection(self):
        metadata = {"metadata": {"c2pa": {"status": "detected", "watermark_declarations": [
            {"name": "SynthID", "scope": "active_manifest", "declaration": True,
             "pixel_detection": False, "signature_verified": True}
        ]}}}
        observed = app._app_synthid_observation(metadata)
        self.assertEqual(observed["status"], "metadata_declared")
        self.assertTrue(observed["metadata_declared"])
        self.assertIsNone(observed["detected"])
        self.assertFalse(observed["pixel_detection_supported"])
        scan = report(c2pa="detected", metadata_present=("c2pa",))
        scan["synthid"] = observed
        view = render(scan)
        self.assertEqual(view.evidence_labels["metadata"][1].values["text"], "SynthID · C2PA")

    def test_ingredient_declaration_does_not_become_current_image_result(self):
        metadata = {"metadata": {"c2pa": {"status": "detected", "watermark_declarations": [
            {"name": "SynthID", "scope": "ingredient", "declaration": True}
        ]}}}
        observed = app._app_synthid_observation(metadata)
        self.assertEqual(observed["status"], "not_tested")
        self.assertFalse(observed["metadata_declared"])
        self.assertEqual(observed["declarations"], [])
        self.assertEqual(len(observed["related_declarations"]), 1)

    def test_no_metadata_is_not_negative_synthid_verdict(self):
        observed = app._app_synthid_observation(report()["metadata"])
        self.assertEqual(observed["status"], "not_tested")
        self.assertIsNone(observed["detected"])
        self.assertFalse(observed["metadata_declared"])

    def test_metadata_positive_reaches_existing_summary_without_ai_probability(self):
        scan = report(c2pa="detected", metadata_present=("c2pa", "exif", "xmp", "jumbf", "other"))
        view = render_state(scan)
        self.assertEqual(view.state_title.values["text"], "메타데이터 탐지됨")
        self.assertEqual(view.primary.values["text"], "이미지 처리")
        detail = view.state_detail.values["text"]
        self.assertIn("메타데이터 · C2PA · EXIF · XMP", detail)
        self.assertNotIn("한 번에 실행합니다", detail)
        self.assertNotIn("%", detail)
        self.assertEqual(view.drawn_icon, "detected")

    def test_primary_recorded_app_is_not_hidden_behind_editor_exif(self):
        scan = report(c2pa="detected", metadata_present=("c2pa", "exif", "xmp"))
        editor = {"name": "Adobe Photoshop 27.1.0 (Macintosh)", "source": "EXIF:Software",
                  "scope": "file_metadata", "role": "file_software"}
        original = {"name": "OpenAI", "source": "C2PA action.softwareAgent",
                    "scope": "ingredient", "role": "action_software", "signature_verified": True}
        scan["metadata"]["recorded_apps"] = [editor, original]
        scan["metadata_observation"] = {"status": "detected", "primary_app": original,
                                        "categories": ["c2pa", "exif", "xmp"]}
        unchanged = copy.deepcopy(scan)
        view = render_state(scan)
        self.assertIn("기록된 앱: OpenAI", view.state_detail.values["text"])
        self.assertEqual(view.evidence_labels["metadata"][2].values["text"], "기록된 앱: OpenAI")
        self.assertEqual(scan, unchanged)
        self.assertEqual(scan["synthid"]["status"], "not_tested")

    def test_three_basic_record_categories_are_visible_in_card(self):
        view = render(report(c2pa="detected", metadata_present=("c2pa", "exif", "xmp", "jumbf")))
        self.assertEqual(view.evidence_labels["metadata"][1].values["text"], "C2PA · EXIF · XMP 외 1종")

    def test_icc_only_does_not_become_a_removal_target(self):
        scan = report(metadata_present=("icc",))
        view = render_state(scan)
        self.assertEqual(view.state_detail.values["text"], "이 앱에는 로컬 SynthID 검출기가 없습니다.\n공식 검사를 진행해 주세요.")
        self.assertEqual(view.evidence_labels["metadata"][2].values["text"], "색상 프로필 유지")

    def test_missing_record_cannot_be_created_by_summary_categories(self):
        scan = report()
        scan["metadata_observation"] = {"status": "detected", "categories": ["c2pa"], "detected": True}
        self.assertFalse(ui._ui_metadata_evidence(scan)["present"])

    def test_incomplete_scan_preserves_positive_file_evidence_and_review_state(self):
        scan = report(c2pa="unavailable", metadata_present=("c2pa", "exif"))
        view = render_state(scan)
        self.assertEqual(view.state_title.values["text"], "메타데이터 탐지됨")
        self.assertIn("C2PA · EXIF", view.state_detail.values["text"])
        self.assertIn("상세에서 검사 내용을 확인해 주세요.", view.state_detail.values["text"])
        self.assertEqual(view.drawn_icon, "detected")
        self.assertNotIn("삭제 대상 정보 없음", view.state_detail.values["text"])

    def test_supported_metadata_parser_failure_is_not_a_negative_summary(self):
        scan = report()
        scan["metadata_observation"] = {"status": "incomplete", "categories": [], "primary_app": None}
        view = render_state(scan)
        self.assertEqual(view.state_title.values["text"], "일부 검사가 필요합니다")
        self.assertEqual(view.evidence_labels["metadata"][1].values["text"], "확인하지 못했습니다")

    def test_synthid_declaration_stays_explicitly_inside_metadata_summary(self):
        scan = report(c2pa="detected", metadata_present=("c2pa",))
        scan["synthid"] = {"status": "metadata_declared", "declarations": [{"name": "SynthID"}],
                           "detected": None, "pixel_detection_supported": False}
        view = render_state(scan)
        self.assertIn("메타데이터 · SynthID · C2PA", view.state_detail.values["text"])
        self.assertNotIn("100", view.state_detail.values["text"])
        self.assertTrue(view.evidence_cards["open_watermark"].visible)
        self.assertIn("이 공개 코덱 검사는 SynthID 판정과 별개입니다.",
                      view.evidence_labels["open_watermark"][2].values["text"])

    def test_negative_public_result_keeps_its_scope_visible(self):
        before = report()
        after = report()
        for view in (render(before), render(before, after)):
            self.assertTrue(view.evidence_cards["open_watermark"].visible)
            self.assertEqual(view.evidence_labels["open_watermark"][0].values["text"],
                             "공개 비가시 워터마크")
            self.assertEqual(view.evidence_labels["open_watermark"][1].values["text"],
                             "등록된 공개 패턴과 불일치")
            self.assertIn("SynthID 판정", view.evidence_labels["open_watermark"][2].values["text"])
            self.assertNotIn("재검사에서 검출되지 않음", view.evidence_labels["open_watermark"][1].values["text"])

    def test_file_without_metadata_does_not_look_synthid_negative(self):
        view = render_state(report())
        self.assertEqual(view.state_title.values["text"], "SynthID 미검사")
        self.assertEqual(view.state_detail.values["text"],
                         "이 앱에는 로컬 SynthID 검출기가 없습니다.\n공식 검사를 진행해 주세요.")
        self.assertEqual(view.primary.values["text"], "이미지 처리")
        self.assertEqual(view.primary.values["state"], "normal")

    def test_positive_public_to_negative_still_reports_only_that_recheck(self):
        before = report(public="detected")
        before["open_watermarks"]["matches"] = [{"name": "Stable Diffusion XL"}]
        view = render(before, report())
        self.assertEqual(view.evidence_labels["open_watermark"][1].values["text"],
                         "재검사에서 검출되지 않음")
        self.assertIn("Stable Diffusion XL", view.evidence_labels["open_watermark"][2].values["text"])
        self.assertIsNone((view.output_scan.get("synthid") or {}).get("detected"))

    def test_public_watermark_retained_in_output_requires_review_after_processing_and_save(self):
        before = report(public="detected")
        after = report(public="detected")
        result = processed_result(after)
        for state in ("done", "saved"):
            with self.subTest(state=state):
                view = render_state(before, state, result=result, after=after)
                self.assertEqual(view.drawn_icon, "detected")
                self.assertIn("공개 비가시 워터마크", view.state_detail.values["text"])
                self.assertIn("출력에서 다시 검출됐습니다.", view.state_detail.values["text"])
                self.assertNotIn("처리와 재검사를 마쳤습니다", view.state_detail.values["text"])
                if state == "done":
                    self.assertEqual(view.state_title.values["text"], "워터마크 탐지됨")

    def test_public_output_check_failure_cannot_look_like_success(self):
        for status in ("error", "unavailable", "unsupported", "insufficient_data"):
            for state in ("done", "saved"):
                with self.subTest(status=status, state=state):
                    after = report(public=status)
                    view = render_state(report(), state, result=processed_result(after), after=after)
                    self.assertEqual(view.drawn_icon, "review")
                    self.assertIn("재검사 확인 필요", view.state_detail.values["text"])

    def test_incomplete_output_metadata_requires_review_even_when_encoding_succeeded(self):
        after = report()
        after["metadata_observation"] = {"status": "incomplete", "detected": None}
        for state in ("done", "saved"):
            with self.subTest(state=state):
                view = render_state(report(), state, result=processed_result(after), after=after)
                self.assertEqual(view.drawn_icon, "review")
                self.assertIn("메타데이터 · 확인하지 못했습니다", view.state_detail.values["text"])

    def test_explicit_preservation_failure_or_review_is_visible(self):
        after = report()
        for status in ("review_required", "failed"):
            for state in ("done", "saved"):
                with self.subTest(status=status, state=state):
                    result = processed_result(after, preservation={"status": status})
                    view = render_state(report(), state, result=result, after=after)
                    self.assertEqual(view.drawn_icon, "review")
                    self.assertEqual(view.state_detail.values["text"],
                                     "원본과 처리 결과를 비교해 주세요.\n상세에서 검사 내용을 확인해 주세요.")

    def test_legacy_missing_preservation_field_and_unverified_quality_are_not_failures(self):
        after = report()
        result = processed_result(after)
        self.assertFalse(result["regeneration"]["pixel_comparison"]["visual_quality_verified"])
        self.assertNotIn("preservation_check", result["regeneration"])
        for state in ("done", "saved"):
            with self.subTest(state=state):
                view = render_state(report(), state, result=result, after=after)
                self.assertEqual(view.drawn_icon, "unverified")
                self.assertEqual(result_notice(view), "")

    def test_passed_preservation_check_does_not_override_other_good_results(self):
        after = report()
        view = render_state(report(), "done", result=processed_result(after, preservation={"status": "passed"}), after=after)
        self.assertEqual(view.drawn_icon, "unverified")
        self.assertEqual(result_notice(view), "")

    def test_generic_detected_flags_are_not_watermark_or_quality_failures(self):
        after = report()
        after["metadata_observation"] = {"status": "detected", "detected": True, "pixel_detection": False}
        after["frequency"] = {"status": "pattern_candidate", "peak_count": 8, "detected": True,
                              "verified_watermark": False}
        after["synthid"] = {"status": "metadata_declared", "metadata_declared": True,
                            "detected": None, "pixel_detection_supported": False}
        view = render_state(report(), "done", result=processed_result(after), after=after)
        self.assertEqual(view.drawn_icon, "unverified")
        self.assertEqual(result_notice(view), "")

    def test_positive_input_with_negative_public_output_is_not_reported_as_retained(self):
        before, after = report(public="detected"), report(public="not_detected")
        view = render_state(before, "done", result=processed_result(after), after=after)
        self.assertEqual(view.drawn_icon, "unverified")
        self.assertEqual(result_notice(view), "")
        self.assertEqual(view.evidence_labels["open_watermark"][1].values["text"], "재검사에서 검출되지 않음")


if __name__ == "__main__":
    unittest.main()

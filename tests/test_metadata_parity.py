"""Desktop metadata evidence regressions; generated assets only, no GPU or API."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metadata = load_module("parity_metadata", "metadata_core.py")
backend = load_module("parity_backend", "app_backend.py")
AI_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
MAIN = b"http://ns.adobe.com/xap/1.0/\0"
EXT = b"http://ns.adobe.com/xmp/extension/\0"


def marker(payload):
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def xmp(attrs="", children=""):
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:xmpNote="http://ns.adobe.com/xmp/note/" '
        'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
        'xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#" '
        'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
        f'{attrs}>{children}</rdf:Description></rdf:RDF></x:xmpmeta>').encode()


class MetadataParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="unmarker-metadata-parity-")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.base = self.directory / "base.jpg"
        Image.new("RGB", (24, 18), (80, 125, 220)).save(self.base)
        self.raw = self.base.read_bytes()
        self.extended = xmp(f'xmp:CreatorTool="OpenAI" Iptc4xmpExt:DigitalSourceType="{AI_TYPE}"')
        self.guid = hashlib.md5(self.extended, usedforsecurity=False).hexdigest().upper()
        self.standard = xmp(f'xmp:CreatorTool="Adobe Photoshop" xmpNote:HasExtendedXMP="{self.guid}"')

    def packets(self, guid=None, total=None):
        guid = guid or self.guid
        total = len(self.extended) if total is None else total
        return [guid.encode() + struct.pack(">II", total, offset) + self.extended[offset:offset + 97]
                for offset in range(0, len(self.extended), 97)]

    def jpeg(self, packets=None, standard=None):
        path = self.directory / "test.jpg"
        packets = self.packets() if packets is None else packets
        standard = self.standard if standard is None else standard
        raw = (marker(MAIN + standard) if standard else b"") + b"".join(marker(EXT + p) for p in packets)
        path.write_bytes(self.raw[:2] + raw + self.raw[2:])
        return path

    def scan(self, path):
        # These extension/metadata tests do not depend on native SDK availability.
        with patch.dict(sys.modules, {"c2pa": None}):
            result = metadata.scan_image(path)
        result["metadata"]["c2pa"].update(status="not_detected", sdk_status="not_detected")
        return result

    def test_extended_xmp_out_of_order_reads_generator_and_ai_source(self):
        path = self.jpeg(list(reversed(self.packets())))
        original = path.read_bytes()
        result = self.scan(path)
        observed = backend._app_metadata_observation(result)
        self.assertTrue(observed["detected"])
        self.assertEqual(observed["categories"], ["xmp"])
        self.assertEqual(observed["primary_app"]["name"], "OpenAI")
        self.assertEqual(observed["primary_app"]["scope"], "file_metadata")
        self.assertFalse(observed["primary_app"]["signature_verified"])
        self.assertTrue(observed["ai_declared"])
        self.assertFalse(observed["pixel_detection"])
        self.assertEqual(path.read_bytes(), original)

    def test_missing_overlap_duplicate_or_disagreeing_fragments_are_incomplete(self):
        parts = self.packets()
        corrupt = [parts[0][:32] + struct.pack(">II", len(self.extended) + 3, 0) + parts[0][40:], *parts[1:]]
        overlap = [parts[0], parts[1][:36] + struct.pack(">I", 96) + parts[1][40:], *parts[2:]]
        for packets in (parts[:-1], [parts[0], *parts], corrupt, overlap):
            with self.subTest(parts=len(packets)):
                result = self.scan(self.jpeg(packets))
                self.assertTrue(result["metadata"]["xmp"]["present"])
                self.assertNotIn("OpenAI", [item["name"] for item in result["recorded_apps"]])
                self.assertEqual(backend._app_metadata_observation(result)["status"], "incomplete")

    def test_unreferenced_extension_and_wrong_namespace_do_not_supply_origin(self):
        for standard in (b"", self.standard.replace(b"http://ns.adobe.com/xmp/note/", b"https://example.invalid/")):
            result = self.scan(self.jpeg(standard=standard))
            self.assertNotIn("OpenAI", [item["name"] for item in result["recorded_apps"]])
            self.assertFalse(backend._app_metadata_observation(result)["ai_declared"])

    def test_digest_mismatch_does_not_attach_unrelated_generator(self):
        wrong = "A" * 32
        result = self.scan(self.jpeg(self.packets(wrong), self.standard.replace(self.guid.encode(), wrong.encode())))
        self.assertNotIn("OpenAI", [item["name"] for item in result["recorded_apps"]])
        self.assertIn("extended_xmp_digest_mismatch", [e["code"] for e in result["metadata"]["xmp"]["container_errors"]])

    def test_large_declared_extension_is_rejected_before_allocation(self):
        result = self.scan(self.jpeg(self.packets(total=0xffffffff)))
        self.assertNotIn("OpenAI", [item["name"] for item in result["recorded_apps"]])
        self.assertTrue(result["metadata"]["xmp"]["container_errors"])

    def test_namespace_qualified_software_and_history_agents_are_recorded(self):
        packet = xmp('tiff:Software="OpenAI"', '<stEvt:softwareAgent>Adobe Photoshop</stEvt:softwareAgent>')
        path = self.jpeg([], packet)
        result = self.scan(path)
        self.assertEqual({item["name"] for item in result["recorded_apps"]}, {"OpenAI", "Adobe Photoshop"})
        self.assertEqual(backend._app_metadata_observation(result)["primary_app"]["name"], "OpenAI")
        wrong = packet.replace(b"http://ns.adobe.com/tiff/1.0/", b"https://example.invalid/")
        result = self.scan(self.jpeg([], wrong))
        self.assertNotIn("OpenAI", [item["name"] for item in result["recorded_apps"]])

    def test_editor_name_does_not_become_ai_generation_declaration(self):
        exif = Image.Exif()
        exif[305] = "Adobe Photoshop 27.1.0"
        path = self.directory / "edited.jpg"
        Image.new("RGB", (24, 18)).save(path, exif=exif)
        observed = backend._app_metadata_observation(self.scan(path))
        self.assertEqual(observed["categories"], ["exif"])
        self.assertTrue(observed["detected"])
        self.assertFalse(observed["ai_declared"])
        self.assertFalse(observed["pixel_detection"])

    def test_colour_profile_alone_is_not_an_ai_or_source_trace(self):
        result = self.scan(self.base)
        result["metadata"]["icc"].update(present=True, count=1)
        observed = backend._app_metadata_observation(result)
        self.assertEqual(observed["status"], "not_detected")
        self.assertFalse(observed["detected"])
        self.assertTrue(observed["color_profile_present"])
        self.assertEqual(observed["categories"], [])

    def test_unavailable_reader_and_invalid_xmp_are_not_negative_results(self):
        result = self.scan(self.base)
        result["metadata"]["c2pa"].update(status="unavailable")
        observed = backend._app_metadata_observation(result)
        self.assertEqual(observed["status"], "incomplete")
        self.assertFalse(observed["complete"])
        self.assertIsNone(observed["detected"])
        result = self.scan(self.jpeg([], b"broken xml"))
        observed = backend._app_metadata_observation(result)
        self.assertEqual(observed["status"], "incomplete")
        self.assertTrue(observed["detected"])

    def test_current_editor_and_linked_generator_keep_independent_scopes(self):
        result = self.scan(self.base)
        report = {"active_manifest": "edit", "validation_results": {"activeManifest": {
            "success": [{"code": "claimSignature.validated"}]}}, "manifests": {
            "edit": {"claim_generator_info": [{"name": "Adobe Photoshop"}], "ingredients": [
                {"active_manifest": "original", "validation_results": {"activeManifest": {
                    "success": [{"code": "claimSignature.validated"}]}}}]},
            "original": {"claim_generator_info": [{"name": "OpenAI"}], "assertions": [
                {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created",
                    "softwareAgent": {"name": "DALL-E"}, "digitalSourceType": AI_TYPE}]}}]},
            "unrelated": {"claim_generator_info": [{"name": "Midjourney"}]}}}
        metadata._c2pa_report(report, result)
        apps = result["recorded_apps"]
        self.assertNotIn("Midjourney", [item["name"] for item in apps])
        adobe = next(item for item in apps if item["name"] == "Adobe Photoshop")
        openai = next(item for item in apps if item["name"] == "OpenAI")
        self.assertEqual(adobe["scope"], "active_manifest")
        self.assertEqual(openai["scope"], "ingredient")
        self.assertTrue(openai["signature_verified"])
        observed = backend._app_metadata_observation(result)
        self.assertEqual(observed["primary_app"]["name"], "DALL-E")
        self.assertEqual(observed["primary_app"]["scope"], "ingredient")
        self.assertTrue(observed["ai_declared"])
        self.assertFalse(observed["pixel_detection"])

    def test_summary_does_not_mutate_metadata_or_promote_name_to_ai_proof(self):
        result = self.scan(self.jpeg([], xmp('xmp:CreatorTool="OpenAI"')))
        before = copy.deepcopy(result)
        observed = backend._app_metadata_observation(result)
        self.assertEqual(result, before)
        self.assertEqual(observed["primary_app"]["name"], "OpenAI")
        self.assertFalse(observed["ai_declared"])
        self.assertFalse(observed["primary_app"]["signature_verified"])

    def test_import_scan_exposes_the_same_actual_metadata_observation(self):
        path = self.jpeg()
        with patch.multiple(backend, create=True, scan_image=self.scan,
                scan_frequency_signature=lambda path: {"status": "measured"},
                app_scan_visible=lambda path: {"status": "not_detected", "marks": []},
                scan_open_watermarks=lambda path: {"status": "not_detected", "matches": []},
                _app_detection_capabilities=lambda *args: None,
                _app_numeric_limits=lambda: None, app_identity=lambda: {}):
            result = backend.app_scan_file(path)
        self.assertEqual(result["metadata_observation"], backend._app_metadata_observation(result["metadata"]))
        self.assertEqual(result["metadata_observation"]["primary_app"]["name"], "OpenAI")
        self.assertEqual(result["synthid"]["status"], "not_tested")


if __name__ == "__main__":
    unittest.main()

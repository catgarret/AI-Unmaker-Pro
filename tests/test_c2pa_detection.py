"""CPU-only C2PA import regressions. No image upload, model, or download."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "c2pa"
SPEC = importlib.util.spec_from_file_location("c2pa_metadata_under_test", ROOT / "metadata_core.py")
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)
HAS_SDK = importlib.util.find_spec("c2pa") is not None


def box(kind, payload, extended=False):
    return ((struct.pack(">I4sQ", 1, kind, 16 + len(payload)) if extended else
             struct.pack(">I4s", 8 + len(payload), kind)) + payload)


def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


def marker(kind, payload):
    return b"\xff" + bytes([kind]) + struct.pack(">H", len(payload) + 2) + payload


def xt_packets(data, chunk_size=97, instance=27):
    header_size = 16 if data[:4] == b"\x00\x00\x00\x01" else 8
    pieces = [data[:chunk_size]]
    pieces.extend(data[i:i + chunk_size] for i in range(chunk_size, len(data), chunk_size))
    return [b"JP" + struct.pack(">HI", instance, i + 1) + (data[:header_size] if i else b"") + piece
            for i, piece in enumerate(pieces)]


def xmp(uri="https://example.invalid/credential.c2pa", creator="Adobe Photoshop"):
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        f'dcterms:provenance="{uri}" xmp:CreatorTool="{creator}"/>'
        '</rdf:RDF></x:xmpmeta>').encode()


class C2PADetectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="unmarker-c2pa-tests-")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.base = self.directory / "base.png"
        Image.new("RGBA", (32, 24), (12, 95, 180, 120)).save(self.base)

    def without_sdk(self, path):
        with patch.dict(sys.modules, {"c2pa": None}):
            return metadata.scan_image(path)

    def report(self, actions=None):
        return {
            "active_manifest": "active", "validation_state": "Valid",
            "validation_results": {"activeManifest": {
                "success": [{"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"}],
                "failure": [{"code": "signingCredential.untrusted"}]}},
            "manifests": {"active": {"claim_generator_info": [{"name": "Adobe Photoshop"}],
                "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions or []}}]}},
        }

    def summarize(self, report):
        result = self.without_sdk(self.base)
        metadata._c2pa_report(report, result)
        return result

    def test_jumd_requestable_and_optional_fields_use_correct_offsets(self):
        for flags, tail in ((3, b""), (7, b"\0\0\0\7"), (11, b"x" * 32), (15, b"\0" * 36)):
            with self.subTest(flags=flags):
                description = metadata._C2PA_UUID + bytes([flags]) + b"c2pa\0" + tail
                self.assertTrue(metadata._c2pa_box(box(b"jumb", box(b"jumd", description))))
        wrong_offset = metadata._C2PA_UUID + b"\x03" + b"\0" * 4 + b"c2pa\0"
        self.assertFalse(metadata._c2pa_box(box(b"jumb", box(b"jumd", wrong_offset))))
        short_id = metadata._C2PA_UUID + b"\x07c2pa\0"
        self.assertFalse(metadata._c2pa_box(box(b"jumb", box(b"jumd", short_id))))

    def test_extended_boxes_and_non_c2pa_jumbf(self):
        description = metadata._C2PA_UUID + b"\x03c2pa\0"
        expected = box(b"jumb", box(b"jumd", description, True), True)
        self.assertTrue(metadata._c2pa_box(expected))
        unrelated = box(b"jumb", box(b"jumd", b"x" * 16 + b"\x03c2pa\0"))
        self.assertFalse(metadata._c2pa_box(unrelated))
        self.assertFalse(metadata._c2pa_box(expected[:-1]))

    def test_fragmented_jpeg_reassembly_without_sdk(self):
        for filename, packet_count in (("CA.jpg", 2), ("CACA.jpg", 4)):
            with self.subTest(filename=filename):
                result = self.without_sdk(FIXTURES / filename)
                credentials = result["metadata"]["c2pa"]
                self.assertTrue(credentials["present"])
                self.assertTrue(credentials["manifest_store_recognized"])
                self.assertEqual(result["metadata"]["jumbf"]["count"], packet_count)
                self.assertEqual(credentials["sdk_status"], "unavailable")
                self.assertFalse(credentials["signature_verified"])

    def test_jpeg_packet_gaps_duplicate_sequences_and_instance_isolation(self):
        descriptor = metadata._C2PA_UUID + b"\x03c2pa\0"
        data = box(b"jumb", box(b"jumd", descriptor) + box(b"free", b"x" * 500))
        packets = xt_packets(data)
        baseline = self.without_sdk(self.base)
        metadata._jpeg_jumbf(packets, baseline)
        self.assertTrue(baseline["metadata"]["c2pa"]["manifest_store_recognized"])
        for broken in (packets[1:], packets[:-1], [packets[0], packets[0], *packets[1:]],
                       [packets[0], *[p[:2] + b"\0\1" + p[4:] for p in packets[1:]]]):
            with self.subTest(packets=len(broken)):
                result = self.without_sdk(self.base)
                metadata._jpeg_jumbf(broken, result)
                self.assertFalse(result["metadata"]["c2pa"]["manifest_store_recognized"])
                self.assertTrue(result["metadata"]["c2pa"]["container_errors"])

    @unittest.skipUnless(HAS_SDK, "c2pa-python is not installed")
    def test_official_reader_finds_fragmented_manifest_and_actual_apps(self):
        for filename, count in (("C.jpg", 1), ("CA.jpg", 1), ("CACA.jpg", 2)):
            with self.subTest(filename=filename):
                result = metadata.scan_image(FIXTURES / filename)
                credentials = result["metadata"]["c2pa"]
                self.assertEqual(credentials["status"], "detected")
                self.assertEqual(credentials["manifest_count"], count)
                self.assertTrue(credentials["signature_verified"])
                self.assertFalse(credentials["validation"]["signer_trusted"])
                self.assertTrue(credentials["validation"]["asset_binding_verified"])
                self.assertTrue(any(item["name"] == "make_test_images" for item in result["recorded_apps"]))
                self.assertEqual(credentials["ai_assertions"], [])  # algorithmicMedia is not trained AI.

    @unittest.skipUnless(HAS_SDK, "c2pa-python is not installed")
    def test_signed_png_webp_declarations_are_not_pixel_detection(self):
        for filename in ("generated-signed.png", "generated-signed.webp"):
            with self.subTest(filename=filename):
                source = FIXTURES / filename
                original = hashlib.sha256(source.read_bytes()).digest()
                result = metadata.scan_image(source)
                credentials = result["metadata"]["c2pa"]
                self.assertEqual(credentials["status"], "detected")
                self.assertTrue(credentials["signature_verified"])
                self.assertTrue(credentials["ai_assertions"])
                declarations = [item for item in credentials["watermark_declarations"] if item["name"] == "SynthID"]
                self.assertEqual(len(declarations), 1)
                self.assertEqual(declarations[0]["scope"], "active_manifest")
                self.assertTrue(declarations[0]["declaration"])
                self.assertFalse(declarations[0]["pixel_detection"])
                self.assertEqual(original, hashlib.sha256(source.read_bytes()).digest())
                with Image.open(source) as image:
                    self.assertEqual(image.getchannel("A").getextrema(), (120, 120))

    def test_plain_names_or_substrings_do_not_create_detection(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "Google Gemini")
        info.add_text("Comment", "c2pa SynthID trainedAlgorithmicMedia 100% detected")
        path = self.directory / "SynthID_c2pa.png"
        Image.new("RGB", (16, 16), "red").save(path, pnginfo=info)
        result = metadata.scan_image(path)
        credentials = result["metadata"]["c2pa"]
        self.assertFalse(credentials["present"])
        self.assertEqual(credentials["ai_assertions"], [])
        self.assertEqual(credentials["watermark_declarations"], [])
        self.assertIn("Google Gemini", [item["name"] for item in result["recorded_apps"]])

    def test_png_text_variants_find_local_remote_references(self):
        for variant in ("itxt", "compressed_itxt", "text", "ztext"):
            with self.subTest(variant=variant):
                info = PngImagePlugin.PngInfo()
                if "itxt" in variant:
                    info.add_itxt("XML:com.adobe.xmp", xmp().decode(), zip=variant.startswith("compressed"))
                else:
                    info.add_text("XML:com.adobe.xmp", xmp().decode(), zip=variant == "ztext")
                path = self.directory / (variant + ".png")
                Image.new("RGB", (16, 16)).save(path, pnginfo=info)
                credentials = metadata.scan_image(path)["metadata"]["c2pa"]
                self.assertFalse(credentials["present"])
                self.assertEqual(credentials["status"], "external_reference")
                self.assertTrue(credentials["external_references"])
                self.assertTrue(all(item["fetched"] is False for item in credentials["external_references"]))

    def test_exif_xmp_jpeg_and_webp_reference_detection(self):
        png = self.directory / "exif-xmp.png"
        exif = Image.Exif()
        exif[700] = xmp()
        Image.new("RGB", (16, 16)).save(png, exif=exif)
        webp = self.directory / "xmp.webp"
        Image.new("RGB", (16, 16)).save(webp, xmp=xmp())
        jpeg = self.directory / "xmp.jpg"
        Image.new("RGB", (16, 16)).save(jpeg)
        raw = jpeg.read_bytes()
        jpeg.write_bytes(raw[:2] + marker(0xe1, b"http://ns.adobe.com/xap/1.0/\0" + xmp()) + raw[2:])
        for source in (png, webp, jpeg):
            with self.subTest(source=source.name):
                result = metadata.scan_image(source)
                self.assertEqual(result["metadata"]["c2pa"]["status"], "external_reference")
                self.assertFalse(result["metadata"]["c2pa"]["present"])
                self.assertIn("Adobe Photoshop", [item["name"] for item in result["recorded_apps"]])

    def test_xmp_requires_exact_namespace_and_limits_expansion(self):
        result = self.without_sdk(self.base)
        metadata._scan_xmp(xmp().replace(b"http://purl.org/dc/terms/", b"http://example.invalid/"), result, "test")
        self.assertEqual(result["metadata"]["c2pa"]["external_references"], [])
        metadata._scan_xmp(b'<!DOCTYPE x [<!ENTITY a "123">]><x>&a;</x>', result, "test")
        with patch.object(metadata, "_MAX_XMP", 128):
            self.assertIsNone(metadata._png_text(b"XML:com.adobe.xmp\0\0" + zlib.compress(b"x" * 512), b"zTXt"))
        self.assertEqual(result["metadata"]["c2pa"]["external_references"], [])

    @unittest.skipUnless(HAS_SDK, "c2pa-python is not installed")
    def test_malformed_c2pa_is_not_reported_as_absent_or_validated(self):
        path = self.directory / "broken.png"
        data = self.base.read_bytes()
        path.write_bytes(data[:33] + png_chunk(b"caBX", b"broken C2PA") + data[33:])
        credentials = metadata.scan_image(path)["metadata"]["c2pa"]
        self.assertTrue(credentials["present"])
        self.assertEqual(credentials["status"], "invalid")
        self.assertFalse(credentials["manifest_store_recognized"])
        self.assertFalse(credentials["signature_verified"])

    def test_editor_created_action_and_unrelated_manifests_are_not_ai(self):
        report = self.report([{"action": "c2pa.created", "description": "SynthID was removed"},
                              {"action": "c2pa.created", "digitalSourceType": "https://example.invalid/trainedAlgorithmicMedia"}])
        report["manifests"]["unrelated"] = {"assertions": [{"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.watermarked", "description": "SynthID watermark inserted"},
            {"action": "c2pa.created", "digitalSourceType": next(iter(metadata._AI_SOURCE_TYPES))}]}}]}
        result = self.summarize(report)
        credentials = result["metadata"]["c2pa"]
        self.assertEqual(credentials["watermark_declarations"], [])
        self.assertEqual(credentials["ai_assertions"], [])
        self.assertEqual(len(credentials["manifests"]), 1)
        self.assertIn("Adobe Photoshop", [item["name"] for item in result["recorded_apps"]])

    def test_real_watermark_action_is_required_and_negation_is_not_positive(self):
        for action, description in (("c2pa.created", "SynthID watermark inserted"),
            ("c2pa.watermarked", "removedSynthID"), ("c2pa.watermarked", "SynthID removed"),
            ("c2pa.watermarked", "without SynthID")):
            with self.subTest(action=action, description=description):
                result = self.summarize(self.report([{"action": action, "description": description}]))
                self.assertFalse(any(item["name"] == "SynthID" for item in result["metadata"]["c2pa"]["watermark_declarations"]))

    def test_ingredient_declaration_scope_and_own_signature_status(self):
        report = self.report()
        report["manifests"]["active"]["ingredients"] = [{"active_manifest": "ingredient", "validation_results": {
            "activeManifest": {"success": [{"code": "claimSignature.validated"}]}}}]
        report["manifests"]["ingredient"] = {"assertions": [{"label": "c2pa.actions.v2", "data": {
            "templates": [{"action": "c2pa.watermarked.unbound", "description": "Watermarked with SynthID"}],
            "actions": [{"action": "c2pa.watermarked.unbound"}]}}]}
        result = self.summarize(report)
        declarations = [item for item in result["metadata"]["c2pa"]["watermark_declarations"] if item["name"] == "SynthID"]
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0]["scope"], "ingredient")
        self.assertEqual(declarations[0]["manifest_label"], "ingredient")
        self.assertTrue(declarations[0]["signature_verified"])
        self.assertFalse(declarations[0]["pixel_detection"])

    def test_claim_signature_failure_cannot_be_overridden_by_success_or_trust(self):
        report = self.report()
        report["validation_results"]["activeManifest"]["failure"].append({"code": "claimSignature.mismatch"})
        result = self.summarize(report)
        credentials = result["metadata"]["c2pa"]
        self.assertTrue(credentials["present"])
        self.assertFalse(credentials["signature_verified"])
        self.assertEqual(credentials["signature_status"], "invalid")

    @unittest.skipUnless(HAS_SDK, "c2pa-python is not installed")
    def test_sdk_uses_stream_and_explicit_offline_context(self):
        import c2pa
        with patch.object(c2pa.Context, "from_dict", wraps=c2pa.Context.from_dict) as context, \
                patch.object(c2pa, "Reader", wraps=c2pa.Reader) as reader:
            result = metadata.scan_image(FIXTURES / "C.jpg")
        settings = context.call_args.args[0]["verify"]
        self.assertFalse(settings["remote_manifest_fetch"])
        self.assertFalse(settings["ocsp_fetch"])
        self.assertTrue(settings["verify_after_reading"])
        self.assertEqual(reader.call_args.args[0], "image/jpeg")
        self.assertTrue(hasattr(reader.call_args.args[1], "read"))
        self.assertEqual(result["metadata"]["c2pa"]["status"], "detected")

    @unittest.skipUnless(HAS_SDK, "c2pa-python is not installed")
    def test_fresh_scan_does_not_import_model_libraries(self):
        script = '''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("metadata_only", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.scan_image(sys.argv[2])
assert result["metadata"]["c2pa"]["status"] == "detected", result
heavy = {"torch", "torchvision", "diffusers", "transformers", "accelerate"}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
json.dumps(result, allow_nan=False)
'''
        run = subprocess.run([sys.executable, "-I", "-B", "-c", script,
            str(ROOT / "metadata_core.py"), str(FIXTURES / "CACA.jpg")], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()

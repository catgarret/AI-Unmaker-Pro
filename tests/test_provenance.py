"""공식 API의 응답 해석과 전송 보안을 검사합니다. 실제 검출 효과 검사는 아닙니다."""
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import provenance as p


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "private-name.png"
        Image.new("RGB", (16, 16), "red").save(self.path)
        self.payload = {"object": "content_provenance_check", "results": [
            {"type": "c2pa", "outcome": "detected", "issuer": "OpenAI"},
            {"type": "synthid", "outcome": "not_detected"}]}

    def connection(self, status=200, raw=None):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = status
        response.read.return_value = raw if raw is not None else json.dumps(self.payload).encode()
        return connection

    def test_consent_and_key_are_required_before_network_or_file_reads(self):
        with patch.object(p.http.client, "HTTPSConnection") as network, patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaises(p.ProvenanceError):
                p.verify_openai_file("does-not-exist")
            with self.assertRaises(p.ProvenanceError):
                p.verify_openai_file("does-not-exist", allow_upload=True)
            network.assert_not_called()

    def test_exact_original_bytes_are_hashed_and_only_official_host_receives_key(self):
        connection = self.connection()
        with patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-only-key", "OPENAI_BASE_URL": "http://wrong.invalid"}), \
                patch.object(p.http.client, "HTTPSConnection", return_value=connection) as network:
            result = p.verify_openai_file(self.path, allow_upload=True)
        self.assertEqual(network.call_args.args, ("api.openai.com",))
        context = network.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        request = connection.request.call_args
        self.assertEqual(request.args, ("POST", "/v1/content_provenance_checks"))
        self.assertNotIn(self.path.name.encode(), request.kwargs["body"])
        self.assertIn(self.path.read_bytes(), request.kwargs["body"])
        self.assertEqual(result["signals"]["synthid"]["detected"], False)
        self.assertEqual(result["signals"]["c2pa"]["detected"], True)
        self.assertNotIn("test-only-key", json.dumps(result))
        connection.close.assert_called_once()

    def test_missing_duplicate_or_unknown_synthid_is_never_negative(self):
        for rows in ([], [{"type": "synthid", "outcome": "error"}],
                     [{"type": "synthid", "outcome": "detected"}] * 2):
            result = p._provenance_signals({"object": "content_provenance_check", "results": rows})
            self.assertIsNone(result["synthid"]["detected"])
            self.assertEqual(result["synthid"]["status"], "not_tested")

    def test_http_errors_and_redirects_are_not_retried_or_recorded_as_negative(self):
        for status in (302, 400, 401, 403, 404, 429, 500):
            with self.subTest(status=status), patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-only-key"}):
                connection = self.connection(status, b'{"error":"secret-server-data"}')
                with patch.object(p.http.client, "HTTPSConnection", return_value=connection) as network:
                    with self.assertRaises(p.ProvenanceError) as caught:
                        p.verify_openai_file(self.path, allow_upload=True)
                network.assert_called_once()
                connection.request.assert_called_once()
                self.assertNotIn("secret-server-data", str(caught.exception))

    def test_network_errors_cannot_leak_keys_or_request_data(self):
        with patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-only-key"}), \
                patch.object(p.http.client, "HTTPSConnection", side_effect=OSError("test-only-key")):
            with self.assertRaises(p.ProvenanceError) as caught:
                p.verify_openai_file(self.path, allow_upload=True)
        self.assertNotIn("test-only-key", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_invalid_format_and_oversize_are_rejected_before_upload(self):
        with patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-only-key"}), \
                patch.object(p.http.client, "HTTPSConnection") as network:
            self.path.write_text("not an image")
            with self.assertRaises(p.ProvenanceError):
                p.verify_openai_file(self.path, allow_upload=True)
            with patch.object(p, "_PROVENANCE_MAX_FILE", 2):
                with self.assertRaises(p.ProvenanceError):
                    p.verify_openai_file(self.path, allow_upload=True)
            network.assert_not_called()

    def test_bad_and_oversize_response_fails_closed(self):
        for body in (b"not JSON", b"x" * (p._PROVENANCE_MAX_RESPONSE + 1), b'{}'):
            with patch.dict(p.os.environ, {"OPENAI_API_KEY": "test-only-key"}), \
                    patch.object(p.http.client, "HTTPSConnection", return_value=self.connection(raw=body)):
                with self.assertRaises(p.ProvenanceError):
                    p.verify_openai_file(self.path, allow_upload=True)


if __name__ == "__main__":
    unittest.main()

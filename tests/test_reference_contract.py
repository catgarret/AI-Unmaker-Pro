"""기준 처리 조건과 TLS 보안의 회귀검사입니다. 제거 효과 검사는 아닙니다."""
import ssl
import unittest
from unittest.mock import patch

import model_runtime as runtime
import qwen_inference as qwen


class ReferenceContractTests(unittest.TestCase):
    def test_default_does_not_silently_change_reference_resolution_or_seed(self):
        with patch.object(qwen.secrets, "randbits", return_value=912345) as random_seed:
            options = qwen._qwen_options({})
        self.assertIsNone(options["max_working_side"])
        self.assertEqual(options["seed"], 912345)
        self.assertEqual(options["steps"] * options["strength"], 2)
        self.assertTrue(options["restore"])
        random_seed.assert_called_once_with(63)

    def test_fixed_seed_and_no_restore_are_explicit_options(self):
        with patch.object(qwen.secrets, "randbits", side_effect=AssertionError("Fixed seed must be reproducible")):
            options = qwen._qwen_options({"seed": 42, "restore": False})
        self.assertEqual(options["seed"], 42)
        self.assertFalse(options["restore"])

    def test_quant_filename_has_one_source_of_truth(self):
        self.assertEqual(qwen._QWEN_GGUF_FILENAME, runtime.MODEL_SPECS["qwen"]["files"][0])
        self.assertIn("Q4_K_M", qwen._QWEN_GGUF_FILENAME)

    def test_tls_still_requires_a_valid_certificate_and_hostname(self):
        context = runtime._model_tls_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_http_redirect_is_rejected(self):
        with self.assertRaises(runtime.ModelRuntimeError):
            runtime._PublicModelRedirect().redirect_request(None, None, 302, "", {}, "http://example.com/model")

    def test_missing_security_dependency_is_reported(self):
        with patch.dict("sys.modules", {"truststore": None}):
            with self.assertRaises(runtime.ModelRuntimeError):
                runtime._model_tls_context()


if __name__ == "__main__":
    unittest.main()

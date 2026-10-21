"""Offline byte-transfer/cache regressions for the Qwen download migration.

Responses and tiny payloads below are explicit HTTP doubles, not model weights.
No test downloads production weights, runs inference, or uploads an image.
"""
from contextlib import ExitStack, contextmanager
import copy
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unmarker_qwen_download_tests", ROOT / "model_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class ByteResponse:
    """Small offline HTTP body that can raise a timeout after its first chunk."""

    def __init__(self, payload, *, offset=0, total=None, interrupt=False):
        self.body = io.BytesIO(payload)
        self.offset = offset
        self.interrupt = interrupt
        self.read_count = 0
        total = len(payload) if total is None else total
        self.headers = {"Content-Length": str(total - offset)}
        if offset:
            self.headers["Content-Range"] = f"bytes {offset}-{total - 1}/{total}"

    def getcode(self):
        return 206 if self.offset else 200

    def read(self, amount):
        self.read_count += 1
        if self.interrupt:
            if self.read_count > 1:
                raise TimeoutError("OFFLINE injected disconnect")
            amount = min(amount, 3)
        return self.body.read(amount)

    read1 = read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.body.close()


def metadata(payload):
    return {"size": len(payload), "algorithm": "sha256", "digest": hashlib.sha256(payload).hexdigest()}


class QwenManifestTests(unittest.TestCase):
    def test_pinned_default_is_one_q4_quant_with_all_required_assets(self):
        self.assertEqual(list(runtime.MODEL_SPECS), ["qwen", "lightning", "vae", "embeddings"])
        self.assertEqual(runtime.MODEL_SPECS["qwen"]["files"], ("Qwen-Image-2512-Q4_K_M.gguf",))
        self.assertIn("gguf", runtime._RUNTIME_MODULES)
        self.assertIn("peft", runtime._RUNTIME_MODULES)
        total = 0
        for spec in runtime.MODEL_SPECS.values():
            self.assertRegex(spec["revision"], r"^[a-f0-9]{40}$")
            for filename in spec["files"]:
                item = runtime._file_identity(spec, filename)
                self.assertEqual(item["algorithm"], "sha256")
                self.assertRegex(item["digest"], r"^[a-f0-9]{64}$")
                total += item["size"]
        self.assertEqual(total, 14_943_745_074)
        embeddings = runtime.MODEL_SPECS["embeddings"]
        self.assertIn(embeddings["revision"], embeddings["source_urls"]["embeds_cache.pt"])
        self.assertTrue(embeddings["source_urls"]["embeds_cache.pt"].startswith(
            "https://raw.githubusercontent.com/0xROOTPLS/DeSynth/"))

    def test_empty_cache_estimate_is_offline_and_does_not_remove_old_files(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"UNMARK_MODEL_CACHE": folder}), \
                patch.object(runtime, "_open_model_stream", side_effect=AssertionError("network forbidden")):
            root = Path(folder)
            old = root / "old-sd-weights.bin"
            old.write_bytes(b"KEEP PREVIOUS MODEL CACHE")
            (root / "ready.json").write_text(json.dumps({"base": {"repo_id": "old/sd", "path": str(old)}}))
            estimate = runtime.setup_estimate()
            self.assertEqual(estimate["total_bytes"], 14_943_745_074)
            self.assertEqual(estimate["remaining_bytes"], 14_943_745_074)
            self.assertFalse(estimate["cached"])
            self.assertTrue(estimate["needs_download"])
            self.assertEqual(old.read_bytes(), b"KEEP PREVIOUS MODEL CACHE")


class RawSourceTransferTests(unittest.TestCase):
    def setUp(self):
        self.payload = b"OFFLINE PINNED PROMPT CACHE BYTE FIXTURE"
        self.spec = {"repo_id": "fixture/embeddings", "revision": "a" * 40, "files": ("embeds_cache.pt",)}
        self.url = "https://raw.githubusercontent.com/fixture/embeddings/" + "a" * 40 + "/embeds_cache.pt"

    def test_raw_source_resumes_after_timeout_and_verifies_the_complete_hash(self):
        events = []
        calls = []

        def open_stream(url, headers):
            calls.append((url, dict(headers)))
            if len(calls) == 1:
                return ByteResponse(self.payload, interrupt=True)
            return ByteResponse(self.payload[3:], offset=3, total=len(self.payload))

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runtime, "MODEL_FILE_METADATA", {self.spec["repo_id"]: {"embeds_cache.pt": metadata(self.payload)}}), \
                patch.object(runtime, "_open_model_stream", side_effect=open_stream), \
                patch.object(runtime, "_wait_model_download_retry"), \
                patch.dict(sys.modules, {"huggingface_hub": None}):
            target = Path(folder) / "embeds_cache.pt"
            progress = runtime._DownloadProgress(events.append, len(self.payload))
            runtime._download_one(target, self.spec, "embeds_cache.pt", progress, "download", None, source_url=self.url)
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(calls, [(self.url, {}), (self.url, {"Range": "bytes=3-"})])
            retry = next(event for event in events if event["phase"] == "retry")
            self.assertEqual(retry["downloaded_bytes"], 3)
            self.assertEqual(retry["bytes_per_second"], 0)
            self.assertIsNone(retry["eta_seconds"])
            self.assertTrue(all(event["progress"] < 1 for event in events))
            for path in runtime._partial_paths(target):
                self.assertFalse(path.exists())

    def test_wrong_hash_is_never_published_even_when_size_matches(self):
        wrong = b"X" * len(self.payload)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runtime, "MODEL_FILE_METADATA", {self.spec["repo_id"]: {"embeds_cache.pt": metadata(self.payload)}}), \
                patch.object(runtime, "_open_model_stream", side_effect=lambda *_: ByteResponse(wrong)) as opened:
            target = Path(folder) / "embeds_cache.pt"
            with self.assertRaises(runtime._DownloadIntegrityError):
                runtime._download_one(target, self.spec, "embeds_cache.pt", runtime._DownloadProgress(None, len(wrong)),
                                      "download", None, source_url=self.url)
            self.assertEqual(opened.call_count, 2)
            self.assertFalse(target.exists())
            for path in runtime._partial_paths(target):
                self.assertFalse(path.exists())


class FourGroupCacheTests(unittest.TestCase):
    def test_prepare_preserves_old_cache_reuses_verified_files_and_repairs_only_corruption(self):
        specs = {}
        meta = {}
        payloads = {}
        for key in ("qwen", "lightning", "vae", "embeddings"):
            repo = "fixture/" + key
            filename = key + ".bin"
            payload = ("OFFLINE fixture for " + key).encode("ascii")
            url = "https://example.invalid/" + key
            specs[key] = {"repo_id": repo, "revision": "b" * 40, "files": (filename,),
                          "source_urls": {filename: url}}
            meta[repo] = {filename: metadata(payload)}
            payloads[url] = payload
        events = []
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"UNMARK_MODEL_CACHE": folder}))
            stack.enter_context(patch.object(runtime, "MODEL_SPECS", specs))
            stack.enter_context(patch.object(runtime, "MODEL_FILE_METADATA", meta))
            stack.enter_context(patch.object(runtime.importlib.util, "find_spec", return_value=SimpleNamespace()))
            opened = stack.enter_context(patch.object(runtime, "_open_model_stream",
                                                      side_effect=lambda url, _: ByteResponse(payloads[url])))
            root = Path(folder)
            old = root / "hub" / "models--old--sd" / "snapshots" / "old" / "weight.bin"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"PRESERVE")
            old_entry = {"repo_id": "old/sd", "revision": "old", "path": str(old.parent), "verified_files": {}}
            (root / "ready.json").write_text(json.dumps({"base": old_entry}))
            paths = runtime.prepare_models(events.append)
            self.assertEqual(set(paths), set(specs))
            self.assertEqual(opened.call_count, 4)
            self.assertEqual(runtime._cached_paths(), paths)
            self.assertTrue(runtime.model_runtime_status()["cached"])
            self.assertEqual(json.loads((root / "ready.json").read_text())["base"], old_entry)
            self.assertEqual(old.read_bytes(), b"PRESERVE")
            self.assertEqual(events[-1]["phase"], "ready")
            self.assertEqual(events[-1]["progress"], 1)
            opened.reset_mock()
            self.assertEqual(runtime.prepare_models(), paths)
            opened.assert_not_called()

            # Same-sized damage bypasses size-only checks but must invalidate
            # the recorded fingerprint and be caught by SHA-256 before reuse.
            prompt = Path(paths["embeddings"]) / "embeddings.bin"
            prompt.write_bytes(b"X" * prompt.stat().st_size)
            self.assertFalse(runtime._cached_paths())
            repaired = runtime.prepare_models()
            self.assertEqual(repaired, paths)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(opened.call_args.args[0], "https://example.invalid/embeddings")
            self.assertEqual(prompt.read_bytes(), payloads["https://example.invalid/embeddings"])
            self.assertEqual(old.read_bytes(), b"PRESERVE")


@unittest.skipUnless(os.name == "posix", "Safe migration requires anchored POSIX directory operations")
class LegacyCacheMigrationTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, replacement_ready=True):
        """Small owned-cache doubles; no production cache or model is used."""
        legacy = copy.deepcopy(runtime._LEGACY_MODEL_SPECS)
        current = {}
        identities = {}
        with tempfile.TemporaryDirectory(prefix="unmarker-legacy-fixture-") as directory, ExitStack() as stack:
            parent = Path(directory).resolve()
            root = parent / "cache"
            root.mkdir()
            stack.enter_context(patch.dict(os.environ, {"UNMARK_MODEL_CACHE": str(root)}))
            manifest, old_paths, new_paths = {}, [], []
            for key, spec in legacy.items():
                path = root / "hub" / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]
                verified = {}
                for name in spec["files"]:
                    payload = ("OFFLINE OLD FILE " + key + "/" + name).encode()
                    target = path / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    identity = metadata(payload)
                    spec["files"][name] = (len(payload), identity["digest"])
                    verified[name] = {"fingerprint": runtime._fingerprint(target), "digest": identity["digest"]}
                    old_paths.append(target)
                manifest[key] = {"repo_id": spec["repo_id"], "revision": spec["revision"],
                                 "path": str(path), "verified_files": verified}
            for key in ("qwen", "lightning", "vae", "embeddings"):
                repo, revision, filename = "offline/" + key, "c" * 40, key + ".bin"
                payload = ("OFFLINE NEW FILE " + key).encode()
                current[key] = {"repo_id": repo, "revision": revision, "files": (filename,)}
                identities[repo] = {filename: metadata(payload)}
                if replacement_ready:
                    path = root / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
                    path.mkdir(parents=True)
                    target = path / filename
                    target.write_bytes(payload)
                    new_paths.append(target)
                    manifest[key] = {"repo_id": repo, "revision": revision, "path": str(path),
                                     "verified_files": {filename: {"fingerprint": runtime._fingerprint(target),
                                                                   "digest": identities[repo][filename]["digest"]}}}
            manifest["unrelated"] = {"note": "keep all unrelated manifest data", "arbitrary": [1, 2, 3]}
            ready = root / "ready.json"
            ready.write_text(json.dumps(manifest))
            stack.enter_context(patch.object(runtime, "_LEGACY_MODEL_SPECS", legacy))
            stack.enter_context(patch.object(runtime, "MODEL_SPECS", current))
            stack.enter_context(patch.object(runtime, "MODEL_FILE_METADATA", identities))
            stack.enter_context(patch.object(runtime, "_open_model_stream", side_effect=AssertionError("Network forbidden")))
            yield SimpleNamespace(root=root, parent=parent, ready=ready, manifest=manifest,
                                  old_paths=old_paths, new_paths=new_paths, legacy=legacy,
                                  total=sum(path.stat().st_size for path in old_paths))

    def test_allowlist_is_exactly_the_twenty_pinned_v16_files(self):
        legacy = runtime._LEGACY_MODEL_SPECS
        self.assertEqual(set(legacy), {"base", "controlnet", "ip_adapter"})
        self.assertEqual(sum(len(spec["files"]) for spec in legacy.values()), 20)
        self.assertEqual(sum(size for spec in legacy.values() for size, _ in spec["files"].values()), 9_500_827_096)
        for spec in legacy.values():
            self.assertRegex(spec["revision"], r"^[a-f0-9]{40}$")
            for name, (_, digest) in spec["files"].items():
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn("..", Path(name).parts)
                self.assertRegex(digest, r"^[a-f0-9]{40}([a-f0-9]{24})?$")

    def test_summary_works_before_new_models_and_prune_refuses(self):
        with self.fixture(replacement_ready=False) as f:
            before = f.ready.read_bytes()
            summary = runtime.legacy_model_cache_summary()
            self.assertEqual(summary["eligible_bytes"], f.total)
            self.assertEqual(len(summary["eligible_files"]), 20)
            self.assertFalse(summary["errors"])
            self.assertEqual(runtime.setup_estimate()["legacy_cache_bytes"], f.total)
            report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["status"], "not_ready")
            self.assertEqual(report["removed_bytes"], 0)
            self.assertTrue(all(path.exists() for path in f.old_paths))
            self.assertEqual(f.ready.read_bytes(), before)

    def test_relative_cache_root_is_refused_without_reading_or_removing_files(self):
        with patch.object(runtime, "_cache_root", return_value=Path("relative-cache")), \
                patch.object(runtime, "_legacy_directory", side_effect=AssertionError("Do not open relative cache")):
            summary = runtime.legacy_model_cache_summary()
            self.assertEqual(summary["eligible_bytes"], 0)
            self.assertEqual(summary["errors"], [{"reason": "cache_root_not_absolute"}])
            report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["removed_bytes"], 0)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["errors"], [{"reason": "cache_root_not_absolute"}])

    def test_only_owned_files_are_removed_and_new_cache_stays_ready(self):
        with self.fixture() as f:
            snapshots = Path(f.manifest["base"]["path"])
            untouched = [snapshots / "user-image.png", snapshots / "weight.unmark.part",
                         snapshots / "weight.unmark.part.json", snapshots / "font.ttf",
                         snapshots.parents[1] / "blobs" / "shared-blob"]
            for path in untouched:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"PRESERVE UNTRACKED FILE")
            new_before = {path: path.read_bytes() for path in f.new_paths}
            report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["status"], "completed", report)
            self.assertEqual(report["removed_bytes"], f.total)
            self.assertEqual(len(report["removed_files"]), 20)
            self.assertFalse(any(path.exists() for path in f.old_paths))
            self.assertTrue(all(path.read_bytes() == b"PRESERVE UNTRACKED FILE" for path in untouched))
            self.assertEqual(new_before, {path: path.read_bytes() for path in f.new_paths})
            self.assertEqual(len(runtime._cached_paths()), 4)
            remaining = json.loads(f.ready.read_text())
            self.assertEqual(set(remaining), {"qwen", "lightning", "vae", "embeddings", "unrelated"})
            self.assertEqual(remaining["unrelated"], f.manifest["unrelated"])
            self.assertEqual(runtime.prune_legacy_model_cache()["removed_bytes"], 0)

    def test_changed_unverified_symlink_and_hardlinked_files_are_skipped(self):
        with self.fixture() as f:
            changed, unverified, linked, shared, untracked = f.old_paths[:5]
            changed.write_bytes(b"X" * changed.stat().st_size)
            base = f.manifest["base"]
            base["verified_files"][str(unverified.relative_to(base["path"]))]["digest"] = "0" * 64
            outside = f.parent / "outside-data"
            outside.write_bytes(linked.read_bytes())
            linked.unlink()
            linked.symlink_to(outside)
            base["verified_files"][str(linked.relative_to(base["path"]))]["fingerprint"] = runtime._fingerprint(outside)
            os.link(shared, f.parent / "external-hardlink")
            base["verified_files"][str(shared.relative_to(base["path"]))]["fingerprint"] = runtime._fingerprint(shared)
            del base["verified_files"][str(untracked.relative_to(base["path"]))]
            f.ready.write_text(json.dumps(f.manifest))
            summary = runtime.legacy_model_cache_summary()
            expected = f.total - sum(path.stat().st_size for path in (changed, unverified, linked, shared, untracked))
            self.assertEqual(summary["eligible_bytes"], expected)
            report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["removed_bytes"], expected)
            reasons = {item["reason"] for item in report["skipped"]}
            self.assertTrue({"file_changed", "unverified_file", "not_regular_file", "shared_hardlink"} <= reasons)
            self.assertTrue(all(path.exists() for path in (changed, unverified, linked, shared, untracked, outside)))
            self.assertTrue(linked.is_symlink())
            self.assertEqual(len(runtime._cached_paths()), 4)

    def test_outside_manifest_paths_directory_links_and_shared_snapshots_are_skipped(self):
        for problem in ("outside", "parent_symlink", "shared_entry"):
            with self.subTest(problem=problem), self.fixture() as f:
                snapshot = Path(f.manifest["base"]["path"])
                base_paths = [path for path in f.old_paths if path.is_relative_to(snapshot)]
                base_bytes = sum(path.stat().st_size for path in base_paths)
                if problem == "outside":
                    f.manifest["base"]["path"] = str(f.parent / "outside")
                elif problem == "parent_symlink":
                    moved = f.parent / "old-model-outside-cache"
                    snapshot.rename(moved)
                    snapshot.symlink_to(moved, target_is_directory=True)
                else:
                    f.manifest["another-consumer"] = {"path": str(snapshot)}
                f.ready.write_text(json.dumps(f.manifest))
                report = runtime.prune_legacy_model_cache()
                self.assertEqual(report["removed_bytes"], f.total - base_bytes)
                self.assertTrue(all(path.exists() for path in base_paths))
                self.assertEqual(len(runtime._cached_paths()), 4)

    def test_partial_unlink_failure_preserves_remaining_manifest_and_can_retry(self):
        with self.fixture() as f:
            failed = f.old_paths[0]
            failed_bytes = failed.stat().st_size
            original_unlink = os.unlink

            def unlink(path, *args, **kwargs):
                if path == failed.name:
                    raise PermissionError(errno.EACCES, "OFFLINE injected file lock")
                return original_unlink(path, *args, **kwargs)

            with patch.object(runtime, "_legacy_secure_paths_supported", return_value=True), \
                    patch.object(runtime.os, "unlink", side_effect=unlink):
                report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["removed_bytes"], f.total - failed_bytes)
            self.assertEqual(len(report["errors"]), 1)
            self.assertTrue(failed.exists())
            self.assertEqual(len(runtime._cached_paths()), 4)
            remaining = json.loads(f.ready.read_text())
            self.assertEqual(set(remaining["base"]["verified_files"]), {failed.name})
            self.assertEqual(runtime.prune_legacy_model_cache()["removed_bytes"], failed_bytes)

    def test_cancellation_and_manifest_write_failure_return_diagnostics(self):
        with self.fixture() as f:
            event = threading.Event()
            event.set()
            self.assertTrue(runtime.prune_legacy_model_cache(event)["cancelled"])
            self.assertTrue(all(path.exists() for path in f.old_paths))
            event.clear()
            original_unlink = os.unlink

            def unlink(path, *args, **kwargs):
                result = original_unlink(path, *args, **kwargs)
                if path == f.old_paths[0].name:
                    event.set()
                return result

            first_size = f.old_paths[0].stat().st_size
            with patch.object(runtime, "_legacy_secure_paths_supported", return_value=True), \
                    patch.object(runtime.os, "unlink", side_effect=unlink):
                report = runtime.prune_legacy_model_cache(event)
            self.assertEqual(report["status"], "cancelled")
            self.assertEqual(report["removed_bytes"], first_size)
            self.assertTrue(all(path.exists() for path in f.old_paths[1:]))
            self.assertEqual(len(runtime._cached_paths()), 4)
            with patch.object(runtime, "_legacy_commit_removals", side_effect=OSError("OFFLINE read-only manifest")):
                report = runtime.prune_legacy_model_cache()
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["removed_bytes"], f.total - first_size)
            self.assertEqual(report["errors"][-1]["reason"], "manifest_update_failed")
            self.assertEqual(len(runtime._cached_paths()), 4)

    def test_file_changed_after_inventory_and_concurrent_manifest_entries_are_preserved(self):
        with self.fixture() as f:
            original_candidates = runtime._legacy_candidates
            changed = f.old_paths[0]

            def inventory(*args):
                result = original_candidates(*args)
                changed.write_bytes(b"Z" * changed.stat().st_size)
                return result

            with patch.object(runtime, "_legacy_candidates", side_effect=inventory):
                report = runtime.prune_legacy_model_cache()
            self.assertTrue(changed.exists())
            self.assertIn("file_changed_before_delete", {item["reason"] for item in report["skipped"]})
        with self.fixture() as f:
            original_unlink = os.unlink

            def unlink(path, *args, **kwargs):
                result = original_unlink(path, *args, **kwargs)
                if path == f.old_paths[0].name:
                    manifest = json.loads(f.ready.read_text())
                    manifest["added-during-cleanup"] = {"keep": True}
                    f.ready.write_text(json.dumps(manifest))
                return result

            with patch.object(runtime, "_legacy_secure_paths_supported", return_value=True), \
                    patch.object(runtime.os, "unlink", side_effect=unlink):
                report = runtime.prune_legacy_model_cache()
            self.assertEqual(len(report["removed_files"]), 1)
            self.assertTrue(json.loads(f.ready.read_text())["added-during-cleanup"]["keep"])
            self.assertTrue(all(path.exists() for path in f.old_paths[1:]))


if __name__ == "__main__":
    unittest.main()

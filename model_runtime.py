import collections
import contextlib
import email.utils
import errno
import gc
import http.client
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
from pathlib import Path
import socket
import ssl
import stat
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


# Qwen-Image-2512 + Lightning uses one GGUF quant, not both variants.
# DeSynth's reference default is Q4_K_M. Do not silently substitute Q2:
# changing quantization changes the generated pixels and invalidates comparisons.
MODEL_SPECS = {'qwen': {'requested_id': 'Frederic75/Qwen-Image-2512-GGUF',
          'repo_id': 'Frederic75/Qwen-Image-2512-GGUF',
          'revision': 'd9a634a44d81f9280323f862b7a277f4f4e88d85',
          'files': ('Qwen-Image-2512-Q4_K_M.gguf',)},
 'lightning': {'requested_id': 'lightx2v/Qwen-Image-2512-Lightning',
               'repo_id': 'lightx2v/Qwen-Image-2512-Lightning',
               'revision': '4609161e308ce82018a42c2a0c9b107e865bb751',
               'files': ('Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors',)},
 'vae': {'requested_id': 'Qwen/Qwen-Image-2512',
         'repo_id': 'Qwen/Qwen-Image-2512',
         'revision': '25468b98e3276ca6700de15c6628e51b7de54a26',
         'files': ('model_index.json',
                   'scheduler/scheduler_config.json',
                   'transformer/config.json',
                   'vae/config.json',
                   'vae/diffusion_pytorch_model.safetensors')},
 'embeddings': {'requested_id': '0xROOTPLS/DeSynth',
                'repo_id': '0xROOTPLS/DeSynth',
                'revision': '0c2b14fa6b92fff3721877838fcb9de78fdec0ff',
                'files': ('embeds_cache.pt',),
                'source_urls': {'embeds_cache.pt': 'https://raw.githubusercontent.com/0xROOTPLS/DeSynth/0c2b14fa6b92fff3721877838fcb9de78fdec0ff/embeds_cache.pt'}}}

# Pinned public model metadata checked 2026-10-20. Large-file digests are
# the official HF LFS SHA-256 values; config and prompt-cache digests were
# independently computed from the exact fixed-revision bytes.
MODEL_FILE_METADATA = {'Frederic75/Qwen-Image-2512-GGUF': {'Qwen-Image-2512-Q4_K_M.gguf': {'size': 12990544416,
                                                                     'algorithm': 'sha256',
                                                                     'digest': '714fd9e69d23af50a2f1f93e68b6f7fd2602a76b0692ed1769c443302938695f'}},
 'lightx2v/Qwen-Image-2512-Lightning': {'Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors': {'size': 1698951104,
                                                                                                   'algorithm': 'sha256',
                                                                                                   'digest': 'ad12117461cb41e2ea637fec8df6392ce8e8550c47fbe2b829ed3deb98262066'}},
 'Qwen/Qwen-Image-2512': {'vae/diffusion_pytorch_model.safetensors': {'size': 253806966,
                                                                      'algorithm': 'sha256',
                                                                      'digest': '0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344'},
                          'model_index.json': {'size': 443,
                                               'algorithm': 'sha256',
                                               'digest': 'caf5f49ee3d897fff6a8e8d6353b084c8d1da3e40ba2f32e9bc554eb0ee4b4e9'},
                          'scheduler/scheduler_config.json': {'size': 485,
                                                              'algorithm': 'sha256',
                                                              'digest': '7ee767e37bae4af31d4eb935e125bb20a2237eeecafd22af6610093865c6f587'},
                          'transformer/config.json': {'size': 339,
                                                      'algorithm': 'sha256',
                                                      'digest': '247c94f9b923a7d6b6035ca76d27b556bd0efd1a8c7672f31f037644fcce340e'},
                          'vae/config.json': {'size': 730,
                                              'algorithm': 'sha256',
                                              'digest': 'c448160dba5ce79c965cb075ee02e18d1c42eb6424f787e5869790d577b56a65'}},
 '0xROOTPLS/DeSynth': {'embeds_cache.pt': {'size': 440591,
                                           'algorithm': 'sha256',
                                           'digest': '1d05bbe05797ed9ec1653e935ba257a016c017b50f0ba206394c04782da8cd4c'}}}

# Only these v16 pinned, app-verified files can be retired after a successful
# Qwen output. This is an allowlist, never a directory or HF blob-cache purge.
_LEGACY_MODEL_SPECS = {'base': {'repo_id': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
          'revision': '451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
          'files': {'model_index.json': (541, 'daf7e2e2dfc64fb437a2b44525667111b00cb9fc'),
                    'scheduler/scheduler_config.json': (308, '82d05b0e688d7ea94675678646c427907419346e'),
                    'tokenizer/merges.txt': (524619, '76e821f1b6f0a9709293c3b6b51ed90980b3166b'),
                    'tokenizer/vocab.json': (1059962, '469be27c5c010538f845f518c4f5e8574c78f7c8'),
                    'tokenizer/tokenizer_config.json': (806, '5ba7bf706515bc60487ad0e1816b4929b82542d6'),
                    'tokenizer/special_tokens_map.json': (472,
                                                          '2c2130b544c0c5a72d5d00da071ba130a9800fb2'),
                    'text_encoder/config.json': (617, '4d3e873ab5086ad989f407abd50fdce66db8d657'),
                    'text_encoder/model.safetensors': (492265874,
                                                       'd008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334'),
                    'unet/config.json': (743, '1a02ee8abc93e840ffbcb2d68b66ccbcb74b3ab3'),
                    'unet/diffusion_pytorch_model.safetensors': (3438167540,
                                                                 '19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1'),
                    'vae/config.json': (547, '55d78924fee13e4220f24320127c5f16284e13b9'),
                    'vae/diffusion_pytorch_model.safetensors': (334643276,
                                                                'a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c'),
                    'feature_extractor/preprocessor_config.json': (342,
                                                                   '5294955ff7801083f720b34b55d0f1f51313c5c5'),
                    'safety_checker/config.json': (4723, '5dbd88952e7e521aa665e5052e6db7def3641d03'),
                    'safety_checker/model.safetensors': (1215981830,
                                                         '9d6a233ff6fd5ccb9f76fd99618d73369c52dd3d8222376384d0e601911089e8')}},
 'controlnet': {'repo_id': 'lllyasviel/control_v11p_sd15_canny',
                'revision': '115a470d547982438f70198e353a921996e2e819',
                'files': {'config.json': (996, 'ebd8387e9b5a9c67032880f7799042436ef94efb'),
                          'diffusion_pytorch_model.safetensors': (1445157124,
                                                                  'be713fb941fc7c625f0c7d816b6a19115783a665f3049a8974f127e0c075d9a9')}},
 'ip_adapter': {'repo_id': 'h94/IP-Adapter',
                'revision': '018e402774aeeddd60609b4ecdb7e298259dc729',
                'files': {'models/ip-adapter_sd15.safetensors': (44642768,
                                                                 '289b45f16d043d0bf542e45831f971dcdaabe18b656f11e86d9dfba7e9ee3369'),
                          'models/image_encoder/config.json': (560,
                                                               '9c841dd0314e27141ed01a24df1734e407a95c4a'),
                          'models/image_encoder/model.safetensors': (2528373448,
                                                                     '6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030')}}}

_RUNTIME_MODULES = ("torch", "diffusers", "transformers", "accelerate",
                    "huggingface_hub", "safetensors", "gguf", "peft", "PIL", "numpy", "cv2", "tqdm", "psutil", "truststore")
# One initial request plus at most five transient-network retries per file.
# urllib applies this socket timeout to connect/TLS handshake and body reads.
_MODEL_DOWNLOAD_TIMEOUT = 60
_MODEL_DOWNLOAD_MAX_RETRIES = 5
_MODEL_DOWNLOAD_RETRY_DELAYS = (2, 4, 8, 16, 30)
_MODEL_DOWNLOAD_MAX_RETRY_AFTER = 300
_MODEL_DOWNLOAD_RETRY_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MODEL_LOCK = threading.RLock()
# MODEL_EXECUTION_POLICY and inference are supplied by qwen_inference.py.


class ModelRuntimeError(RuntimeError):
    pass


class ModelCancelled(ModelRuntimeError):
    pass


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ModelCancelled("모델 작업을 취소했습니다.")


def _notify(callback, title, detail=""):
    if callback is not None:
        callback(title, detail)


def _cache_root():
    override = os.environ.get("UNMARK_MODEL_CACHE")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "UNMARK" / "models"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "UNMARK" / "models"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "unmark" / "models"
    root = root.resolve()
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle and root.is_relative_to(Path(bundle).resolve()):
        raise ModelRuntimeError("모델 캐시는 실행파일 번들 밖의 사용자 폴더여야 합니다.")
    return root


def _valid_weight(path):
    """Reject empty/LFS-pointer/truncated files without reading tensor payloads."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 2 <= header_size <= min(16 * 1024 * 1024, size - 8):
                return False
            header = json.loads(handle.read(header_size))
        offsets = [v["data_offsets"] for k, v in header.items() if k != "__metadata__"]
        return bool(offsets) and max(int(v[1]) for v in offsets) == size - header_size - 8
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, struct.error):
        return False


def _invalid_snapshot_files(path, spec):
    """Return only missing/corrupt required files for a bounded repair."""
    invalid = []
    for relative in spec["files"]:
        file = path / relative
        try:
            expected = MODEL_FILE_METADATA.get(spec["repo_id"], {}).get(relative, {}).get("size")
            if not file.is_file() or file.stat().st_size == 0 or (expected is not None and file.stat().st_size != expected):
                invalid.append(relative)
                continue
            if relative.endswith(".safetensors") and not _valid_weight(file):
                invalid.append(relative)
                continue
            if relative.endswith(".json"):
                json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            invalid.append(relative)
    return invalid


def _valid_snapshot(path, spec):
    return not _invalid_snapshot_files(path, spec)


def _cached_paths():
    root = _cache_root()
    try:
        manifest = json.loads((root / "ready.json").read_text(encoding="utf-8"))
        paths = {}
        for key, spec in MODEL_SPECS.items():
            item = manifest[key]
            path = Path(item["path"])
            if (item["repo_id"] != spec["repo_id"] or item.get("revision") != spec["revision"]
                    or not path.resolve().is_relative_to(root / "hub")):
                return {}
            if not _valid_snapshot(path, spec):
                return {}
            verified_files = item.get("verified_files")
            if not isinstance(verified_files, dict):
                # Older manifests only recorded paths. Let prepare_models()
                # verify their existing files before they unlock the app.
                return {}
            for relative in spec["files"]:
                verified = verified_files.get(relative)
                identity = _file_identity(spec, relative)
                if (not isinstance(verified, dict)
                        or verified.get("digest") != identity["digest"]
                        or verified.get("fingerprint") != _fingerprint(path / relative)):
                    return {}
            paths[key] = str(path)
        return paths
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def model_runtime_status():
    missing = [name for name in _RUNTIME_MODULES if importlib.util.find_spec(name) is None]
    paths = _cached_paths()
    policy = globals().get("MODEL_EXECUTION_POLICY", {})
    return {"installed": not missing, "missing": missing, "cached": len(paths) == len(MODEL_SPECS),
            "modelpaths": paths, "cache_dir": str(_cache_root()),
            "models": {key: spec["repo_id"] for key, spec in MODEL_SPECS.items()},
            "model_variant": "Q4_K_M",
            "cuda_memory_policy": policy.get("cuda_memory_policy"),
            "mps_memory_policy": policy.get("mps_memory_policy"),
            "execution_policy": dict(policy),
            "detail": "Qwen-Image-2512 Q4_K_M · Lightning · 이미지 재생성과 디테일 복원"}


def _snapshot_path(spec):
    """Use the documented HF cache snapshot layout, including old cache files."""
    canonical = _cache_root() / "hub" / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]
    try:
        manifest = json.loads((_cache_root() / "ready.json").read_text(encoding="utf-8"))
        item = next(item for item in manifest.values() if item.get("repo_id") == spec["repo_id"]
                    and item.get("revision") == spec["revision"])
        candidate = Path(item["path"])
        if candidate.resolve().is_relative_to(_cache_root() / "hub"):
            return candidate
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
        pass
    return canonical


def _fingerprint(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _legacy_fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _legacy_secure_paths_supported():
    # dir_fd + O_NOFOLLOW anchor every directory. On unsupported platforms
    # leave the old cache intact instead of using an unsafe path-only fallback.
    return (os.name == "posix" and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
            and all(function in os.supports_dir_fd for function in (os.open, os.stat, os.unlink)))


@contextlib.contextmanager
def _legacy_directory(parts, base_fd=None):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.path.sep, flags) if base_fd is None else os.dup(base_fd)
    try:
        for part in parts:
            if part in {"", ".", ".."} or os.path.sep in part:
                raise ValueError("unsafe_directory_component")
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _legacy_read_manifest(cache_fd):
    descriptor = os.open("ready.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cache_fd)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1024 * 1024:
            raise ValueError("unsafe_manifest")
        payload = handle.read(1024 * 1024 + 1)
        if _legacy_fingerprint(os.fstat(handle.fileno())) != _legacy_fingerprint(info):
            raise ValueError("manifest_changed")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("invalid_manifest")
    return manifest, _legacy_fingerprint(info)


def _legacy_candidates(cache_fd, root, manifest, report):
    candidates = []
    for key, spec in _LEGACY_MODEL_SPECS.items():
        item = manifest.get(key)
        if item is None:
            continue
        snapshot = root / "hub" / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]
        if (not isinstance(item, dict) or item.get("repo_id") != spec["repo_id"]
                or item.get("revision") != spec["revision"] or item.get("path") != str(snapshot)
                or not isinstance(item.get("verified_files"), dict)):
            report["skipped"].append({"key": key, "reason": "unrecognized_manifest_entry"})
            continue
        for relative, (size, digest) in spec["files"].items():
            record = {"key": key, "filename": relative, "path": str(snapshot / relative), "bytes": size}
            verified = item["verified_files"].get(relative)
            if (not isinstance(verified, dict) or verified.get("digest") != digest
                    or not isinstance(verified.get("fingerprint"), list)
                    or len(verified["fingerprint"]) != 5
                    or any(type(value) is not int for value in verified["fingerprint"])
                    or verified["fingerprint"][2] != size):
                report["skipped"].append({**record, "reason": "unverified_file"})
                continue
            # A second manifest entry may intentionally share this snapshot.
            shared = False
            for other_key, other in manifest.items():
                other_path = other.get("path") if isinstance(other, dict) else None
                if other_key != key and isinstance(other_path, str) and Path(other_path).is_absolute():
                    shared |= Path(record["path"]).is_relative_to(Path(os.path.normpath(other_path)))
            if shared:
                report["skipped"].append({**record, "reason": "shared_manifest_path"})
                continue
            try:
                parts = (snapshot / relative).relative_to(root).parts
                with _legacy_directory(parts[:-1], cache_fd) as parent_fd:
                    info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                reason = ("not_regular_file" if not stat.S_ISREG(info.st_mode) else
                          "shared_hardlink" if info.st_nlink != 1 else
                          "file_changed" if _legacy_fingerprint(info) != verified["fingerprint"] else None)
                if reason:
                    report["skipped"].append({**record, "reason": reason})
                    continue
                candidates.append({**record, "fingerprint": verified["fingerprint"], "digest": digest})
            except FileNotFoundError:
                report["skipped"].append({**record, "reason": "missing_file"})
            except OSError as exc:
                collection = "skipped" if exc.errno in {errno.ELOOP, errno.ENOTDIR} else "errors"
                report[collection].append({**record, "reason": "unsafe_or_unavailable_path",
                                           "error": f"{type(exc).__name__}: {exc}"})
    return candidates


def legacy_model_cache_summary():
    """Cheap offline inventory of unchanged app-managed v16 files only.

    Digests must match the pinned allowlist and the previously verified inode,
    size and timestamps must still match. No model payload is read, no links
    are followed, and untracked files/partials/HF blobs are never considered.
    """
    root = _cache_root()
    report = {"cache_dir": str(root), "eligible_bytes": 0, "eligible_files": [], "skipped": [], "errors": []}
    if not root.is_absolute():
        report["errors"].append({"reason": "cache_root_not_absolute"})
        return report
    if not _legacy_secure_paths_supported():
        report["skipped"].append({"reason": "secure_cleanup_unavailable"})
        return report
    with _MODEL_LOCK:
        try:
            with _legacy_directory(root.parts[1:]) as cache_fd:
                manifest, _ = _legacy_read_manifest(cache_fd)
                report["eligible_files"] = _legacy_candidates(cache_fd, root, manifest, report)
            report["eligible_bytes"] = sum(item["bytes"] for item in report["eligible_files"])
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            report["errors"].append({"reason": "manifest_unavailable", "error": f"{type(exc).__name__}: {exc}"})
    return report


def _legacy_commit_removals(cache_fd, removed):
    # Re-read before merging so unrelated/new Qwen entries are preserved even
    # if another writer updated the manifest while file deletion was running.
    manifest, fingerprint = _legacy_read_manifest(cache_fd)
    for record in removed:
        key, relative = record["key"], record["filename"]
        spec, item = _LEGACY_MODEL_SPECS[key], manifest.get(key)
        verified = item.get("verified_files") if isinstance(item, dict) else None
        if (not isinstance(verified, dict) or item.get("repo_id") != spec["repo_id"]
                or item.get("revision") != spec["revision"]
                or str(Path(item.get("path", "")) / relative) != record["path"]
                or verified.get(relative) != {"fingerprint": record["fingerprint"], "digest": record["digest"]}):
            continue
        del verified[relative]
        if not verified and set(item) <= {"repo_id", "revision", "path", "verified_files"}:
            del manifest[key]
    temporary = ".ready-legacy-" + os.urandom(12).hex() + ".json"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=cache_fd)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        current = os.stat("ready.json", dir_fd=cache_fd, follow_symlinks=False)
        if current.st_nlink != 1 or _legacy_fingerprint(current) != fingerprint:
            raise ValueError("manifest_changed_before_commit")
        os.replace(temporary, "ready.json", src_dir_fd=cache_fd, dst_dir_fd=cache_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=cache_fd)


def prune_legacy_model_cache(cancel_event=None):
    """Retire only verified v16 files after the caller's successful Qwen output.

    Returns diagnostics instead of invalidating a completed image. A fully
    ready replacement cache is required; this never recursively removes a
    directory, follows symlinks, deletes partials or unlinks shared hardlinks.
    removed_bytes is the logical size of removed files, not measured free RAM
    or a guarantee about when an already-open file releases filesystem space.
    """
    root = _cache_root()
    report = {"cache_dir": str(root), "status": "unchanged", "removed_bytes": 0,
              "removed_files": [], "skipped": [], "errors": [], "cancelled": False}
    if not root.is_absolute():
        report["status"] = "failed"
        report["errors"].append({"reason": "cache_root_not_absolute"})
        return report
    with _MODEL_LOCK:
        try:
            _check_cancel(cancel_event)
            if not _legacy_secure_paths_supported():
                report["skipped"].append({"reason": "secure_cleanup_unavailable"})
                return report
            with _legacy_directory(root.parts[1:]) as cache_fd:
                manifest, manifest_fingerprint = _legacy_read_manifest(cache_fd)
                if len(_cached_paths()) != len(MODEL_SPECS):
                    report["status"] = "not_ready"
                    report["skipped"].append({"reason": "replacement_models_not_ready"})
                    return report
                candidates = _legacy_candidates(cache_fd, root, manifest, report)
                for record in candidates:
                    try:
                        _check_cancel(cancel_event)
                        current_manifest = os.stat("ready.json", dir_fd=cache_fd, follow_symlinks=False)
                        if _legacy_fingerprint(current_manifest) != manifest_fingerprint:
                            report["skipped"].append({"reason": "manifest_changed_during_cleanup"})
                            break
                        parts = Path(record["path"]).relative_to(root).parts
                        with _legacy_directory(parts[:-1], cache_fd) as parent_fd:
                            info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                                    or _legacy_fingerprint(info) != record["fingerprint"]):
                                report["skipped"].append({**record, "reason": "file_changed_before_delete"})
                                continue
                            _check_cancel(cancel_event)
                            os.unlink(parts[-1], dir_fd=parent_fd)
                        report["removed_files"].append(record)
                        report["removed_bytes"] += record["bytes"]
                    except ModelCancelled:
                        report["cancelled"] = True
                        break
                    except OSError as exc:
                        report["errors"].append({**record, "reason": "unlink_failed",
                                                 "error": f"{type(exc).__name__}: {exc}"})
                if report["removed_files"]:
                    try:
                        _legacy_commit_removals(cache_fd, report["removed_files"])
                    except (OSError, ValueError, TypeError) as exc:
                        report["errors"].append({"reason": "manifest_update_failed",
                                                 "error": f"{type(exc).__name__}: {exc}"})
                    report["status"] = "partial" if report["errors"] or report["cancelled"] else "completed"
        except ModelCancelled:
            report["cancelled"] = True
        except (OSError, ValueError, TypeError) as exc:
            report["errors"].append({"reason": "cleanup_unavailable", "error": f"{type(exc).__name__}: {exc}"})
    if report["cancelled"]:
        report["status"] = "cancelled"
    elif report["errors"] and report["status"] == "unchanged":
        report["status"] = "failed"
    return report


def _file_identity(spec, relative):
    metadata = MODEL_FILE_METADATA[spec["repo_id"]][relative]
    return {"repo_id": spec["repo_id"], "revision": spec["revision"], "filename": relative,
            "size": metadata["size"], "algorithm": metadata["algorithm"], "digest": metadata["digest"]}


def _partial_paths(path):
    return path.with_name(path.name + ".unmark.part"), path.with_name(path.name + ".unmark.part.json")


def _partial_bytes(path, identity):
    """Count only our append-written, checkpointed partials, never Xet files."""
    partial, checkpoint = _partial_paths(path)
    try:
        if partial.is_symlink() or checkpoint.is_symlink():
            return 0
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        received = data["received_bytes"]
        if (data["identity"] == identity and isinstance(received, int)
                and 0 <= received <= identity["size"]
                and _fingerprint(partial) == data["fingerprint"]
                and partial.stat().st_size == received):
            return received
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 0


def setup_estimate():
    """Read-only, offline estimate for the first-run confirmation dialog.

    Pre-download times are assumptions, not measured network speeds. Actual
    throughput/ETA is available only from prepare_models() progress events.
    """
    total = cached = resumable = 0
    models = []
    for key, spec in MODEL_SPECS.items():
        path = _snapshot_path(spec)
        invalid = set(_invalid_snapshot_files(path, spec))
        group_total = group_cached = group_resume = 0
        for relative in spec["files"]:
            identity = _file_identity(spec, relative)
            group_total += identity["size"]
            if relative not in invalid:
                group_cached += identity["size"]
            else:
                group_resume += _partial_bytes(path / relative, identity)
        total += group_total; cached += group_cached; resumable += group_resume
        models.append({"key": key, "repo_id": spec["repo_id"], "total_bytes": group_total,
                       "cached_bytes": group_cached, "resumable_bytes": group_resume})
    remaining = max(0, total - cached - resumable)
    return {"total_bytes": total, "cached_bytes": cached, "resumable_bytes": resumable,
            "remaining_bytes": remaining, "cached": bool(_cached_paths()),
            "needs_download": cached < total, "estimated_seconds": None,
            "estimated_seconds_range": [remaining / 37_500_000, remaining / 12_500_000],
            "estimate_basis": "100~300 Mbps 가정의 순수 전송 시간 · 실제 속도와 검증 시간에 따라 달라집니다.",
            "models": models, "cache_dir": str(_cache_root()),
            "legacy_cache_bytes": legacy_model_cache_summary()["eligible_bytes"]}


def _emit_model_progress(callback, payload):
    if callback is None:
        return
    # Preserve older two-required-argument callers without catching exceptions
    # actually raised by the callback itself.
    import inspect
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(payload)
        return
    try:
        signature.bind(payload)
    except TypeError:
        callback(payload["title"], payload["detail"])
    else:
        callback(payload)


class _DownloadProgress:
    def __init__(self, callback, total):
        self.callback = callback
        self.total = total
        self.available = {}
        self.transferred = 0
        self.samples = collections.deque()
        self.last_emit = 0.0
        self.started = time.monotonic()
        self.samples.append((self.started, 0))

    def set_available(self, key, count):
        self.available[key] = max(0, int(count))

    def received(self, key, available, count, title, detail):
        self.available[key] = available
        self.transferred += count
        self.emit("download", title, detail)

    def emit(self, phase, title, detail, force=False, **extra):
        now = time.monotonic()
        if not force and now - self.last_emit < .15:
            return
        self.last_emit = now
        if phase in {"connecting", "retry"}:
            # Do not display transfer speed from the previous connection while
            # waiting for a new HTTPS connection or a retry backoff.
            self.samples.clear()
        self.samples.append((now, self.transferred))
        while len(self.samples) > 2 and self.samples[1][0] < now - 12:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        rate = ((self.transferred - self.samples[0][1]) / elapsed
                if phase == "download" and elapsed >= .05
                and self.transferred > self.samples[0][1] else 0.0)
        available = min(self.total, sum(self.available.values()))
        remaining = max(0, self.total - available)
        eta = remaining / rate if rate > 0 and phase == "download" else (0.0 if phase == "ready" else None)
        fraction = available / self.total if self.total else 1.0
        # The final validation/atomic ready manifest must finish before 100%.
        progress = 1.0 if phase == "ready" else min(.99, fraction)
        _emit_model_progress(self.callback, {
            "phase": phase, "title": title, "detail": detail,
            "downloaded_bytes": available, "total_bytes": self.total,
            "transferred_bytes": self.transferred, "bytes_per_second": rate,
            "eta_seconds": eta, "progress": progress, **extra,
        })


def _new_hasher(identity):
    hasher = hashlib.new(identity["algorithm"])
    if identity["algorithm"] == "sha1":
        # Non-LFS HF objects use Git blob hashes, not a plain file SHA1.
        hasher.update(f"blob {identity['size']}\0".encode("ascii"))
    return hasher


def _hash_file(path, identity, cancel_event, byte_limit=None):
    hasher = _new_hasher(identity)
    remaining = identity["size"] if byte_limit is None else byte_limit
    with path.open("rb") as handle:
        while remaining:
            _check_cancel(cancel_event)
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise _DownloadIntegrityError("파일 길이가 예상보다 짧습니다.")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher


class _DownloadIntegrityError(ModelRuntimeError):
    pass


class _DownloadInterruptedError(ConnectionError):
    """A short HTTP body is resumable; it is not a completed corrupt file."""


def _model_download_network_reason(exc):
    """Unwrap urllib's network reason without exposing a signed CDN URL."""
    while isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        reason = exc.reason
        if not isinstance(reason, BaseException) or reason is exc:
            break
        exc = reason
    return exc


def _retryable_model_download_error(exc):
    """Retry transport interruptions and selected server errors only.

    Authentication, absent files, certificate validation and local filesystem
    failures remain immediate errors. In particular this does not broaden the
    independent, narrowly scoped OpenSSL error-92 compatibility handling.
    """
    error = _model_download_network_reason(exc)
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _MODEL_DOWNLOAD_RETRY_HTTP_STATUS
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, (TimeoutError, ConnectionError, http.client.IncompleteRead,
                          ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    if isinstance(error, OSError):
        return error.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
            errno.ECONNREFUSED, errno.EPIPE, errno.ENETRESET,
            errno.ENETUNREACH, errno.EHOSTUNREACH,
        }
    return False


def _model_download_error_label(exc):
    """A short diagnosis safe for the UI/log; never stringify response URLs."""
    error = _model_download_network_reason(exc)
    if isinstance(error, urllib.error.HTTPError):
        return f"서버 응답 HTTP {error.code}"
    if isinstance(error, TimeoutError) or getattr(error, "errno", None) == errno.ETIMEDOUT:
        return "연결 시간 초과"
    if isinstance(error, socket.gaierror):
        return "서버 주소 확인 지연"
    return "다운로드 연결 끊김"


def _model_download_retry_delay(exc, retry_attempt):
    """Cap normal backoff at 30s; honor server Retry-After up to 300s.

    Longer requested waits stop with a resumable error instead of making an
    earlier request than the server permits. Attempts are one-based.
    """
    delay = float(_MODEL_DOWNLOAD_RETRY_DELAYS[
        min(max(0, retry_attempt - 1), len(_MODEL_DOWNLOAD_RETRY_DELAYS) - 1)])
    error = _model_download_network_reason(exc)
    if isinstance(error, urllib.error.HTTPError) and error.headers is not None:
        value = error.headers.get("Retry-After")
        if value:
            try:
                requested = float(value)
            except (ValueError, TypeError):
                try:
                    requested = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    requested = 0.0
            if math.isfinite(requested) and requested > 0:
                if requested > _MODEL_DOWNLOAD_MAX_RETRY_AFTER:
                    failure = ModelRuntimeError(
                        "서버가 긴 대기 시간을 요청했습니다. 받은 파일은 보관했습니다. "
                        "잠시 후 다시 시도해 주세요.")
                    failure.retry_after_seconds = requested
                    raise failure from exc
                return max(delay, requested)
    return min(float(_MODEL_DOWNLOAD_RETRY_DELAYS[-1]), delay)


def _wait_model_download_retry(cancel_event, delay):
    _check_cancel(cancel_event)
    if cancel_event is None:
        time.sleep(delay)
    else:
        cancel_event.wait(delay)
    _check_cancel(cancel_event)


class _PublicModelRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ModelRuntimeError("모델 서버가 안전하지 않은 다운로드 주소를 반환했습니다.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            redirected.remove_header("Cookie")
        return redirected


def _model_tls_context():
    """Use OS certificate validation plus portable roots, without global patches.

    macOS keychain validation can fetch missing intermediate certificates;
    certifi/OpenSSL alone rejected the Hugging Face CDN on the target Mac.
    truststore preserves certificate and hostname checks on each request.
    """
    try:
        import certifi
        import truststore
        ca_file = Path(certifi.where())
        if not ca_file.is_file():
            raise FileNotFoundError(ca_file)
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cafile=str(ca_file))
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise ModelRuntimeError(
            "보안 연결에 필요한 인증서 파일을 준비하지 못했습니다. 준비 화면에서 다시 시도해 주세요."
        ) from exc
    try:
        # Retain certifi's portable roots and add any platform/OpenSSL trust
        # store exposed by Python, including locally installed proxy CAs.
        context.load_default_certs(purpose=ssl.Purpose.SERVER_AUTH)
    except OSError:
        # System roots are optional; the verified certifi bundle remains loaded.
        logging.getLogger(__name__).debug("Additional system CA roots could not be loaded", exc_info=True)
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise ModelRuntimeError("보안 연결 설정을 확인하지 못했습니다. 앱을 다시 실행해 주세요.")
    return context


def _legacy_model_ca_key_usage_error(exc):
    """Match OpenSSL's specific legacy-CA error, never generic TLS failures."""
    error = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return (isinstance(error, ssl.SSLCertVerificationError)
            and getattr(error, "verify_code", None) == 92
            and str(getattr(error, "verify_message", "")).strip().casefold()
            == "ca cert does not include key usage extension")


class _ModelHTTPSHandler(urllib.request.HTTPSHandler):
    """Allow one narrowly triggered legacy-CA retry for a model request.

    Python 3.13+ enables VERIFY_X509_STRICT by default. OpenSSL error 92
    identifies a trusted-chain CA that lacks the keyUsage extension. Only that
    error can retry with STRICT removed from a fresh context; trust-chain,
    certificate validity, hostname and all other verification flags remain.
    Redirects return to the original strict handler, not the retry context.
    """
    def __init__(self, context):
        super().__init__(context=context)
        self._model_strict_context = context
        self._legacy_ca_retry_used = False

    def https_open(self, request):
        try:
            return super().https_open(request)
        except (urllib.error.URLError, ssl.SSLCertVerificationError) as exc:
            strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
            flags = self._model_strict_context.verify_flags
            if (self._legacy_ca_retry_used or not strict or not (flags & strict)
                    or not _legacy_model_ca_key_usage_error(exc)):
                raise
            self._legacy_ca_retry_used = True
            retry_context = _model_tls_context()
            retry_context.verify_flags = int(flags) & ~int(strict)
            if retry_context.verify_mode != ssl.CERT_REQUIRED or not retry_context.check_hostname:
                raise ModelRuntimeError("보안 연결 설정을 확인하지 못했습니다. 앱을 다시 실행해 주세요.") from exc
            # Log only the hostname, never a CDN's signed URL/query parameters.
            logging.getLogger(__name__).warning(
                "Model HTTPS compatibility retry for %s: OpenSSL verify_code=92 "
                "(CA keyUsage missing); CERT_REQUIRED and hostname checks remain enabled",
                urllib.parse.urlsplit(request.full_url).hostname,
            )
            # Use a plain one-shot handler so a second failure cannot recurse.
            # The caller's opener still applies the existing redirect policy.
            return urllib.request.HTTPSHandler(context=retry_context).https_open(request)


def _open_model_stream(url, headers):
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ModelRuntimeError("모델 다운로드에는 HTTPS 주소가 필요합니다.")
    # All requested repositories are public. Never read or send HF tokens,
    # netrc credentials or cookies to the Hub or its CDN redirect targets.
    request = urllib.request.Request(url, headers={
        "User-Agent": "UNMARK-model-downloader/1.0",
        "Accept-Encoding": "identity", **headers,
    })
    tls_handler = _ModelHTTPSHandler(context=_model_tls_context())
    try:
        return urllib.request.build_opener(_PublicModelRedirect(), tls_handler).open(
            request, timeout=_MODEL_DOWNLOAD_TIMEOUT)
    except (urllib.error.URLError, ssl.SSLCertVerificationError) as exc:
        error = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if not isinstance(error, ssl.SSLCertVerificationError):
            raise
        code = getattr(error, "verify_code", None)
        message = str(getattr(error, "verify_message", "인증서 확인 실패"))
        failure = ModelRuntimeError(f"모델 서버 인증서 확인 실패 (검증 코드 {code}: {message}).")
        failure.verify_code, failure.verify_message = code, message
        # Preserve the original verification exception for detailed diagnostics;
        # the visible message contains no URL, CDN signature or credentials.
        raise failure from exc


def _write_checkpoint(partial, checkpoint, identity, received, hasher):
    if not partial.is_file():
        return
    data = {"identity": identity, "received_bytes": received,
            "prefix_digest": hasher.hexdigest(), "fingerprint": _fingerprint(partial)}
    fd, temporary = tempfile.mkstemp(prefix=".unmark-checkpoint-", dir=checkpoint.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temporary, checkpoint)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _discard_partial(partial, checkpoint):
    for path in (partial, checkpoint):
        if path.exists() or path.is_symlink():
            path.unlink()


def _download_one(path, spec, relative, tracker, title, cancel_event, *, source_url=None):
    """Download a pinned object with resumable, bounded transport recovery.

    An existing checkpoint is hash-checked once before resuming. During this
    invocation, the in-memory hasher tracks every flushed/checkpointed prefix,
    so a transient disconnection does not re-read gigabytes already received.
    Each new connection starts at the original pinned source URL, obtaining
    fresh CDN redirects. Range/size/hash validation and atomic publication remain
    mandatory. Integrity failure permits one clean repair; transport retries
    have a separate five-retry budget shared by both integrity attempts.

    Progress contract: ``connecting`` and ``retry`` preserve received bytes but
    report zero speed and no ETA. ``retry`` additionally includes one-based
    ``retry_attempt``, ``max_retries`` and ``retry_delay_seconds``. It is emitted
    before the cancellable backoff; ``connecting`` follows before opening TLS.
    """
    identity = _file_identity(spec, relative)
    key = (spec["repo_id"], relative)
    partial, checkpoint = _partial_paths(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if partial.is_symlink() or checkpoint.is_symlink():
        raise ModelRuntimeError("모델 임시 파일 경로가 올바르지 않습니다.")
    if source_url is None:
        from huggingface_hub import hf_hub_url
        source_url = hf_hub_url(spec["repo_id"], relative, revision=spec["revision"],
                               endpoint="https://huggingface.co")
    # Raw GitHub prompt embeddings share the same HTTPS, resume, hash and
    # atomic-publication checks as the HF weight files.
    url = source_url
    detail = f"{spec['repo_id']} · {relative}"
    retry_attempt = 0
    for repair_attempt in range(2):
        _check_cancel(cancel_event)
        received = _partial_bytes(path, identity) if repair_attempt == 0 else 0
        hasher = _new_hasher(identity)
        if received:
            tracker.emit("verify", "이어받기 파일 확인", detail, force=True)
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            candidate = _hash_file(partial, identity, cancel_event, byte_limit=received)
            if candidate.hexdigest() != data["prefix_digest"]:
                received = 0
            else:
                hasher = candidate
        if not received:
            _discard_partial(partial, checkpoint)
        tracker.set_available(key, received)
        last_checkpoint = time.monotonic()
        try:
            while received < identity["size"]:
                _check_cancel(cancel_event)
                offset = received
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                tracker.emit("connecting", "모델 다운로드 서버에 연결하고 있습니다.", detail, force=True)
                try:
                    # Always reopen the original resolve URL. Never retain or
                    # log a redirect's expiring signed CDN URL for a retry.
                    with _open_model_stream(url, headers) as response:
                        status = response.getcode()
                        if status == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not match or tuple(map(int, match.groups())) != (offset, identity["size"] - 1, identity["size"]):
                                raise _DownloadIntegrityError("이어받기 범위가 요청과 일치하지 않습니다.")
                        elif status == 200:
                            if offset:
                                offset = received = 0
                                hasher = _new_hasher(identity)
                                tracker.set_available(key, 0)
                        else:
                            raise _DownloadIntegrityError(f"예상하지 못한 다운로드 응답: HTTP {status}")
                        encoding = response.headers.get("Content-Encoding", "identity").lower()
                        if encoding != "identity":
                            raise _DownloadIntegrityError("모델 다운로드 압축 형식이 예상과 다릅니다.")
                        declared = response.headers.get("Content-Length")
                        if declared is not None and int(declared) != identity["size"] - offset:
                            raise _DownloadIntegrityError("서버의 파일 길이가 고정 모델 정보와 다릅니다.")
                        tracker.emit("download", title, detail, force=True)
                        with partial.open("ab" if offset else "wb") as handle:
                            try:
                                reader = getattr(response, "read1", response.read)
                                while True:
                                    _check_cancel(cancel_event)
                                    interrupted = None
                                    try:
                                        chunk = reader(256 * 1024)
                                    except http.client.IncompleteRead as exc:
                                        # HTTPResponse.read() can carry bytes
                                        # received just before a short EOF.
                                        chunk, interrupted = exc.partial, exc
                                    if chunk:
                                        if received + len(chunk) > identity["size"]:
                                            raise _DownloadIntegrityError("모델 파일이 예상 크기보다 큽니다.")
                                        handle.write(chunk)
                                        hasher.update(chunk)
                                        received += len(chunk)
                                        tracker.received(key, received, len(chunk), title, detail)
                                        now = time.monotonic()
                                        if now - last_checkpoint >= 1:
                                            handle.flush()
                                            _write_checkpoint(partial, checkpoint, identity, received, hasher)
                                            last_checkpoint = now
                                    if interrupted is not None:
                                        raise interrupted
                                    if not chunk:
                                        break
                            finally:
                                # Flush before recording its fingerprint/hash;
                                # this also runs for timeout and cancellation.
                                handle.flush()
                                _write_checkpoint(partial, checkpoint, identity, received, hasher)
                    if received != identity["size"]:
                        raise _DownloadInterruptedError("모델 서버가 파일 전송을 중단했습니다.")
                except Exception as exc:
                    _check_cancel(cancel_event)
                    if not _retryable_model_download_error(exc):
                        raise
                    if received == identity["size"]:
                        # The final body bytes may arrive before an EOF error.
                        # The pinned hash below, not the connection close,
                        # decides whether this complete object is valid.
                        break
                    label = _model_download_error_label(exc)
                    if retry_attempt >= _MODEL_DOWNLOAD_MAX_RETRIES:
                        failure = ModelRuntimeError(
                            f"모델 다운로드 연결이 {_MODEL_DOWNLOAD_MAX_RETRIES}회 재시도 후에도 복구되지 않았습니다. "
                            "받은 파일은 보관했습니다. 네트워크 연결을 확인한 뒤 다시 시도하면 이어받습니다."
                            f"\n{label} · {relative}")
                        failure.retry_attempts = retry_attempt
                        failure.filename = relative
                        raise failure from exc
                    retry_attempt += 1
                    response_error = _model_download_network_reason(exc)
                    try:
                        delay = _model_download_retry_delay(exc, retry_attempt)
                    finally:
                        if isinstance(response_error, urllib.error.HTTPError):
                            # open() raised before entering its context manager.
                            # Release even when Retry-After exceeds the limit.
                            response_error.close()
                    tracker.emit(
                        "retry", "다운로드 연결을 다시 시도합니다.",
                        f"{label} · {relative} · {delay:g}초 후 자동 재시도 "
                        f"({retry_attempt}/{_MODEL_DOWNLOAD_MAX_RETRIES}) · 받은 파일은 유지됩니다.",
                        force=True, retry_attempt=retry_attempt,
                        max_retries=_MODEL_DOWNLOAD_MAX_RETRIES, retry_delay_seconds=delay)
                    _wait_model_download_retry(cancel_event, delay)
            _check_cancel(cancel_event)
            tracker.emit("verify", "다운로드 파일 확인", detail, force=True)
            if hasher.hexdigest() != identity["digest"]:
                raise _DownloadIntegrityError("다운로드 파일의 해시가 고정 모델 정보와 다릅니다.")
            _check_cancel(cancel_event)
            os.replace(partial, path)  # Replaces an old HF snapshot symlink, never its shared blob.
            if checkpoint.exists():
                checkpoint.unlink()
            tracker.set_available(key, identity["size"])
            return
        except _DownloadIntegrityError:
            _discard_partial(partial, checkpoint)
            tracker.set_available(key, 0)
            if repair_attempt:
                raise
            tracker.emit("download", "손상 파일 다시 받는 중", detail, force=True)


def prepare_models(progress_callback=None, cancel_event=None):
    """Receive real HTTPS bytes after the app's first-run confirmation.

    Byte totals and hashes come from fixed public model and prompt-cache revisions.
    Existing standard Hugging Face snapshots are reused. New files are written
    atomically into that documented snapshot layout. No Xet preallocated size,
    file count, synthetic progress, token or external executable is involved.
    """
    with _MODEL_LOCK:
        _check_cancel(cancel_event)
        estimate = setup_estimate()
        tracker = _DownloadProgress(progress_callback, estimate["total_bytes"])
        for spec in MODEL_SPECS.values():
            path = _snapshot_path(spec)
            invalid = set(_invalid_snapshot_files(path, spec))
            for relative in spec["files"]:
                identity = _file_identity(spec, relative)
                available = identity["size"] if relative not in invalid else _partial_bytes(path / relative, identity)
                tracker.set_available((spec["repo_id"], relative), available)
        paths = _cached_paths()
        if paths:
            tracker.emit("ready", "모델 준비 완료", "저장된 모델을 사용합니다.", force=True)
            return paths
        root = _cache_root()
        root.mkdir(parents=True, exist_ok=True)
        tracker.emit("verify", "저장된 모델 확인", "필요한 파일만 확인하고 준비합니다.", force=True)
        try:
            old_manifest = json.loads((root / "ready.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            old_manifest = {}
        if not isinstance(old_manifest, dict):
            old_manifest = {}
        # Retain old verification records until a successful Qwen output lets
        # the explicit, allowlisted legacy-cache migration retire those files.
        manifest = dict(old_manifest)
        for index, (key, spec) in enumerate(MODEL_SPECS.items(), 1):
            path = _snapshot_path(spec)
            invalid = set(_invalid_snapshot_files(path, spec))
            verified_files = {}
            for relative in spec["files"]:
                _check_cancel(cancel_event)
                identity = _file_identity(spec, relative)
                file = path / relative
                cached_ok = relative not in invalid
                old_model = old_manifest.get(key)
                old_files = old_model.get("verified_files") if isinstance(old_model, dict) else None
                old = old_files.get(relative) if isinstance(old_files, dict) else None
                if cached_ok and (not isinstance(old, dict)
                        or old.get("fingerprint") != _fingerprint(file)
                        or old.get("digest") != identity["digest"]):
                    tracker.emit("verify", "저장된 파일 확인", f"{spec['repo_id']} · {relative}", force=True)
                    cached_ok = _hash_file(file, identity, cancel_event).hexdigest() == identity["digest"]
                if not cached_ok:
                    tracker.set_available((spec["repo_id"], relative), 0)
                    _download_one(
                        file, spec, relative, tracker,
                        f"모델 다운로드 {index}/{len(MODEL_SPECS)}", cancel_event,
                        source_url=spec.get("source_urls", {}).get(relative),
                    )
                if relative in _invalid_snapshot_files(path, {**spec, "files": (relative,)}):
                    raise ModelRuntimeError(f"모델 파일 검증에 실패했습니다: {relative}")
                verified_files[relative] = {"fingerprint": _fingerprint(file), "digest": identity["digest"]}
                tracker.set_available((spec["repo_id"], relative), identity["size"])
            paths[key] = str(path)
            manifest[key] = {"repo_id": spec["repo_id"], "revision": spec["revision"],
                             "path": str(path), "verified_files": verified_files}
            tracker.emit("verify", f"모델 확인 {index}/{len(MODEL_SPECS)}", spec["repo_id"], force=True)
        _check_cancel(cancel_event)
        fd, temporary = tempfile.mkstemp(prefix="ready-", suffix=".json", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, root / "ready.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        tracker.emit("ready", "모델 준비 완료", "모든 모델 파일을 확인했습니다.", force=True)
        return paths

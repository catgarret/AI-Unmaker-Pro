#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Unmarker Pro v1.2.0: unified local processing and official verification.

Run: python main.py
Build: python build.py
The image pipeline uses quantized Qwen-Image-2512 with Lightning, then
restores fine detail from the original image using the DeSynth method.
Weights are downloaded on first setup and never bundled in the application.
Public DWT-DCT payload decoding and FFT measurements are separate checks;
neither is a SynthID detector. Every output is reopened and measured again.
Local processing does not upload images. Optional official OpenAI verification
requires an API key and explicit upload consent; Google uses its official site.
See README.md for installation, supported checks, and measured verification.
"""
from __future__ import annotations

import os
import sys

APP_VERSION = "1.2.0"
ENGINE_ID = "local-qwen-q4-detail-v3"
RELEASE_ID = "2026-10-21.4"

# Leave CPU capacity for the desktop while a single image is being processed.
for _thread_option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "2"

# PyInstaller's windowed bootloader can leave these streams unset. Libraries
# such as tqdm/logging still expect writable streams while fetching models.
if getattr(sys, "frozen", False):
    for _stream_name in ("stdout", "stderr"):
        if getattr(sys, _stream_name, None) is None:
            setattr(sys, _stream_name, open(os.devnull, "w", encoding="utf-8"))

# runtime_setup.py
# ------------------------------------------------------------------------
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid


_RS_REQUIREMENTS = {'customtkinter': ('customtkinter>=5.2.2,<6', '>=5.2.2,<6'),
 'tkinterdnd2': ('tkinterdnd2==0.6.4', '==0.6.4'),
 'pillow': ('Pillow>=10.4,<13', '>=10.4,<13'),
 'numpy': ('numpy>=1.26,<3', '>=1.26,<3'),
 'pywavelets': ('PyWavelets>=1.8,<2', '>=1.8,<2'),
 'opencv-python-headless': ('opencv-python-headless>=4.8,<5', '>=4.8,<5'),
 'remove-ai-watermarks': ('remove-ai-watermarks[visible,detect]==0.46.0', '==0.46.0'),
 'c2pa-python': ('c2pa-python==0.38.0', '==0.38.0'),
 'packaging': ('packaging>=24,<27', '>=24,<27'),
 'requests': ('requests>=2.32,<3', '>=2.32,<3'),
 'certifi': ('certifi>=2024.7.4', '>=2024.7.4'),
 'truststore': ('truststore>=0.10,<1', '>=0.10,<1'),
 'tqdm': ('tqdm>=4.66,<5', '>=4.66,<5'),
 'psutil': ('psutil>=5.9,<8', '>=5.9,<8'),
 'torch': ('torch>=2.6,<3', '>=2.6,<3'),
 'torchvision': ('torchvision>=0.21,<1', '>=0.21,<1'),
 'diffusers': ('diffusers==0.36.0', '==0.36.0'),
 'peft': ('peft==0.18.0', '==0.18.0'),
 'gguf': ('gguf==0.17.1', '==0.17.1'),
 'transformers': ('transformers>=4.51,<5', '>=4.51,<5'),
 'huggingface-hub': ('huggingface-hub>=0.34,<1', '>=0.34,<1'),
 'accelerate': ('accelerate>=1.2,<2', '>=1.2,<2'),
 'safetensors': ('safetensors>=0.4.5,<1', '>=0.4.5,<1'),
 'regex': ('regex>=2024.5.15', '>=2024.5.15'),
 'filelock': ('filelock>=3.15,<4', '>=3.15,<4'),
 'pyyaml': ('PyYAML>=6.0,<7', '>=6.0,<7')}


def _rs_requirements(profile="core"):
    if profile not in ("core", "models"):
        raise ValueError("Unknown runtime profile")
    return dict(_RS_REQUIREMENTS)


_RS_ACTIVE_PATHS = []
_RS_DLL_HANDLES = []
_RS_THREAD_LOCK = threading.Lock()
_RS_PROBE_TIMEOUT = 240
_RS_INSTALL_TIMEOUT = 5400
_RS_MAX_ATTEMPTS = 2
_RS_PIP_BOOTSTRAP_VERSION = "25.3"
_RS_PROGRESS_PATTERN = re.compile(r"Progress\s+(\d+)\s+of\s+(\d+)")
_RS_ERROR_MESSAGES = {
    "network": "인터넷 연결이 원활하지 않아 실행 준비를 마치지 못했습니다. 연결을 확인한 뒤 다시 시도해 주세요.",
    "disk_space": "저장 공간이 부족해 실행 준비를 마치지 못했습니다. 여유 공간을 확보한 뒤 다시 시도해 주세요.",
    "permission": "앱 전용 폴더에 파일을 저장하지 못했습니다. 폴더 접근 권한을 확인한 뒤 다시 시도해 주세요.",
    "unsupported": "현재 환경에서 사용할 수 있는 실행 구성요소를 준비하지 못했습니다. 이 운영체제와 Python 버전을 지원하는 앱 배포본이 필요합니다.",
    "cancelled": "실행 준비를 취소했습니다. 필요할 때 다시 준비할 수 있습니다.",
    "timeout": "실행 준비가 오래 걸려 중단했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.",
    "verification": "실행 구성요소의 시작 확인을 통과하지 못했습니다. 다시 시도해 주세요.",
    "installer": "설치 구성요소를 실행하지 못했습니다. 앱 배포본을 확인해 주세요.",
    "dependency_conflict": "필요한 구성요소의 버전 조건이 맞지 않아 설치를 마치지 못했습니다.",
    "installer_environment": "현재 Python 환경에서 설치 준비 명령을 실행하지 못했습니다.",
}


class _RS_Cancelled(Exception):
    pass


class _RS_Failure(Exception):
    def __init__(self, code, raw=""):
        super().__init__(raw or code)
        self.code = code


def _rs_frozen():
    return bool(getattr(sys, "frozen", False))


def _rs_abi():
    tag = getattr(sys.implementation, "cache_tag", None) or f"py{sys.version_info.major}{sys.version_info.minor}"
    value = f"{tag}-{sysconfig.get_platform()}-{platform.machine().lower()}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def _rs_root():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    root = (base / "UNMARK" / "runtime" / _rs_abi()).expanduser().resolve()
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle and root.is_relative_to(Path(bundle).resolve()):
        raise _RS_Failure("permission", "Runtime directory must be outside the application bundle")
    if root.is_relative_to(Path(sys.prefix).resolve()):
        raise _RS_Failure("permission", "Runtime directory must be outside the Python installation")
    return root


def _rs_mkdir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _rs_atomic_json(path, value):
    _rs_mkdir(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".unmark-json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _rs_read_json(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Runtime report exceeds the permitted size")
    return json.loads(path.read_text(encoding="utf-8"))


def _rs_layers():
    root = _rs_root()
    try:
        manifest = _rs_read_json(root / "current.json")
        if manifest.get("schema") != 1 or manifest.get("abi") != _rs_abi():
            return []
        names = manifest.get("layers", [])
        if not isinstance(names, list) or len(names) > 64:
            return []
        result = []
        for name in names:
            if not isinstance(name, str) or not re.fullmatch(r"g-[a-f0-9]{32}", name):
                return []
            path = (root / "packages" / name).resolve()
            if not path.is_relative_to(root / "packages") or not path.is_dir():
                return []
            marker = _rs_read_json(path / ".unmark-runtime.json")
            if marker.get("abi") != _rs_abi() or marker.get("complete") is not True:
                return []
            result.append(path)
        return list(dict.fromkeys(result))
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def _rs_activate(paths):
    global _RS_ACTIVE_PATHS
    # Do not execute arbitrary .pth files or reload already imported extensions.
    valid = []
    package_root = (_rs_root() / "packages").resolve()
    for item in paths:
        path = Path(item).resolve()
        if not path.is_dir() or not path.is_relative_to(package_root):
            raise _RS_Failure("permission", f"Invalid runtime package directory: {path}")
        valid.append(str(path))
    valid = list(dict.fromkeys(valid))
    sys.path[:] = [item for item in sys.path if item not in _RS_ACTIVE_PATHS and item not in valid]
    sys.path[:0] = valid
    _RS_ACTIVE_PATHS = valid
    importlib.invalidate_caches()
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        # Closing a search-directory handle does not unload an already loaded
        # DLL. Rebuild the search paths so an older generation cannot win a
        # duplicate native-library lookup after a repair.
        for _path, handle in _RS_DLL_HANDLES:
            with contextlib.suppress(OSError):
                handle.close()
        _RS_DLL_HANDLES.clear()
        previous = set()
        for text in valid:
            root = Path(text)
            directories = [root / "torch" / "lib", root / "torchvision", root / "PIL",
                           root / "cv2", root / "c2pa", root / "c2pa" / "libs"]
            directories += list(root.glob("*.libs"))
            for directory in directories:
                role = str(directory.relative_to(root))
                if directory.is_dir() and role not in previous:
                    with contextlib.suppress(OSError):
                        _RS_DLL_HANDLES.append((str(directory), os.add_dll_directory(str(directory))))
                        previous.add(role)
    return valid


def activate_runtime_packages():
    """Put verified app-owned packages before bundled/system packages on sys.path."""
    return _rs_activate(_rs_layers())


def _rs_notify(callback, detail="필요한 구성요소를 설치합니다.", **values):
    if callback is not None:
        event = {"phase": "dependencies", "title": "실행 준비 중", "detail": detail,
                 "indeterminate": True, "current": None, "total": None,
                 "downloaded_bytes": None, "total_bytes": None}
        event.update(values)
        callback(event)


def _rs_check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _RS_Cancelled()


def _rs_new_log():
    directory = _rs_mkdir(_rs_root() / "logs")
    path = directory / (time.strftime("runtime-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8] + ".log")
    path.touch(mode=0o600)
    return path


def _rs_log(path, text):
    # A full/unwritable disk must still produce a usable status for the UI.
    with contextlib.suppress(OSError):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(str(text) + "\n")


def _rs_safe_text(value):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    text = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", text)
    return re.sub(r"(?i)([?&](?:token|key|password|secret|signature|authorization)=)[^&\s]+", r"\1[redacted]", text)


def _rs_error_cause(raw):
    """Keep the actionable error, rather than the traceback or retry wrapper."""
    lines = [line.strip() for line in _rs_safe_text(raw).splitlines() if line.strip()]
    errors = [line for line in lines if re.search(
        r"(?i)(^error:|\b(?:ModuleNotFoundError|ImportError|OSError|RuntimeError|ValueError|"
        r"SSLError|PermissionError|FileNotFoundError):|no module named|no such option|"
        r"invalid choice|externally-managed-environment)", line)]
    useful = [line for line in errors if not re.search(r"(?i)(ResolutionImpossible:.*(?:help|https?://)|^ERROR: Exception:)", line)]
    selected = (useful or errors or lines)[-2:]
    if not selected:
        return ""
    # One line per actual failure; remove duplicate copies from worker/parent logs.
    summary = " / ".join(dict.fromkeys(re.sub(r"\s+", " ", line) for line in selected))
    return summary[:417] + "…" if len(summary) > 420 else summary


def _rs_failed_checks(status):
    return "\n".join(f"{name}: {check.get('error', 'import check failed')}"
                     for name, check in status.get("checks", {}).items() if not check.get("ok")) or status.get("error", "Runtime validation failed")


def _rs_error(code, log_path=None, missing=None, raw=""):
    if log_path and raw:
        _rs_log(log_path, raw)
    message = _RS_ERROR_MESSAGES.get(code, _RS_ERROR_MESSAGES["verification"])
    cause = _rs_error_cause(raw) if code != "cancelled" else ""
    detail = message + ("\n원인: " + cause if cause else "")
    return {"ready": False, "missing": list(missing or []), "error": detail,
            "error_code": code, "detail": detail, "cause": cause, "cancelled": code == "cancelled",
            "restart_required": False, "log_path": str(log_path) if log_path else ""}


def _rs_error_kind(text):
    lower = str(text).lower()
    if any(token in lower for token in ("no space left", "disk full", "errno 28", "not enough space", "winerror 112")):
        return "disk_space"
    if any(token in lower for token in ("permission denied", "access is denied", "winerror 5", "errno 13")):
        return "permission"
    if "externally-managed-environment" in lower or "externally managed environment" in lower:
        return "installer_environment"
    if any(token in lower for token in ("connection", "timed out", "name resolution", "network", "sslerror", "certificate verify", "proxyerror", "temporary failure", "urlopen error")):
        return "network"
    if any(token in lower for token in ("no matching distribution", "not a supported wheel", "requires a different python", "could not find a version", "wrong elf", "not a valid win32", "glibc_", "dll load failed")):
        return "unsupported"
    if any(token in lower for token in ("resolutionimpossible", "conflicting dependencies", "resolutiontoodeep")):
        return "dependency_conflict"
    if (any(token in lower for token in ("no such option", "invalid choice", "no module named 'pip", 'no module named "pip'))
            or ("pip._" in lower and any(token in lower for token in ("modulenotfounderror", "importerror")))):
        return "installer"
    return "verification"


def _rs_version_ok(version, constraint):
    # The installer can work even if the application packaging distribution is
    # itself absent. Pip vendors the same PEP 440 parser in packaged builds.
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        try:
            from pip._vendor.packaging.specifiers import SpecifierSet
        except ImportError:
            def release(text):
                if not re.fullmatch(r"\d+(?:\.\d+)*(?:\+[a-zA-Z0-9.]+)?", text):
                    return None
                parts = tuple(int(item) for item in text.split("+")[0].split("."))
                return parts + (0,) * (6 - len(parts))
            current = release(version)
            if current is None:
                return False
            for part in constraint.split(","):
                match = re.fullmatch(r"(==|>=|<=|>|<)(.+)", part)
                if not match or release(match[2]) is None:
                    return False
                other = release(match[2])
                if not {"==": current == other, ">=": current >= other, "<=": current <= other,
                        ">": current > other, "<": current < other}[match[1]]:
                    return False
            return True
    return version in SpecifierSet(constraint)


def _rs_platform_error():
    if sys.implementation.name != "cpython" or not (3, 11) <= sys.version_info[:2] < (3, 15) or sys.maxsize <= 2 ** 32:
        return "A supported 64-bit CPython 3.11–3.14 runtime is required"
    if sys.platform not in {"win32", "darwin", "linux"}:
        return f"Unsupported operating system: {sys.platform}"
    if importlib.util.find_spec("_tkinter") is None:
        return "The Python distribution does not include Tcl/Tk; it cannot be supplied by a pip wheel"
    return ""


def _rs_probe_actual(profile="core", *, include_gui=True):
    """Called only in a disposable worker; never calls from_pretrained/download."""
    unsupported = _rs_platform_error()
    if unsupported:
        print(unsupported, file=sys.stderr)
        return _rs_error("unsupported", raw=unsupported)
    checks, versions = {}, {}
    for name, (_requirement, constraint) in _rs_requirements(profile).items():
        try:
            version = importlib.metadata.version(name)
            versions[name] = version
            if not _rs_version_ok(version, constraint):
                raise ImportError(f"{name}{constraint} required; found {version}")
            checks[name] = {"ok": True, "version": version}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()

    def verify(name, operation):
        if not checks.get(name, {}).get("ok"):
            return
        try:
            operation()
        except Exception as exc:
            checks[name]["ok"] = False
            checks[name]["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    for distribution, module in (("customtkinter", "customtkinter"), ("tkinterdnd2", "tkinterdnd2"),
                                 ("packaging", "packaging"), ("requests", "requests"),
                                 ("regex", "regex"), ("filelock", "filelock"),
                                 ("pyyaml", "yaml"), ("accelerate", "accelerate"),
                                 ("safetensors", "safetensors.torch")):
        verify(distribution, lambda module=module: importlib.import_module(module))

    def tkdnd_check():
        # Importing tkinterdnd2 does not load its platform-specific Tcl/Tk
        # extension. Exercise the native library and widget registration in
        # this disposable worker so an incompatible/missing binary cannot
        # silently pass the finished application's runtime probe.
        result = checks["tkinterdnd2"]
        result.update(native_checked=False, callback_checked=False, native_check_stage="main_thread")
        if not include_gui:
            # A desktop caller validates TkDND on its already-existing root.
            # Creating another Tk root here registers this worker as a second
            # macOS GUI application even when that root is withdrawn. Keep
            # build/standalone probes native by default, and never call this
            # deferred result a native-check success.
            result.update(native_check_stage="parent_gui",
                          native_check_deferred="The desktop must validate its existing Tk root")
            return
        import tkinter as tk
        from tkinterdnd2 import TkinterDnD, DND_FILES, COPY
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("TkDND native check must run on the worker's main thread")
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            result["native_checked"] = False
            result["native_check_deferred"] = "Linux DISPLAY is unavailable"
            print("TkDND native check deferred: Linux DISPLAY is unavailable", file=sys.stderr)
            return
        window = None
        try:
            result["native_check_stage"] = "create_window"
            window = tk.Tk()
            window.withdraw()
            result["tcl_version"] = str(window.tk.call("info", "patchlevel"))
            result["tk_version"] = str(window.tk.call("package", "require", "Tk"))
            result["native_check_stage"] = "load_library"
            result["tkdnd_version"] = str(TkinterDnD.require(window))
            result["native_check_stage"] = "create_target"
            target = tk.Frame(window)
            target.pack()
            window.update_idletasks()
            result["native_check_stage"] = "register_target"
            target.drop_target_register(DND_FILES)
            result["native_check_stage"] = "bind_callback"
            received = []
            test_file = "/AI Unmarker Pro/드롭 검사.png"

            def on_drop(event):
                received.append((event.widget, window.tk.splitlist(event.data)))
                return COPY

            callback = target.dnd_bind("<<Drop>>", on_drop)
            binding = window.tk.splitlist(target.dnd_bind("<<Drop>>"))
            if not callback or not binding or binding[0] != callback:
                raise RuntimeError("TkDND drop callback was not registered")
            # Call the registered Tcl command through the wrapper's own bound
            # substitutions, exercising Tcl -> Python, DnDEvent and COPY. This
            # is a callback check, not a simulated native Finder drag gesture.
            values = {"%A": COPY, "%a": COPY, "%b": "1", "%C": "0", "%c": "0",
                      "%CST": DND_FILES, "%CTT": DND_FILES, "%D": (test_file,),
                      "%e": "<<Drop>>", "%L": DND_FILES, "%m": "", "%ST": DND_FILES,
                      "%T": DND_FILES, "%t": DND_FILES, "%TT": DND_FILES,
                      "%W": str(target), "%X": "0", "%Y": "0"}
            result["native_check_stage"] = "check_substitutions"
            result["callback_substitutions"] = list(binding[1:])
            unknown = sorted(set(binding[1:]).difference(values))
            if unknown:
                # An empty replacement would hide an incompatible wrapper and
                # can make this self-check pass while real drops still fail.
                raise RuntimeError("Unsupported TkDND callback substitutions: " + ", ".join(unknown))
            result["native_check_stage"] = "invoke_callback"
            action = window.tk.call(callback, *(values[token] for token in binding[1:]))
            if action != COPY or received != [(target, (test_file,))]:
                raise RuntimeError("TkDND drop callback did not accept the test file")
            result["callback_checked"] = True
            result["native_checked"] = True
            result["native_check_stage"] = "complete"
        except Exception as exc:
            result["native_checked"] = False
            cause = exc.__cause__ or exc.__context__
            if window is not None and (isinstance(exc, tk.TclError) or isinstance(cause, tk.TclError)):
                try:
                    result["native_tcl_error"] = str(window.tk.call("set", "errorInfo"))
                except tk.TclError:
                    pass
            detail = f"{type(exc).__name__}: {exc}"
            if cause is not None and str(cause) != str(exc):
                detail += f"; {type(cause).__name__}: {cause}"
            raise RuntimeError(
                "TkDND native check failed "
                f"({platform.system()} {platform.machine()}, "
                f"Tcl {result.get('tcl_version', 'unknown')}, "
                f"Tk {result.get('tk_version', 'unknown')}, "
                f"stage={result['native_check_stage']}): {detail}"
            ) from exc
        finally:
            if window is not None:
                # TkDND 0.6.4's macOS/aqua unregister branch raises 'error todo'.
                # This disposable window is destroyed instead of unregistering
                # a live target. Keep require/register/callback errors fatal.
                window.destroy()

    def pillow_check():
        from PIL import Image, ImageOps, ImageTk
        output = io.BytesIO()
        Image.new("RGBA", (8, 8), (17, 34, 51, 128)).save(output, format="PNG")
        output.seek(0)
        with Image.open(output) as image:
            image.load()
            assert image.size == (8, 8) and image.mode == "RGBA"

    def numpy_check():
        import numpy as np
        assert int(np.arange(4).sum()) == 6

    def cv2_check():
        import cv2
        import numpy as np
        assert cv2.Canny(np.zeros((16, 16), dtype=np.uint8), 50, 100).shape == (16, 16)

    def wavelet_check():
        import numpy as np
        import pywt
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        coeffs = pywt.dwt2(data, "haar")
        restored = pywt.idwt2(coeffs, "haar")
        assert np.allclose(data, restored, atol=1e-4)

    def hub_check():
        from huggingface_hub import snapshot_download, hf_hub_download
        assert callable(snapshot_download) and callable(hf_hub_download)

    def tqdm_check():
        from tqdm.auto import tqdm
        with tqdm(total=1, file=io.StringIO(), disable=True) as progress:
            progress.update(1)

    def certificates_check():
        import certifi
        import ssl
        ca_file = Path(certifi.where())
        if not ca_file.is_file():
            raise ImportError("The bundled HTTPS certificate file is missing")
        context = ssl.create_default_context(cafile=str(ca_file))
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        assert context.cert_store_stats()["x509_ca"] > 0

    def torch_check():
        import torch
        assert torch.tensor([1, 2], device="cpu").sum().item() == 3

    def torchvision_check():
        import torch
        from torchvision import transforms
        from torchvision.ops import nms
        from PIL import Image
        assert transforms.ToTensor()(Image.new("RGB", (4, 4))).shape == (3, 4, 4)
        # Exercise the bundled native extension, not only pure-Python transforms.
        boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device="cpu")
        scores = torch.tensor([0.9, 0.6], device="cpu")
        assert nms(boxes, scores, 0.5).tolist() == [0]

    def diffusers_check():
        import torch
        from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                               GGUFQuantizationConfig, QwenImageImg2ImgPipeline,
                               QwenImageTransformer2DModel)
        assert all(isinstance(item, type) for item in (
            AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
            QwenImageImg2ImgPipeline, QwenImageTransformer2DModel))
        assert callable(QwenImageTransformer2DModel.from_single_file)
        assert callable(QwenImageImg2ImgPipeline.load_lora_weights)
        config = GGUFQuantizationConfig(compute_dtype=torch.float32)
        assert config.compute_dtype == torch.float32
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(2, device="cpu")
        sample = torch.zeros((1, 4), dtype=torch.float32)
        result = scheduler.step(torch.ones_like(sample), scheduler.timesteps[0], sample).prev_sample
        assert result.shape == sample.shape and torch.isfinite(result).all()

    def transformers_check():
        from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
        assert all(isinstance(item, type) for item in (Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration))

    def peft_check():
        import torch
        from peft import LoraConfig, get_peft_model
        # A small local adapter exercises PEFT without loading a trained LoRA.
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
        adapter = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
        output = adapter(torch.ones((1, 8), dtype=torch.float32))
        assert output.shape == (1, 8) and torch.isfinite(output).all()
        assert any("lora_A" in name for name, _ in adapter.named_parameters())

    def gguf_check():
        import numpy as np
        from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
        from gguf.quants import dequantize, quantize
        assert isinstance(GGUFReader, type) and isinstance(GGUFWriter, type)
        values = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
        packed = quantize(values, GGMLQuantizationType.Q8_0)
        restored = dequantize(packed, GGMLQuantizationType.Q8_0)
        assert restored.shape == values.shape and np.allclose(values, restored, atol=0.01)

    def c2pa_check():
        from c2pa import Context, Reader
        assert isinstance(Context, type) and isinstance(Reader, type)

    def visible_check():
        # Exercise the exact supported API and strict/cv2 arguments on a local,
        # synthetic blank file. This also checks packaged templates/native code.
        from PIL import Image
        from remove_ai_watermarks import image_io, remove_visible_detailed, watermark_registry
        assert importlib.metadata.version("remove-ai-watermarks") == "0.46.0"
        with tempfile.TemporaryDirectory(prefix="unmark-visible-probe-") as directory:
            source = Path(directory) / "input.png"
            target = Path(directory) / "output.png"
            Image.new("RGB", (128, 128), (113, 113, 113)).save(source)
            bgr = image_io.load_image_bgr(source)
            detections = watermark_registry.detect_marks(bgr, include_explicit=False, provenance=frozenset())
            assert isinstance(detections, (tuple, list))
            assert not any(item.detected for item in detections)
            result = remove_visible_detailed(source, target, sensitivity="strict", backend="cv2", strip_metadata=False, write_noop=True)
            assert result.status == "no_watermark"
            assert target.is_file() and target.stat().st_size > 0
            with Image.open(target) as image:
                image.load()
                assert image.size == (128, 128)

    for name, operation in (("tkinterdnd2", tkdnd_check),
                            ("pillow", pillow_check), ("numpy", numpy_check), ("pywavelets", wavelet_check),
                            ("opencv-python-headless", cv2_check), ("huggingface-hub", hub_check),
                            ("tqdm", tqdm_check), ("torch", torch_check),
                            ("certifi", certificates_check),
                            ("torchvision", torchvision_check), ("diffusers", diffusers_check),
                            ("peft", peft_check), ("gguf", gguf_check),
                            ("transformers", transformers_check), ("c2pa-python", c2pa_check),
                            ("remove-ai-watermarks", visible_check)):
        verify(name, operation)
    if _rs_frozen():
        try:
            versions["pip"] = importlib.metadata.version("pip")
            from pip._internal.cli.main import main as pip_main
            assert callable(pip_main)
            checks["pip"] = {"ok": True, "version": versions["pip"]}
        except Exception as exc:
            checks["pip"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
    missing = [name for name, result in checks.items() if not result["ok"]]
    return {"ready": not missing, "missing": missing, "versions": versions, "checks": checks,
            "error": "" if not missing else "실행에 필요한 구성요소를 준비해야 합니다.",
            "error_code": "" if not missing else "dependencies",
            "detail": "실행에 필요한 구성요소를 확인했습니다." if not missing else "필요한 구성요소를 자동으로 설치합니다.",
            "restart_required": False, "cancelled": False, "profile": profile,
            "native_gui_check": bool(include_gui)}


def _rs_worker_command(flag, *arguments):
    if _rs_frozen():
        return [sys.executable, flag, *map(str, arguments)]
    return [sys.executable, str(Path(__file__).resolve()), flag, *map(str, arguments)]


def _rs_child_environment(paths, *, offline=False):
    environment = os.environ.copy()
    # No pip configuration may redirect the installation out of --target.
    for key in list(environment):
        if key.startswith("PIP_"):
            environment.pop(key)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    path_items = [str(path) for path in paths]
    if environment.get("PYTHONPATH"):
        path_items.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(path_items)
    if offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    return environment


def _rs_stop_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(OSError):
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=3)


def _rs_run_process(command, log_path, environment, cancel_event=None, callback=None, timeout=_RS_INSTALL_TIMEOUT):
    _rs_check_cancel(cancel_event)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _rs_log(log_path, "Worker command: " + _rs_safe_text(json.dumps(list(map(str, command)), ensure_ascii=False)))
    offset = log_path.stat().st_size if log_path.exists() else 0
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    started, fragment, recent, last_notify = time.monotonic(), "", "", 0.0
    with open(log_path, "ab", buffering=0) as writer:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=writer, stderr=subprocess.STDOUT,
                                   env=environment, **options)
        try:
            with open(log_path, "rb") as reader:
                reader.seek(offset)
                while True:
                    _rs_check_cancel(cancel_event)
                    if time.monotonic() - started > timeout:
                        raise _RS_Failure("timeout", f"Worker exceeded {timeout} seconds")
                    chunk = reader.read(65536).decode("utf-8", errors="replace")
                    recent = (recent + chunk)[-65536:]
                    fragment += chunk
                    lines = fragment.split("\n")
                    fragment = lines.pop()
                    for line in lines:
                        match = _RS_PROGRESS_PATTERN.search(line)
                        if callback and match and time.monotonic() - last_notify >= 0.15:
                            current, total = int(match[1]), int(match[2])
                            if total > 0:
                                _rs_notify(callback, indeterminate=False, current=current, total=total,
                                           downloaded_bytes=current, total_bytes=total, unit="bytes", scope="file")
                                last_notify = time.monotonic()
                    code = process.poll()
                    if code is not None:
                        recent = (recent + reader.read().decode("utf-8", errors="replace"))[-65536:]
                        _rs_log(log_path, f"Worker exited {code}; elapsed={time.monotonic() - started:.1f}s")
                        return code, recent
                    if cancel_event is not None:
                        cancel_event.wait(0.1)
                    else:
                        time.sleep(0.1)
        except BaseException:
            _rs_stop_process(process)
            raise


def _rs_probe(paths=None, *, log_path=None, callback=None, cancel_event=None, profile="core", include_gui=True):
    log_path = log_path or _rs_new_log()
    paths = _rs_layers() if paths is None else list(paths)
    jobs = _rs_mkdir(_rs_root() / "jobs")
    with tempfile.TemporaryDirectory(prefix="probe-", dir=jobs) as directory:
        result_file = Path(directory) / "result.json"
        command = _rs_worker_command("--unmark-runtime-probe", "--result-file", result_file, "--log-file", log_path, "--ignore-overlay", "--profile", profile)
        if not include_gui:
            command.append("--no-gui-probe")
        for path in paths:
            command += ["--packages", str(path)]
        _rs_notify(callback, "실행에 필요한 구성요소를 확인합니다.")
        code, tail = _rs_run_process(command, log_path, _rs_child_environment(paths, offline=True),
                                     cancel_event, timeout=_RS_PROBE_TIMEOUT)
        try:
            result = _rs_read_json(result_file)
            if not isinstance(result, dict) or not isinstance(result.get("ready"), bool) or not isinstance(result.get("missing"), list):
                raise ValueError("Invalid runtime probe result")
        except (OSError, ValueError, TypeError) as exc:
            return _rs_error(_rs_error_kind(tail), log_path, raw=f"Probe returned {code}; {exc}\n{tail}")
        if code != 0 and result["ready"]:
            return _rs_error("verification", log_path, raw=f"Probe reported success but exited {code}\n{tail}")
        result["log_path"] = str(log_path)
        return result


def runtime_dependency_snapshot(*, profile="core"):
    """Fast metadata-only observation for choosing the startup screen.

    This never imports Torch/Diffusers or starts a probe. Even when every
    distribution is present, ready remains False until the actual API check
    succeeds. Missing packages can therefore show consent at once, before any
    cold model-library import or model download.
    """
    missing, versions = [], {}
    try:
        activate_runtime_packages()
        unsupported = _rs_platform_error()
        if unsupported:
            result = _rs_error("unsupported", raw=unsupported)
            result["metadata_only"] = True
            return result
        requirements = {name: constraint for name, (_, constraint) in _rs_requirements(profile).items()}
        if _rs_frozen():
            requirements["pip"] = ">=25.2,<27"
        for name, constraint in requirements.items():
            try:
                version = importlib.metadata.version(name)
                versions[name] = version
                if not _rs_version_ok(version, constraint):
                    missing.append(name)
            except (importlib.metadata.PackageNotFoundError, OSError, ValueError, TypeError):
                missing.append(name)
        return {"ready": False, "metadata_only": True, "missing": missing,
                "versions": versions, "checks": {},
                "error_code": "dependencies" if missing else "",
                "error": "실행 구성요소의 준비가 필요합니다." if missing else "",
                "detail": "필요한 구성요소를 함께 준비합니다." if missing else "실행 구성을 확인하고 있습니다.",
                "restart_required": False, "cancelled": False}
    except Exception as exc:
        result = _rs_error(getattr(exc, "code", _rs_error_kind(exc)), raw=traceback.format_exc())
        result["metadata_only"] = True
        return result


def runtime_dependency_status(*, cancel_event=None, progress_callback=None, profile="core", include_gui=True):
    """Run the actual import/API check in a cancellable background worker."""
    log_path = None
    try:
        _rs_check_cancel(cancel_event)
        log_path = _rs_new_log()
        activate_runtime_packages()
        return _rs_probe(log_path=log_path, callback=progress_callback, cancel_event=cancel_event,
                         profile=profile, include_gui=include_gui)
    except _RS_Cancelled:
        return _rs_error("cancelled", log_path)
    except Exception as exc:
        return _rs_error(getattr(exc, "code", _rs_error_kind(exc)), log_path, raw=traceback.format_exc())


@contextlib.contextmanager
def _rs_installation_lock(cancel_event, callback):
    acquired = False
    try:
        while not acquired:
            _rs_check_cancel(cancel_event)
            acquired = _RS_THREAD_LOCK.acquire(timeout=0.2)
        path = _rs_mkdir(_rs_root()) / "install.lock"
        with open(path, "a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b"0")
                stream.flush()
            locked, announced = False, False
            try:
                while not locked:
                    _rs_check_cancel(cancel_event)
                    try:
                        stream.seek(0)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                    except (OSError, BlockingIOError):
                        if not announced:
                            _rs_notify(callback, "다른 실행 창에서 준비 중인 구성요소를 기다립니다.")
                            announced = True
                        time.sleep(0.2)
                yield
            finally:
                if locked:
                    if os.name == "nt":
                        import msvcrt
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        if acquired:
            _RS_THREAD_LOCK.release()


def _rs_bootstrap_wheel(log_path, cancel_event):
    spec = importlib.util.find_spec("ensurepip")
    if spec and spec.submodule_search_locations:
        bundled = Path(next(iter(spec.submodule_search_locations))) / "_bundled"
        for wheel in sorted(bundled.glob("pip-*.whl"), reverse=True):
            if _rs_version_ok(wheel.name.split("-")[1], ">=25.2" if sys.version_info[:2] >= (3, 14) else ">=23"):
                return wheel
    # Distributions may omit both pip and ensurepip. Download a verified pure
    # Python pip wheel into this application's tooling folder, without installing
    # it into either the interpreter or the user's shared site-packages.
    tooling = _rs_mkdir(_rs_root() / "tooling")
    request = urllib.request.Request(f"https://pypi.org/pypi/pip/{_RS_PIP_BOOTSTRAP_VERSION}/json", headers={"User-Agent": "UNMARK-runtime-setup"})
    _rs_check_cancel(cancel_event)
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.loads(response.read(4 * 1024 * 1024))
    wheel = next(item for item in metadata["urls"] if item["filename"].endswith("-py3-none-any.whl"))
    url = urllib.parse.urlsplit(wheel["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
        raise _RS_Failure("installer", "Unexpected pip wheel download host")
    expected = wheel["digests"]["sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise _RS_Failure("installer", "Pip wheel has no valid SHA-256 digest")
    target = tooling / Path(wheel["filename"]).name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return target
    part = target.with_suffix(".part")
    digest, total = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(wheel["url"], timeout=30) as response, open(part, "wb") as output:
            while True:
                _rs_check_cancel(cancel_event)
                block = response.read(262144)
                if not block:
                    break
                total += len(block)
                if total > 16 * 1024 * 1024:
                    raise _RS_Failure("installer", "Pip bootstrap wheel is unexpectedly large")
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected:
            raise _RS_Failure("installer", "Pip wheel integrity verification failed")
        os.replace(part, target)
        _rs_log(log_path, f"Verified isolated pip bootstrap wheel: {target.name}")
        return target
    finally:
        with contextlib.suppress(OSError):
            part.unlink()


def _rs_make_pip_command(arguments, job, log_path, paths, wheel=None):
    if _rs_frozen():
        request = job / ("pip-request-" + uuid.uuid4().hex + ".json")
        _rs_atomic_json(request, {"arguments": arguments, "layers": [str(path) for path in paths],
                                  "log_file": str(log_path), "result_file": str(request.with_suffix(".result.json")),
                                  "pip_wheel": str(wheel) if wheel else "",
                                  "pip_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest() if wheel else ""})
        return _rs_worker_command("--unmark-install-runtime", "--request-file", request)
    if wheel is None:
        return [sys.executable, "-m", "pip", *arguments]
    code = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
    return [sys.executable, "-c", code, str(wheel), *arguments]


def _rs_run_pip_command(command, log_path, paths, cancel_event, callback=None, timeout=_RS_INSTALL_TIMEOUT):
    code, tail = _rs_run_process(command, log_path, _rs_child_environment(paths), cancel_event, callback, timeout)
    if "--unmark-install-runtime" in command:
        try:
            request = _rs_read_json(Path(command[command.index("--request-file") + 1]))
            result = _rs_read_json(Path(request["result_file"]))
            if result.get("error") and result["error"] not in tail:
                tail += "\n" + result["error"]
            if code == 0 and result.get("exit_code") != 0:
                code = int(result.get("exit_code") or 1)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            tail += f"\nInstaller result file unavailable: {exc}"
            code = code or 1
    return code, tail


def _rs_pip_health(job, log_path, paths, cancel_event, wheel=None):
    command = _rs_make_pip_command(["--version"], job, log_path, paths, wheel)
    code, tail = _rs_run_pip_command(command, log_path, paths, cancel_event, timeout=60)
    match = re.search(r"(?m)^pip (\S+) from ", tail)
    minimum = ">=25.2" if sys.version_info[:2] >= (3, 14) else ">=23"
    if code or not match or not _rs_version_ok(match[1], minimum):
        raise _RS_Failure("installer", f"pip CLI check failed (exit {code}; required {minimum})\n{tail}")
    # --version alone does not load the install command or all its vendored
    # dependencies. This read-only check catches incomplete frozen pip bundles.
    command = _rs_make_pip_command(["--isolated", "--disable-pip-version-check", "install", "--help"], job, log_path, paths, wheel)
    code, help_text = _rs_run_pip_command(command, log_path, paths, cancel_event, timeout=60)
    if code or "--dry-run" not in help_text or "--report" not in help_text:
        raise _RS_Failure("installer", f"pip install command check failed (exit {code})\n{help_text}")
    progress_help = help_text.partition("--progress-bar")[2][:500]
    return {"version": match[1], "wheel": str(wheel) if wheel else "", "raw_progress": bool(re.search(r"\braw\b", progress_help))}


def _rs_select_pip(job, log_path, paths, cancel_event, force_bootstrap=False):
    selected_path = job / "installer.json"
    if selected_path.is_file() and not force_bootstrap:
        return _rs_read_json(selected_path)
    if not force_bootstrap:
        try:
            selected = _rs_pip_health(job, log_path, paths, cancel_event)
            _rs_atomic_json(selected_path, selected)
            return selected
        except _RS_Failure as exc:
            _rs_log(log_path, "Existing pip is not executable; trying an isolated pip wheel.\n" + str(exc))
    wheel = _rs_bootstrap_wheel(log_path, cancel_event)
    selected = _rs_pip_health(job, log_path, paths, cancel_event, wheel)
    _rs_atomic_json(selected_path, selected)
    return selected


def _rs_pip_command(arguments, job, log_path, paths, cancel_event, force_bootstrap=False):
    selected = _rs_select_pip(job, log_path, paths, cancel_event, force_bootstrap)
    arguments = list(arguments)
    if "--progress-bar" in arguments:
        arguments[arguments.index("--progress-bar") + 1] = "raw" if selected["raw_progress"] else "off"
    return _rs_make_pip_command(arguments, job, log_path, paths, selected["wheel"] or None)


def _rs_pip(arguments, job, log_path, paths, callback, cancel_event):
    _rs_notify(callback, "현재 환경에 맞는 구성요소를 확인합니다." if "--dry-run" in arguments else "필요한 구성요소를 설치합니다.")
    command = _rs_pip_command(arguments, job, log_path, paths, cancel_event)
    code, tail = _rs_run_pip_command(command, log_path, paths, cancel_event, callback)
    if code and _rs_error_kind(tail) == "installer" and not _rs_read_json(job / "installer.json").get("wheel"):
        # A late import failure can occur beyond --help; retry the same operation
        # once with the private wheel, never by changing the shared interpreter.
        _rs_log(log_path, "Installed pip failed during execution; retrying once with isolated pip.")
        command = _rs_pip_command(arguments, job, log_path, paths, cancel_event, force_bootstrap=True)
        code, tail = _rs_run_pip_command(command, log_path, paths, cancel_event, callback)
    if code:
        raise _RS_Failure(_rs_error_kind(tail), f"Installer exited {code}\n{tail}")


def _rs_pip_options(job, profile="core"):
    # --target is added only for installation. Resolution uses --dry-run, so
    # healthy existing packages can satisfy dependencies instead of re-fetching
    # a whole Torch/CUDA stack for a missing tqdm or Hugging Face Hub import.
    constraints = job / "constraints.txt"
    constraints.write_text("\n".join(name + constraint for name, (_req, constraint) in _rs_requirements(profile).items()) + "\n", encoding="utf-8")
    cache = _rs_mkdir(_rs_root() / "wheel-cache")
    return ["--isolated", "--disable-pip-version-check", "--no-input", "install",
            "--only-binary=:all:", "--retries", "2", "--timeout", "30",
            "--index-url", "https://pypi.org/simple", "--cache-dir", str(cache),
            "--progress-bar", "off", "--constraint", str(constraints)]


def _rs_read_plan(path):
    report = _rs_read_json(path)
    if report.get("version") != "1" or not isinstance(report.get("install"), list):
        raise _RS_Failure("installer", "Unsupported pip installation report")
    return report["install"]


def _rs_canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _rs_plan(status, job, log_path, paths, callback, cancel_event, force_clean=False, *, profile="core"):
    options = _rs_pip_options(job, profile)
    report = job / "plan.json"
    required = _rs_requirements(profile)
    if _rs_frozen():
        # Repair an incomplete bundled installer in the same external generation.
        # The bootstrapping worker itself can use the verified standalone wheel.
        required["pip"] = ("pip>=25.2,<27", ">=25.2,<27")
    requirements = [value[0] for value in required.values()]
    resolve = ["--dry-run", "--report", str(report)]
    if force_clean:
        # A damaged transitive wheel can retain valid installed metadata. One
        # final complete resolution bypasses that metadata and repairs the
        # required dependency closure in a new generation, never in shared sites.
        resolve.append("--ignore-installed")
    _rs_pip(options + [*resolve, *requirements], job, log_path, paths, callback, cancel_event)
    plan = {_rs_canonical(item["metadata"]["name"]): item for item in _rs_read_plan(report)}
    # Installed metadata can exist while an import/native extension is broken.
    # Repair those wheels explicitly, but keep the resolver's selected upgrades.
    repair = []
    for name in status.get("missing", []):
        if name not in required or name in plan:
            continue
        version = status.get("versions", {}).get(name)
        constraint = required[name][1]
        repair.append(name + "==" + version if version and _rs_version_ok(version, constraint) else required[name][0].replace("[visible]", ""))
    if repair:
        repair_report = job / "repair-plan.json"
        _rs_pip(options + ["--dry-run", "--ignore-installed", "--no-deps", "--report", str(repair_report), *repair],
                job, log_path, paths, callback, cancel_event)
        for item in _rs_read_plan(repair_report):
            plan.setdefault(_rs_canonical(item["metadata"]["name"]), item)
    if not plan:
        raise _RS_Failure("verification", "Imports failed but the resolver produced no repairable wheel")
    lines = []
    for name, item in sorted(plan.items()):
        info = item.get("download_info", {})
        url = info.get("url", "")
        parsed = urllib.parse.urlsplit(url)
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        if (parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org"
                or not parsed.path.endswith(".whl") or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
            raise _RS_Failure("installer", f"Refusing an unverified/non-wheel artifact for {name}")
        lines.append(f"{name} @ {url} --hash=sha256:{digest}")
    wheels = job / "wheels.txt"
    wheels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan, wheels


def _rs_loaded_replacements(stage, loaded):
    top_names = {path.name for path in stage.iterdir() if path.is_dir() and path.name.isidentifier()}
    top_names.update(path.stem for path in stage.glob("*.py") if path.stem.isidentifier())
    return sorted(name for name in top_names if name in loaded)


def ensure_runtime_dependencies(progress_callback=None, cancel_event=None, *, profile="core", include_gui=True):
    """Install missing/broken required wheels privately, then prove readiness.

    At most two attempts are made. Existing installations and model caches are
    never modified. A changed native package already imported in this process
    requires an application restart; the result explicitly reports that case.
    """
    log_path, missing = None, []
    try:
        _rs_requirements(profile)  # Validate before any installation or mutation.
        log_path = _rs_new_log()
        _rs_log(log_path, f"Runtime setup: profile={profile}, Python {platform.python_version()}, ABI {_rs_abi()}, frozen={_rs_frozen()}")
        _rs_check_cancel(cancel_event)
        unsupported = _rs_platform_error()
        if unsupported:
            return _rs_error("unsupported", log_path, raw=unsupported)
        with _rs_installation_lock(cancel_event, progress_callback):
            activate_runtime_packages()
            status = _rs_probe(log_path=log_path, callback=progress_callback, cancel_event=cancel_event,
                               profile=profile, include_gui=include_gui)
            if status["ready"] or status.get("error_code") == "unsupported":
                return status
            missing = status.get("missing", [])
            root = _rs_root()
            packages, jobs = _rs_mkdir(root / "packages"), _rs_mkdir(root / "jobs")
            last_failure = None
            for attempt in range(_RS_MAX_ATTEMPTS):
                _rs_check_cancel(cancel_event)
                paths = _rs_layers()
                generation = None
                with tempfile.TemporaryDirectory(prefix="install-", dir=jobs) as directory:
                    job = Path(directory)
                    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=packages))
                    try:
                        if shutil.disk_usage(root).free < 256 * 1024 * 1024:
                            raise _RS_Failure("disk_space", "Less than 256 MiB available in the runtime volume")
                        plan, wheel_requirements = _rs_plan(status, job, log_path, paths, progress_callback, cancel_event,
                                                           force_clean=attempt > 0, profile=profile)
                        options = _rs_pip_options(job, profile)
                        _rs_pip(options + ["--target", str(stage), "--no-deps", "--require-hashes", "--no-compile",
                                           "--no-warn-script-location", "--requirement", str(wheel_requirements)],
                                job, log_path, paths, progress_callback, cancel_event)
                        _rs_check_cancel(cancel_event)
                        replaced_loaded = _rs_loaded_replacements(stage, set(sys.modules))
                        _rs_atomic_json(stage / ".unmark-runtime.json", {"abi": _rs_abi(), "complete": True,
                                         "created_at": time.time(), "distributions": sorted(plan)})
                        generation = packages / ("g-" + uuid.uuid4().hex)
                        os.replace(stage, generation)
                        candidate_paths = [generation, *paths]
                        checked = _rs_probe(candidate_paths, log_path=log_path, callback=progress_callback,
                                            cancel_event=cancel_event, profile=profile, include_gui=include_gui)
                        if not checked["ready"]:
                            status = checked
                            missing = checked.get("missing", [])
                            raise _RS_Failure(checked.get("error_code") if checked.get("error_code") == "unsupported" else "verification",
                                              "Installed runtime failed validation:\n" + _rs_failed_checks(checked))
                        _rs_check_cancel(cancel_event)
                        old_manifest = {"schema": 1, "abi": _rs_abi(), "layers": [path.name for path in paths]}
                        _rs_atomic_json(root / "current.json", {"schema": 1, "abi": _rs_abi(),
                                        "layers": [path.name for path in candidate_paths]})
                        activate_runtime_packages()
                        try:
                            # Use the actual published manifest in a fresh process.
                            final = _rs_probe(log_path=log_path, callback=progress_callback, cancel_event=cancel_event,
                                              profile=profile, include_gui=include_gui)
                            if not final["ready"]:
                                raise _RS_Failure("verification", "Published runtime failed its final import check:\n" + _rs_failed_checks(final))
                        except BaseException:
                            _rs_atomic_json(root / "current.json", old_manifest)
                            activate_runtime_packages()
                            raise
                        generation = None  # Published: never remove a live generation.
                        final.update(installed=True, installed_distributions=sorted(plan), runtime_dir=str(root / "packages"),
                                     restart_required=bool(replaced_loaded), restart_modules=replaced_loaded)
                        if replaced_loaded:
                            final.update(ready=False, error_code="restart_required",
                                         error="새 구성요소를 적용하려면 앱을 다시 실행해야 합니다.",
                                         detail="설치를 마쳤습니다. 새 구성요소를 적용합니다.")
                        else:
                            _rs_notify(progress_callback, "실행에 필요한 구성요소를 준비했습니다.")
                        return final
                    except _RS_Failure as exc:
                        last_failure = exc
                        _rs_log(log_path, f"Attempt {attempt + 1}/{_RS_MAX_ATTEMPTS}: {exc}")
                        if exc.code in {"permission", "disk_space", "installer", "unsupported", "dependency_conflict", "installer_environment"}:
                            break
                        if attempt + 1 < _RS_MAX_ATTEMPTS:
                            _rs_notify(progress_callback, "준비를 마치지 못한 구성요소를 다시 확인합니다.")
                    finally:
                        shutil.rmtree(stage, ignore_errors=True)
                        if generation is not None:
                            shutil.rmtree(generation, ignore_errors=True)
            return _rs_error(getattr(last_failure, "code", "verification"), log_path, missing,
                             str(last_failure) if last_failure else "Runtime setup did not finish")
    except _RS_Cancelled:
        return _rs_error("cancelled", log_path, missing)
    except Exception as exc:
        return _rs_error(getattr(exc, "code", _rs_error_kind(exc)), log_path, missing, traceback.format_exc())


def ensure_model_dependencies(progress_callback=None, cancel_event=None):
    """Prepare the local three-model desktop and CLI runtime."""
    return ensure_runtime_dependencies(progress_callback, cancel_event, profile="models")


def _rs_cli_arguments(flag, argv):
    if argv is not None:
        return list(argv)
    arguments = list(sys.argv[1:])
    if arguments and arguments[0] == flag:
        arguments.pop(0)
    return arguments


def runtime_probe_cli(argv=None):
    """Worker entry: main must route this before importing third-party modules."""
    parser = argparse.ArgumentParser(description="AI Unmarker Pro runtime validation worker")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--packages", type=Path, action="append", default=[])
    parser.add_argument("--ignore-overlay", action="store_true")
    parser.add_argument("--profile", choices=("core", "models"), default="core")
    parser.add_argument("--no-gui-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(_rs_cli_arguments("--unmark-runtime-probe", argv))
    log_path = args.log_file
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    sink = open(log_path or os.devnull, "a", encoding="utf-8", buffering=1)
    with sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            _rs_activate(args.packages if args.ignore_overlay or args.packages else _rs_layers())
            os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
            result = _rs_probe_actual(args.profile, include_gui=not args.no_gui_probe)
        except BaseException as exc:
            traceback.print_exc()
            result = _rs_error(getattr(exc, "code", _rs_error_kind(exc)), raw=str(exc))
        if log_path:
            result["log_path"] = str(log_path)
        if args.result_file:
            _rs_atomic_json(args.result_file, result)
    if not args.result_file and sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ready"] else 1


def run_runtime_installer_cli(argv=None):
    """Run bundled pip once in a dedicated worker, never `app.exe -m pip`."""
    parser = argparse.ArgumentParser(description="AI Unmarker Pro private installer worker")
    parser.add_argument("--request-file", type=Path, required=True)
    args = parser.parse_args(_rs_cli_arguments("--unmark-install-runtime", argv))
    root = _rs_root()
    request_path = args.request_file.resolve()
    if not request_path.is_relative_to(root / "jobs"):
        return 2
    request = _rs_read_json(request_path)
    arguments = request.get("arguments", [])
    log_path, result_path = Path(request["log_file"]).resolve(), Path(request["result_file"]).resolve()
    if not log_path.is_relative_to(root / "logs") or not result_path.is_relative_to(root / "jobs"):
        return 2
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code, error = 2, ""
        start_offset = log_path.stat().st_size
        try:
            health_check = arguments in (["--version"], ["--isolated", "--disable-pip-version-check", "install", "--help"])
            if (not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments)
                    or (not health_check and ("install" not in arguments or "--only-binary=:all:" not in arguments))
                    or any(value in arguments for value in ("--user", "--prefix", "--root", "--python", "-e", "--editable"))):
                raise ValueError("Invalid installer arguments")
            if "--target" in arguments:
                target = Path(arguments[arguments.index("--target") + 1]).resolve()
                if not target.is_relative_to(root / "packages") or not target.name.startswith(".staging-"):
                    raise ValueError("Installer target must be an app-owned staging directory")
            elif "--dry-run" not in arguments and not health_check:
                raise ValueError("Only private --target installs or read-only resolution are permitted")
            _rs_activate(request.get("layers", []))
            if request.get("pip_wheel"):
                wheel = Path(request["pip_wheel"]).resolve()
                allowed_roots = [root / "tooling"]
                spec = importlib.util.find_spec("ensurepip")
                if spec and spec.submodule_search_locations:
                    allowed_roots.append((Path(next(iter(spec.submodule_search_locations))) / "_bundled").resolve())
                if (wheel.suffix != ".whl" or not any(wheel.is_relative_to(base) for base in allowed_roots)
                        or wheel.stat().st_size > 16 * 1024 * 1024
                        or hashlib.sha256(wheel.read_bytes()).hexdigest() != request.get("pip_sha256")):
                    raise ValueError("Invalid isolated pip wheel")
                # This is a disposable worker; no application DLLs are reloaded.
                for name in list(sys.modules):
                    if name == "pip" or name.startswith("pip."):
                        sys.modules.pop(name, None)
                sys.path.insert(0, str(wheel))
                importlib.invalidate_caches()
            print("pip arguments: " + _rs_safe_text(json.dumps(arguments, ensure_ascii=False)))
            sys.argv = ["pip", *arguments]
            try:
                runpy.run_module("pip", run_name="__main__", alter_sys=True)
                code = 0
            except SystemExit as exc:
                code = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        sink.flush()
        if code and not error:
            with open(log_path, "rb") as reader:
                reader.seek(max(start_offset, log_path.stat().st_size - 65536))
                error = _rs_error_cause(reader.read().decode("utf-8", errors="replace"))
        _rs_atomic_json(result_path, {"exit_code": code, "error": error, "error_code": _rs_error_kind(error) if code else ""})
    return code


def _rs_cli_main():
    if "--unmark-runtime-probe" in sys.argv[1:2]:
        return runtime_probe_cli()
    if "--unmark-install-runtime" in sys.argv[1:2]:
        return run_runtime_installer_cli()
    raise SystemExit("This module is an internal AI Unmarker Pro runtime worker.")

# Installer/probe workers must run before any external GUI/image imports.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--unmark-install-runtime":
        raise SystemExit(run_runtime_installer_cli())
    if len(sys.argv) > 1 and sys.argv[1] == "--unmark-runtime-probe":
        raise SystemExit(runtime_probe_cli())
if "--ignore-overlay" not in sys.argv:
    activate_runtime_packages()


# file_publish.py
# ------------------------------------------------------------------------
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile


_PUBLISH_BLOCK_SIZE = 2 * 1024 * 1024
_LINK_UNSUPPORTED = {errno.EXDEV, errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP}


def _publication_identity(info):
    return info.st_dev, info.st_ino


def _publication_signature(info):
    return (*_publication_identity(info), info.st_size, info.st_mtime_ns)


def _rename_no_replace(source, target):
    """Atomically rename or fail; never fall back to a replacing POSIX rename."""
    if sys.platform == "win32":
        # Windows os.rename always rejects an existing destination.
        os.rename(source, target)
        return
    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if b"\0" in source_bytes or b"\0" in target_bytes:
        raise ValueError("저장 경로에 null 문자를 사용할 수 없습니다.")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, target_bytes, 0x00000004)
    else:
        rename = None
    if rename is None:
        raise OSError(errno.ENOTSUP, "이 환경은 기존 파일을 보호하는 새 파일 저장을 지원하지 않습니다.", str(target))
    if result != 0:
        code = ctypes.get_errno() or errno.EIO
        if code in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}:
            raise OSError(code, "이 파일 시스템은 기존 파일을 보호하는 새 파일 저장을 지원하지 않습니다.", str(target))
        raise OSError(code, os.strerror(code), str(target))


def publish_new_file(staged_path, target_path):
    """Return the new target Path; never overwrite it or remove the source.

    Existing files, directories and dangling symlinks fail with FileExistsError.
    Only unsupported-hard-link errors enable the copy fallback. Copying happens
    in an O_EXCL-created private file; a no-replace rename publishes it after
    verification. No code path unlinks or replaces the public destination.
    The caller guarantees that the source is a private, immutable staged file.
    """
    source = Path(staged_path).expanduser().resolve(strict=True)
    target = Path(target_path).expanduser().absolute()
    with source.open("rb") as src:
        original = os.fstat(src.fileno())
        if not stat.S_ISREG(original.st_mode):
            raise ValueError("저장할 임시 파일이 일반 파일이 아닙니다.")
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in _LINK_UNSUPPORTED:
                raise
        else:
            linked = target.lstat()
            if (not stat.S_ISREG(linked.st_mode)
                    or _publication_identity(linked) != _publication_identity(original)
                    or _publication_signature(os.fstat(src.fileno())) != _publication_signature(original)):
                # Do not unlink the public name: a concurrent writer could
                # replace it between an identity check and unlink.
                raise RuntimeError("저장하는 동안 파일 또는 저장 경로가 변경되었습니다.")
            return target

        with tempfile.TemporaryDirectory(prefix=".unmarker-publish-", dir=target.parent) as directory:
            private = Path(directory) / "complete"
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(private, flags, 0o666)
            try:
                dst = os.fdopen(descriptor, "w+b")
            except BaseException:
                os.close(descriptor)
                raise
            with dst:
                expected = hashlib.sha256()
                total = 0
                for block in iter(lambda: src.read(_PUBLISH_BLOCK_SIZE), b""):
                    if dst.write(block) != len(block):
                        raise OSError(errno.EIO, "저장 파일을 끝까지 쓰지 못했습니다.")
                    expected.update(block)
                    total += len(block)
                dst.flush()
                os.fsync(dst.fileno())
                if (total != original.st_size
                        or _publication_signature(os.fstat(src.fileno())) != _publication_signature(original)
                        or _publication_signature(source.stat()) != _publication_signature(original)):
                    raise RuntimeError("저장하는 동안 임시 파일이 변경되었습니다.")
                dst.seek(0)
                actual = hashlib.sha256()
                for block in iter(lambda: dst.read(_PUBLISH_BLOCK_SIZE), b""):
                    actual.update(block)
                if (os.fstat(dst.fileno()).st_size != total or actual.digest() != expected.digest()):
                    raise RuntimeError("저장 파일의 내용 검증에 실패했습니다.")
                copied = os.fstat(dst.fileno())
            _rename_no_replace(private, target)
            published = target.lstat()
            if (not stat.S_ISREG(published.st_mode)
                    or _publication_identity(published) != _publication_identity(copied)):
                raise RuntimeError("저장하는 동안 저장 경로가 변경되었습니다.")
    return target


# metadata_core.py
# ------------------------------------------------------------------------
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
import time
from xml.etree import ElementTree
import zlib

try:
    from PIL import Image, ImageChops, ImageOps
except ImportError:
    # The first-run setup must be able to open before Pillow is available.
    Image = ImageChops = ImageOps = None


def _require_pillow():
    global Image, ImageChops, ImageOps
    if Image is None:
        try:
            from PIL import Image, ImageChops, ImageOps
        except ImportError as exc:
            raise RuntimeError("이미지 구성요소를 준비하지 못했습니다. 앱을 다시 열어 준비를 완료해 주세요.") from exc


class MetadataError(ValueError):
    """Invalid/unsupported image or unsuccessful verification."""


_C2PA_UUID = bytes.fromhex("6332706100110010800000aa00389b71")
_KEYS = ("exif", "xmp", "text", "icc", "c2pa", "jumbf", "comment", "other")
_MAX_METADATA = 64 * 1024 * 1024
_MAX_XMP = 2 * 1024 * 1024
_MAX_C2PA_JSON = 16 * 1024 * 1024
_MAX_OBSERVATIONS = 256
_AI_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    "https://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "https://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
}
_WATERMARK_ACTIONS = {"c2pa.watermarked", "c2pa.watermarked.bound", "c2pa.watermarked.unbound"}


def _metadata_text(value, limit=1024):
    if isinstance(value, bytes):
        value = value[:limit * 4].decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return ""
    return " ".join(value[:limit * 4].replace("\x00", " ").split())[:limit]


def _record_software(result, value, source, manifest_label=None, *,
                     scope="file_metadata", signature_verified=False,
                     role="file_software"):
    if isinstance(value, dict):
        name = _metadata_text(value.get("name"), 80)
        version = _metadata_text(value.get("version"), 40)
    else:
        name, version = _metadata_text(value, 80), ""
    if not name or len(result["recorded_apps"]) >= _MAX_OBSERVATIONS:
        return
    item = {"name": name, "source": source, "scope": scope,
            "signature_verified": signature_verified is True, "role": role}
    if version:
        item["version"] = version
    if manifest_label:
        item["manifest_label"] = manifest_label
    if item not in result["recorded_apps"]:
        result["recorded_apps"].append(item)


def _external_reference(result, value, source):
    uri = _metadata_text(value, 4096)
    # Legacy self#jumbf references refer to an embedded store, not a remote one.
    if not uri or uri.startswith("self#jumbf="):
        return
    references = result["metadata"]["c2pa"]["external_references"]
    item = {"uri": uri, "source": source, "fetched": False}
    if item not in references and len(references) < 32:
        references.append(item)


def _xmp_error(result, code, source):
    errors = result["metadata"]["xmp"].setdefault("container_errors", [])
    item = {"code": code, "source": source}
    if item not in errors and len(errors) < 32:
        errors.append(item)


def _scan_xmp(payload, result, source):
    """Read namespace-qualified XMP fields; never treat a text mention as C2PA."""
    if not payload:
        return []
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if len(payload) > _MAX_XMP:
        _xmp_error(result, "xmp_packet_limit", source)
        return []
    # ElementTree does not fetch external entities. Also reject internal DTDs
    # rather than allowing their expansion in an untrusted metadata packet.
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        _xmp_error(result, "xmp_entities_not_allowed", source)
        return []
    try:
        root = ElementTree.fromstring(payload)
    except (ElementTree.ParseError, ValueError, TypeError):
        _xmp_error(result, "invalid_xmp", source)
        return []
    provenance = "{http://purl.org/dc/terms/}provenance"
    software_fields = {
        "{http://ns.adobe.com/xap/1.0/}CreatorTool": "xmp:CreatorTool",
        "{http://ns.adobe.com/tiff/1.0/}Software": "tiff:Software",
        "{http://ns.adobe.com/xap/1.0/sType/ResourceEvent#}softwareAgent": "stEvt:softwareAgent",
    }
    source_type = "{http://iptc.org/std/Iptc4xmpExt/2008-02-29/}DigitalSourceType"
    extended = "{http://ns.adobe.com/xmp/note/}HasExtendedXMP"
    resource = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
    references = []
    for index, node in enumerate(root.iter()):
        if index >= 4096:
            _xmp_error(result, "xmp_node_limit", source)
            break
        fields = list(node.attrib.items())
        if node.tag in (provenance, source_type, extended) or node.tag in software_fields:
            fields.append((node.tag, node.attrib.get(resource) or node.text or ""))
        for name, value in fields:
            if name == provenance:
                _external_reference(result, value, source + ":dcterms:provenance")
            elif name in software_fields:
                _record_software(result, value, source + ":" + software_fields[name])
            elif name == extended:
                guid = str(value).strip().upper()
                if re.fullmatch(r"[0-9A-F]{32}", guid) and guid not in references and len(references) < 16:
                    references.append(guid)
                elif not re.fullmatch(r"[0-9A-F]{32}", guid):
                    _xmp_error(result, "invalid_extended_xmp_reference", source)
            elif name == source_type and value in _AI_SOURCE_TYPES:
                item = {"source": source, "field": "Iptc4xmpExt:DigitalSourceType",
                        "value": value, "declaration": True, "pixel_detection": False,
                        "signature_verified": False, "scope": "file_metadata"}
                records = result["metadata"]["c2pa"]["ai_assertions"]
                if item not in records and len(records) < _MAX_OBSERVATIONS:
                    records.append(item)
    return references


def _jpeg_extended_xmp(packets, references, result):
    """Read only complete extension streams named by the standard XMP packet.

    Adobe's JPEG handler stores a 32-byte GUID, total UInt32 length and UInt32
    offset after the 35-byte extension signature. Pieces can arrive out of
    order. The GUID is an MD5 content identifier, not a signature/trust check.
    """
    source = "APP1:Extended XMP"
    groups = {}
    reference_set = set(references)
    for packet in packets:
        if len(packet) < 40:
            _xmp_error(result, "short_extended_xmp_header", source)
            continue
        guid = packet[:32].decode("ascii", errors="replace").upper()
        if not re.fullmatch(r"[0-9A-F]{32}", guid):
            _xmp_error(result, "invalid_extended_xmp_guid", source)
            continue
        if guid not in reference_set:
            _xmp_error(result, "unreferenced_extended_xmp", source)
            continue
        total, offset = struct.unpack_from(">II", packet, 32)
        part = packet[40:]
        group = groups.setdefault(guid, {"total": total, "parts": {}, "bytes": 0, "invalid": False})
        if (not 0 < total <= _MAX_XMP or not part or offset + len(part) > total
                or group["total"] != total or offset in group["parts"]
                or group["bytes"] + len(part) > _MAX_XMP):
            group["invalid"] = True
            _xmp_error(result, "invalid_extended_xmp_fragment", source)
            continue
        group["parts"][offset] = part
        group["bytes"] += len(part)
    for guid in reference_set:
        group = groups.get(guid)
        if group is None:
            _xmp_error(result, "missing_extended_xmp", source)
            continue
        if group["invalid"]:
            continue
        offset = 0
        ordered = []
        for start, part in sorted(group["parts"].items()):
            if start != offset:
                break
            ordered.append(part)
            offset += len(part)
        if offset != group["total"]:
            _xmp_error(result, "incomplete_extended_xmp", source)
            continue
        content = b"".join(ordered)
        if hashlib.md5(content, usedforsecurity=False).hexdigest().upper() != guid:
            _xmp_error(result, "extended_xmp_digest_mismatch", source)
            continue
        _scan_xmp(content, result, source)


def _png_text(payload, kind):
    """Decode a bounded PNG text payload after its keyword."""
    parts = payload.split(b"\x00", 1)
    if len(parts) != 2:
        return None
    body = parts[1]
    compressed = False
    if kind == b"zTXt":
        if not body or body[0] != 0:
            return None
        body, compressed = body[1:], True
    elif kind == b"iTXt":
        if len(body) < 2 or body[0] not in (0, 1) or body[1] != 0:
            return None
        compressed = bool(body[0])
        parts = body[2:].split(b"\x00", 2)  # language, translated keyword, text
        if len(parts) != 3:
            return None
        body = parts[2]
    if not compressed:
        return body if len(body) <= _MAX_XMP else None
    try:
        decoder = zlib.decompressobj()
        text = decoder.decompress(body, _MAX_XMP + 1)
        return text if len(text) <= _MAX_XMP and decoder.eof and not decoder.unconsumed_tail else None
    except zlib.error:
        return None


def _mark(result, category, location):
    item = result["metadata"][category]
    item["present"] = True
    item["count"] += 1
    if len(item["locations"]) < 100:
        item["locations"].append(location)


def _c2pa_box(data, start=0, end=None, depth=0):
    """Match a bounded JUMBF description, never a free text substring."""
    if depth > 16:
        return False
    end = len(data) if end is None else end
    if start < 0 or end > len(data) or start > end:
        return False
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        header = 8
        if size == 1:
            if pos + 16 > end:
                return False
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return False
        box_end = pos + size
        payload = pos + header
        if kind == b"jumd" and box_end - payload >= 22:
            if data[payload:payload + 16] == _C2PA_UUID:
                flags = data[payload + 16]
                # ISO 19566-5: bit 0 means Requestable, not an inserted ID.
                # The null-terminated label follows the toggles. An optional
                # ID (bit 2) and 32-byte signature (bit 3) follow that label.
                # Keep all reads inside this description, including its tail.
                required = payload + 22 + (4 if flags & 4 else 0) + (32 if flags & 8 else 0)
                if (flags & 2 and required <= box_end
                        and data[payload + 17:payload + 22] == b"c2pa\x00"):
                    return True
        if kind == b"jumb" and _c2pa_box(data, payload, box_end, depth + 1):
            return True
        pos += size
    return False


def _metadata_bytes(data, start, end):
    if end - start > _MAX_METADATA:
        raise MetadataError("메타데이터 블록이 64 MiB를 초과해 안전하게 중단했습니다.")
    return data[start:end]


def _jpeg_jumbf(packets, result):
    """Reassemble JPEG XT APP11 packets by instance and sequence number.

    LBox/TBox are repeated in *every* packet. Concatenating entire packet
    payloads, or asking a box parser to read only the first 64 KiB, is wrong.
    This structural fallback also works when the optional C2PA SDK is missing.
    """
    groups = {}
    errors = []
    for packet in packets:
        if len(packet) < 16:
            errors.append("short_packet")
            continue
        instance, sequence = struct.unpack_from(">HI", packet, 2)
        length = struct.unpack_from(">I", packet, 8)[0]
        header_size = 16 if length == 1 else 8
        if len(packet) < 8 + header_size:
            errors.append("short_box_header")
            continue
        if length == 1:
            length = struct.unpack_from(">Q", packet, 16)[0]
        if length > _MAX_METADATA or (length and length < header_size):
            errors.append("invalid_box_length")
            continue
        header = packet[8:8 + header_size]
        group = groups.get(instance)
        if group is None:
            if sequence != 1 or len(groups) >= _MAX_OBSERVATIONS:
                errors.append("missing_first_packet")
                continue
            groups[instance] = {"data": bytearray(packet[8:]), "header": header,
                                "sequence": sequence, "length": length, "invalid": False}
        elif sequence != group["sequence"] + 1 or header != group["header"]:
            group["invalid"] = True
            errors.append("packet_sequence_or_header")
        else:
            group["sequence"] = sequence
            group["data"].extend(packet[8 + header_size:])
            if len(group["data"]) > _MAX_METADATA:
                group["invalid"] = True
                group["data"].clear()
                errors.append("metadata_limit")
    for instance, group in groups.items():
        data = group["data"]
        if group["invalid"] or (group["length"] and len(data) != group["length"]):
            errors.append("incomplete_box")
            continue
        if _c2pa_box(data):
            _mark(result, "c2pa", "APP11:JUMBF c2pa description")
            result["metadata"]["c2pa"]["manifest_store_recognized"] = True
    if errors:
        result["metadata"]["c2pa"]["container_errors"] = list(dict.fromkeys(errors))[:32]


def _parse_png(data, result):
    pos = 8
    found_end = False
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        start, end = pos + 8, pos + 8 + length
        if end + 4 > len(data):
            raise MetadataError("PNG 청크가 잘려 있습니다.")
        name = kind.decode("ascii", errors="replace")
        if kind == b"IHDR":
            if length != 13:
                raise MetadataError("PNG IHDR 길이가 잘못되었습니다.")
            result["bit_depth"] = data[start + 8]
        elif kind == b"eXIf":
            _mark(result, "exif", name)
        elif kind == b"iCCP":
            _mark(result, "icc", name)
        elif kind == b"caBX":
            _mark(result, "c2pa", name)
            _mark(result, "jumbf", name)
            result["metadata"]["c2pa"]["manifest_store_recognized"] |= _c2pa_box(
                _metadata_bytes(data, start, end))
        elif kind in (b"tEXt", b"zTXt", b"iTXt"):
            payload = _metadata_bytes(data, start, end)
            keyword = payload.split(b"\x00", 1)[0]
            _mark(result, "text", name + ":" + keyword.decode("latin-1", errors="replace"))
            if keyword == b"XML:com.adobe.xmp":
                _mark(result, "xmp", name)
                _scan_xmp(_png_text(payload, kind), result, name + ":XMP")
            if keyword.lower() == b"raw profile type exif":
                _mark(result, "exif", name)
        elif kind == b"acTL":
            result["animated"] = True
        elif kind == b"IEND":
            found_end = True
            if end + 4 < len(data):
                result["warnings"].append("PNG IEND 뒤에 추가 데이터가 있습니다. 재저장 시 제거됩니다.")
            break
        elif kind not in (b"IDAT", b"PLTE", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"sBIT", b"bKGD", b"pHYs", b"fcTL", b"fdAT"):
            _mark(result, "other", name)
        # Validate non-image metadata headers/data; Pillow checks image decoding.
        if kind not in (b"IDAT", b"fdAT"):
            payload = _metadata_bytes(data, start, end)
            expected = struct.unpack_from(">I", data, end)[0]
            actual = zlib.crc32(payload, zlib.crc32(kind)) & 0xffffffff
            if expected != actual:
                raise MetadataError("PNG %s 청크의 CRC가 일치하지 않습니다." % name)
        pos = end + 4
    if not found_end:
        raise MetadataError("PNG 종료 청크가 없습니다.")


def _parse_jpeg(data, result):
    pos = 2
    ended = False
    jumbf_packets = []
    packet_bytes = 0
    extended_xmp = []
    extended_references = []
    xmp_bytes = 0
    while pos < len(data):
        if data[pos] != 0xff:
            raise MetadataError("JPEG 마커가 잘못되었습니다.")
        while pos < len(data) and data[pos] == 0xff:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker == 0xd9:
            ended = True
            if pos < len(data):
                result["warnings"].append("JPEG 종료 마커 뒤에 추가 데이터가 있습니다. 재저장 시 제거됩니다.")
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd8:
            continue
        if pos + 2 > len(data):
            raise MetadataError("JPEG 세그먼트 길이가 잘렸습니다.")
        length = struct.unpack_from(">H", data, pos)[0]
        if length < 2 or pos + length > len(data):
            raise MetadataError("JPEG 세그먼트 범위를 벗어났습니다.")
        start, end = pos + 2, pos + length
        payload = data[start:end]
        if marker == 0xe1:
            if payload.startswith(b"Exif\x00\x00"):
                _mark(result, "exif", "APP1:Exif")
            elif payload.startswith((b"http://ns.adobe.com/xap/1.0/\x00", b"http://ns.adobe.com/xmp/extension/\x00")):
                _mark(result, "xmp", "APP1:XMP")
                if payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                    references = _scan_xmp(payload.split(b"\x00", 1)[1], result, "APP1:XMP")
                    extended_references.extend(guid for guid in references if guid not in extended_references)
                    extended_references = extended_references[:16]
                else:
                    xmp_bytes += len(payload)
                    if xmp_bytes <= _MAX_METADATA and len(extended_xmp) < 2048:
                        extended_xmp.append(payload[35:])
                    else:
                        _xmp_error(result, "extended_xmp_total_limit", "APP1:Extended XMP")
            else:
                _mark(result, "other", "APP1")
        elif marker == 0xe2 and payload.startswith(b"ICC_PROFILE\x00"):
            _mark(result, "icc", "APP2:ICC_PROFILE")
        elif marker == 0xeb and payload.startswith(b"JP"):
            _mark(result, "jumbf", "APP11:JP")
            packet_bytes += len(payload)
            if packet_bytes > _MAX_METADATA:
                raise MetadataError("JPEG JUMBF 메타데이터가 64 MiB를 초과했습니다.")
            jumbf_packets.append(payload)
        elif marker == 0xfe:
            _mark(result, "comment", "COM")
            _mark(result, "text", "COM")
        elif 0xe1 <= marker <= 0xef and marker != 0xee:
            _mark(result, "other", "APP%d" % (marker - 0xe0))
        elif marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            if payload:
                result["bit_depth"] = payload[0]
        pos = end
        if marker == 0xda:
            # Entropy data uses FF00 escaping and unlengthened restart markers.
            while pos < len(data):
                next_ff = data.find(b"\xff", pos)
                if next_ff < 0:
                    raise MetadataError("JPEG 스캔 종료 마커가 없습니다.")
                j = next_ff + 1
                while j < len(data) and data[j] == 0xff:
                    j += 1
                if j >= len(data):
                    raise MetadataError("JPEG 스캔이 잘렸습니다.")
                if data[j] == 0 or 0xd0 <= data[j] <= 0xd7:
                    pos = j + 1
                    continue
                pos = next_ff
                break
    if not ended:
        raise MetadataError("JPEG 종료 마커가 없습니다.")
    _jpeg_jumbf(jumbf_packets, result)
    _jpeg_extended_xmp(extended_xmp, extended_references, result)


def _parse_webp(data, result):
    limit = struct.unpack_from("<I", data, 4)[0] + 8
    if limit > len(data) or limit < 12:
        raise MetadataError("WebP RIFF 길이가 잘못되었습니다.")
    if limit < len(data):
        result["warnings"].append("WebP RIFF 뒤에 추가 데이터가 있습니다. 재저장 시 제거됩니다.")
    pos = 12
    while pos < limit:
        if pos + 8 > limit:
            raise MetadataError("WebP 청크 헤더가 잘렸습니다.")
        kind = data[pos:pos + 4]
        length = struct.unpack_from("<I", data, pos + 4)[0]
        start, end = pos + 8, pos + 8 + length
        if end + (length & 1) > limit:
            raise MetadataError("WebP 청크가 잘렸습니다.")
        category = {b"EXIF": "exif", b"XMP ": "xmp", b"ICCP": "icc", b"C2PA": "c2pa"}.get(kind)
        if category:
            _mark(result, category, kind.decode("ascii").strip())
        if kind == b"XMP ":
            _scan_xmp(_metadata_bytes(data, start, end), result, "WebP:XMP")
        if kind == b"C2PA":
            _mark(result, "jumbf", "C2PA")
            result["metadata"]["c2pa"]["manifest_store_recognized"] |= _c2pa_box(
                _metadata_bytes(data, start, end))
        elif kind in (b"ANIM", b"ANMF"):
            result["animated"] = True
        elif kind not in (b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"EXIF", b"XMP ", b"ICCP"):
            _mark(result, "other", kind.decode("ascii", errors="replace"))
        pos = end + (length & 1)


def _validation_entries(branch):
    branch = branch if isinstance(branch, dict) else {}
    groups = {}
    for name in ("success", "informational", "failure"):
        items = branch.get(name)
        groups[name] = [{key: _metadata_text(item.get(key), 512)
                        for key in ("code", "url", "explanation") if item.get(key)}
                       for item in (items[:_MAX_OBSERVATIONS] if isinstance(items, list) else [])
                       if isinstance(item, dict)]
    return groups


def _signature_validated(branch):
    groups = _validation_entries(branch)
    success = {item.get("code") for item in groups["success"]}
    failed = {item.get("code", "") for item in groups["failure"]}
    return "claimSignature.validated" in success and not any(
        code.startswith("claimSignature.") for code in failed)


def _c2pa_report(report, result):
    """Summarize parsed claims without equating credentials with AI pixels."""
    credentials = result["metadata"]["c2pa"]
    manifests = report.get("manifests")
    active = report.get("active_manifest")
    if not isinstance(manifests, dict) or not manifests or not isinstance(active, str) or active not in manifests:
        raise ValueError("C2PA reader did not return an active manifest")
    validation = report.get("validation_results") or {}
    active_validation = validation.get("activeManifest") if isinstance(validation, dict) else {}
    groups = _validation_entries(active_validation)
    verified = _signature_validated(active_validation)
    failures = {entry.get("code", "") for entry in groups["failure"]}
    successes = {entry.get("code", "") for entry in groups["success"]}
    credentials.update(status="detected", sdk_status="ready", engine="c2pa-python.Reader",
                       manifest_store_recognized=True, manifest_count=len(manifests),
                       active_manifest=_metadata_text(active, 1024), signature_verified=verified)
    if not credentials["present"]:
        _mark(result, "c2pa", "C2PA Reader:embedded manifest")
    if not result["metadata"]["jumbf"]["present"]:
        _mark(result, "jumbf", "C2PA Reader:embedded manifest")
    credentials["signature_status"] = "verified" if verified else (
        "invalid" if any(code.startswith("claimSignature.") for code in failures) else "not_verified")
    credentials["validation"] = {
        "performed": True, "state": _metadata_text(report.get("validation_state"), 40) or "unknown",
        **groups, "network_access": False,
        "signer_trusted": True if "signingCredential.trusted" in successes else (
            False if "signingCredential.untrusted" in failures else None),
        "asset_binding_verified": any(code in successes for code in (
            "assertion.dataHash.match", "assertion.boxesHash.match", "assertion.bmffHash.match"))
            and not any("Hash.mismatch" in code or "Hash.invalid" in code for code in failures),
    }
    active_manifest = manifests[active]
    signature = active_manifest.get("signature_info") if isinstance(active_manifest, dict) else {}
    credentials["signer"] = {key: _metadata_text(signature.get(key), 512)
                             for key in ("issuer", "common_name", "alg", "time")
                             if isinstance(signature, dict) and signature.get(key)}

    # Only follow the active manifest's ingredient graph. An unrelated manifest
    # appended to the store must not supply a claim about this image's origin.
    queue = [(active, "active_manifest", active_validation)]
    seen = set()
    while queue and len(seen) < 64:
        label, scope, branch = queue.pop(0)
        if label in seen or label not in manifests or not isinstance(manifests[label], dict):
            continue
        seen.add(label)
        manifest = manifests[label]
        label_text = _metadata_text(label, 1024)
        signed = _signature_validated(branch)
        info = manifest.get("claim_generator_info")
        info = info[:64] if isinstance(info, list) else []
        generator = _metadata_text(manifest.get("claim_generator"), 512)
        credentials["manifests"].append({"label": label_text, "scope": scope,
            "title": _metadata_text(manifest.get("title"), 256),
            "claim_generator": generator, "signature_verified": signed})
        for agent in info:
            if not isinstance(agent, dict):
                continue
            _record_software(result, agent, "C2PA claim_generator_info", label_text,
                             scope=scope, signature_verified=signed, role="claim_generator")
        if generator and not info:
            _record_software(result, generator, "C2PA claim_generator", label_text,
                             scope=scope, signature_verified=signed, role="claim_generator")
        assertions = manifest.get("assertions")
        for assertion in assertions[:_MAX_OBSERVATIONS] if isinstance(assertions, list) else []:
            if not isinstance(assertion, dict):
                continue
            assertion_label = _metadata_text(assertion.get("label"), 160)
            data = assertion.get("data")
            if not re.fullmatch(r"c2pa\.actions(?:\.v\d+)?(?:__\d+)?", assertion_label) or not isinstance(data, dict):
                continue
            actions, templates = data.get("actions"), data.get("templates")
            templates = templates[:64] if isinstance(templates, list) else []
            for index, raw in enumerate(actions[:_MAX_OBSERVATIONS] if isinstance(actions, list) else []):
                if not isinstance(raw, dict) or len(credentials["actions"]) >= _MAX_OBSERVATIONS:
                    continue
                action_name = _metadata_text(raw.get("action"), 160)
                # v2 templates supply common softwareAgent/description fields.
                merged = {}
                for template in templates:
                    if isinstance(template, dict) and template.get("action") in ("", action_name):
                        for key in ("softwareAgent", "description", "digitalSourceType"):
                            if key in template:
                                merged[key] = template[key]
                merged.update(raw)
                description = _metadata_text(merged.get("description"), 2048)
                source_type = _metadata_text(merged.get("digitalSourceType"), 512)
                agent = merged.get("softwareAgent")
                agent_name = _metadata_text(agent.get("name") if isinstance(agent, dict) else agent, 160)
                origin = {"manifest_label": label_text, "scope": scope,
                          "assertion": assertion_label, "action_index": index,
                          "signature_verified": signed}
                observation = {**origin, "action": action_name, "description": description,
                               "digitalSourceType": source_type, "softwareAgent": agent_name}
                credentials["actions"].append(observation)
                if agent_name:
                    _record_software(result, agent, "C2PA action.softwareAgent", label_text,
                        scope=scope, signature_verified=signed,
                        role="ai_generator" if action_name == "c2pa.created" and source_type in _AI_SOURCE_TYPES
                             else "action_software")
                if source_type in _AI_SOURCE_TYPES and len(credentials["ai_assertions"]) < _MAX_OBSERVATIONS:
                    credentials["ai_assertions"].append({**origin, "source": "C2PA action",
                        "field": "digitalSourceType", "value": source_type,
                        "declaration": True, "pixel_detection": False})
                if action_name in _WATERMARK_ACTIONS:
                    if len(credentials["watermark_declarations"]) < _MAX_OBSERVATIONS:
                        credentials["watermark_declarations"].append({**origin, "field": "action",
                            "value": action_name, "name": "", "scheme": None,
                            "declaration": True, "pixel_detection": False})
                    negated = re.search(r"(?:\b(?:removed?|stripped|without|no|not)\b.{0,60}\bsynthid\b|"
                        r"\bsynthid\b.{0,60}(?:\b(?:removed|stripped|absent|not present|not detected)\b|제거|없음|미검출))",
                        description, re.IGNORECASE)
                    if (re.search(r"\bsynthid\b", description, re.IGNORECASE) and not negated
                            and len(credentials["watermark_declarations"]) < _MAX_OBSERVATIONS):
                        credentials["watermark_declarations"].append({**origin, "field": "description",
                            "value": description, "name": "SynthID", "scheme": "SynthID", "action": action_name,
                            "declaration": True, "pixel_detection": False})
        ingredients = manifest.get("ingredients")
        for ingredient in ingredients[:64] if isinstance(ingredients, list) else []:
            if not isinstance(ingredient, dict):
                continue
            reference = ingredient.get("active_manifest")
            if not isinstance(reference, str) or reference not in manifests:
                continue
            checks = ingredient.get("validation_results") or {}
            queue.append((reference, "ingredient", checks.get("activeManifest") if isinstance(checks, dict) else {}))
    credentials["software_agents"] = [dict(item) for item in result["recorded_apps"]
                                      if item.get("source", "").startswith("C2PA")]


def _scan_c2pa(path, result):
    """Read/validate embedded credentials locally, with remote access disabled.

    A stream is deliberately used: no automatic sidecar lookup and no upload.
    The optional native C2PA reader has no Torch/model/GPU dependency.
    """
    credentials = result["metadata"]["c2pa"]
    if credentials["present"]:
        credentials["signature_status"] = "not_verified"
    try:
        import c2pa
    except Exception as exc:
        credentials.update(sdk_status="unavailable", status=("detected" if credentials["manifest_store_recognized"] else
            "external_reference" if credentials["external_references"] else "unavailable"),
            error=_metadata_text(f"{type(exc).__name__}: {exc}", 1024))
        return
    started = time.perf_counter()
    try:
        with c2pa.Context.from_dict({"verify": {"remote_manifest_fetch": False,
                "ocsp_fetch": False, "verify_after_reading": True}}) as context:
            result["capabilities"]["c2pa_signature_validation"] = True
            mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[result["format"]]
            with Path(path).open("rb") as stream, c2pa.Reader(mime, stream, context=context) as reader:
                serialized = reader.json()
                if not isinstance(serialized, str) or len(serialized) > _MAX_C2PA_JSON:
                    raise ValueError("C2PA report exceeds the 16 MiB inspection limit")
                report = json.loads(serialized)
                if not isinstance(report, dict):
                    raise ValueError("C2PA reader returned a non-object report")
                _c2pa_report(report, result)
        result["capabilities"]["c2pa_signature_validation"] = True
    except Exception as exc:
        error = _metadata_text(f"{type(exc).__name__}: {exc}", 1024)
        manifest_not_found = getattr(getattr(c2pa, "C2paError", None), "ManifestNotFound", None)
        if isinstance(manifest_not_found, type) and isinstance(exc, manifest_not_found):
            credentials.update(sdk_status="not_detected", status=("invalid" if credentials["present"] or credentials["container_errors"] else
                "external_reference" if credentials["external_references"] else "not_detected"))
            credentials["manifest_count"] = 0
            if credentials["status"] == "invalid":
                credentials["error"] = error
        elif str(exc).startswith("Remote: must fetch remote manifests from url "):
            _external_reference(result, str(exc).split("from url ", 1)[1], "C2PA Reader:external manifest")
            credentials.update(sdk_status="external_reference", status="external_reference")
        else:
            credentials.update(sdk_status="error", status="invalid" if credentials["present"] else "error", error=error)
    finally:
        credentials["elapsed_seconds"] = round(time.perf_counter() - started, 6)


def scan_image(path):
    """Return JSON-serializable observations from actual container structures.

    C2PA presence means an embedded container/description was found. It says
    nothing about signer trust, signature validity, or whether pixels use AI.
    """
    _require_pillow()
    path = Path(path)
    result = {
        "path": str(path), "format": None, "width": None, "height": None,
        "mode": None, "bit_depth": 8, "animated": False, "frame_count": 1,
        "metadata": {key: {"present": False, "count": 0, "locations": []} for key in _KEYS},
        "warnings": [], "recorded_apps": [],
        "capabilities": {"invisible_watermark_detection": False, "c2pa_signature_validation": False,
                         "c2pa_manifest_detection": True},
    }
    result["metadata"]["c2pa"].update(manifest_store_recognized=False,
        signature_verified=False, signature_status="not_present", status="not_detected",
        engine="container_structure", sdk_status="not_tested", manifest_count=None,
        active_manifest=None, manifests=[], actions=[], ai_assertions=[],
        watermark_declarations=[], software_agents=[], signer={}, external_references=[],
        container_errors=[], validation={"performed": False, "network_access": False})
    with path.open("rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            raise MetadataError("비어 있는 파일입니다.")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                result["format"] = "PNG"
                _parse_png(data, result)
            elif data[:2] == b"\xff\xd8":
                result["format"] = "JPEG"
                _parse_jpeg(data, result)
            elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                result["format"] = "WEBP"
                _parse_webp(data, result)
            else:
                raise MetadataError("PNG, JPEG, WebP 이미지만 지원합니다.")
    with Image.open(path) as img:
        result.update(width=img.width, height=img.height, mode=img.mode,
                      frame_count=int(getattr(img, "n_frames", 1)))
        # Display literal software tags as recorded information. A software
        # name does not establish who generated the pixels or prove authenticity.
        if result["metadata"]["exif"]["present"]:
            try:
                exif = img.getexif()
                _record_software(result, exif.get(305), "EXIF Software")
                packet = exif.get(700)  # XMP-in-TIFF/Exif tag, not arbitrary EXIF text.
                if isinstance(packet, (bytes, str)):
                    if not result["metadata"]["xmp"]["present"]:
                        _mark(result, "xmp", "EXIF:XMP tag 700")
                    _scan_xmp(packet, result, "EXIF:XMP tag 700")
            except (OSError, ValueError, TypeError, SyntaxError):
                result["warnings"].append("EXIF의 편집 앱 이름을 읽지 못했습니다.")
        for key, value in img.info.items():
            if isinstance(key, str) and key.casefold() == "software":
                _record_software(result, value, f"{result['format']} Software")
        result["animated"] |= bool(getattr(img, "is_animated", False))
        if result["animated"]:
            result["warnings"].append("애니메이션은 검사만 지원하며 변환은 지원하지 않습니다.")
        if result["bit_depth"] > 8 or img.mode.startswith("I;16") or img.mode in ("I", "F"):
            result["warnings"].append("16비트 이상/부동소수점 이미지는 변환을 지원하지 않습니다.")
        if img.mode == "CMYK":
            result["warnings"].append("CMYK 이미지는 RGB 및 색상 보존을 보장할 수 없어 변환을 지원하지 않습니다.")
    _scan_c2pa(path, result)
    if result["metadata"]["jumbf"]["present"] and not result["metadata"]["c2pa"]["present"]:
        result["warnings"].append("JUMBF 구조가 있습니다. C2PA manifest 식별자는 확인되지 않았습니다.")
    return result


def _oriented_pixels(img):
    """Detach decoded pixels; respect palette/colour-key transparency."""
    oriented = ImageOps.exif_transpose(img)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    if oriented.mode in ("1", "L") and not has_alpha:
        mode = "L"
    elif oriented.mode == "LA":
        mode = "LA"
    else:
        mode = "RGBA" if has_alpha else "RGB"
    pixels = oriented.convert(mode)
    pixels.info.clear()
    return pixels


def _same_pixels(a, b):
    if a.mode != b.mode or a.size != b.size:
        return False
    return all(low == 0 and high == 0 for low, high in ImageChops.difference(a, b).getextrema()) if a.mode in ("RGB", "RGBA", "LA") else ImageChops.difference(a, b).getextrema() == (0, 0)


def clean_image(input_path, output_path, preserve_color_profile=True, *, overwrite=False):
    """Create a verified PNG with only an optional ICC profile retained.

    Pixels are compared to the orientation-corrected decoded source. JPEG/WebP
    compression already present in the input cannot be undone. Input bytes are
    never edited. Existing outputs are rejected unless overwrite=True.
    """
    _require_pillow()
    source = Path(input_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().absolute()
    if output.suffix.lower() != ".png":
        raise MetadataError("출력 파일 확장자는 .png여야 합니다.")
    if output.resolve() == source or (output.exists() and os.path.samefile(source, output)):
        raise MetadataError("원본과 같은 경로에는 저장할 수 없습니다.")
    if output.is_symlink():
        raise MetadataError("심볼릭 링크 경로에는 저장할 수 없습니다.")
    if output.exists() and not overwrite:
        raise FileExistsError("출력 파일이 이미 있습니다: %s" % output)
    if not output.parent.is_dir():
        raise FileNotFoundError("저장 폴더가 없습니다: %s" % output.parent)
    before = scan_image(source)
    if before["animated"] or before["frame_count"] != 1:
        raise MetadataError("애니메이션/다중 프레임 이미지는 원본 보존을 위해 변환하지 않습니다.")
    if before["bit_depth"] > 8 or before["mode"].startswith("I;16") or before["mode"] in ("I", "F", "CMYK"):
        raise MetadataError("8비트 RGB/그레이스케일/팔레트 이미지만 변환할 수 있습니다. 고비트·CMYK는 지원하지 않습니다.")
    with Image.open(source) as raw:
        raw.load()
        profile = raw.info.get("icc_profile") if preserve_color_profile else None
        if preserve_color_profile and before["metadata"]["icc"]["present"] and not profile:
            raise MetadataError("ICC 프로필을 읽을 수 없어 색상 보존을 확인할 수 없습니다.")
        pixels = _oriented_pixels(raw)
    fd, temp_name = tempfile.mkstemp(prefix=".unmarker-", suffix=".png", dir=output.parent)
    os.close(fd)
    try:
        options = {"format": "PNG", "compress_level": 6}
        if profile:
            options["icc_profile"] = profile
        pixels.save(temp_name, **options)
        after = scan_image(temp_name)
        forbidden = ("exif", "xmp", "text", "c2pa", "jumbf", "comment", "other")
        if any(after["metadata"][key]["present"] for key in forbidden):
            raise MetadataError("저장 결과에서 제거 대상 메타데이터가 다시 발견되었습니다.")
        with Image.open(temp_name) as check:
            check.load()
            if not _same_pixels(pixels, check):
                raise MetadataError("저장 전후 RGB/알파 픽셀이 일치하지 않아 저장을 취소했습니다.")
            if (check.info.get("icc_profile") or None) != (profile or None):
                raise MetadataError("ICC 프로필 저장 검증에 실패했습니다.")
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_name, output)
        else:
            # The shared publisher also supports exFAT and cross-device
            # destinations while retaining exclusive, non-overwriting output.
            publish_new_file(temp_name, output)
        after["path"] = str(output)
        return {"output_path": str(output), "format": "PNG", "before": before,
                "after": after, "pixels_verified": True, "icc_preserved": bool(profile),
                "removed_metadata": [key for key in forbidden if before["metadata"][key]["present"]],
                "c2pa_signature_verified": False, "invisible_watermark_processed": False}
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


# visible_bridge.py
# ------------------------------------------------------------------------
import importlib.metadata
import math
import os
from pathlib import Path
import tempfile
from typing import Any



VISIBLE_PACKAGE = "remove-ai-watermarks"
VISIBLE_VERSION = "0.46.0"
VISIBLE_INSTALL = 'python3 -m pip install "remove-ai-watermarks[visible]==0.46.0"'


def _engine():
    installed = importlib.metadata.version(VISIBLE_PACKAGE)
    if installed != VISIBLE_VERSION:
        raise ImportError(f"검증한 엔진 버전은 {VISIBLE_VERSION}입니다. 설치된 버전: {installed}")
    from remove_ai_watermarks import image_io, remove_visible_detailed, watermark_registry
    return image_io, remove_visible_detailed, watermark_registry


def _report(status: str, detail: str, marks=None, **extra) -> dict[str, Any]:
    return {
        "status": status,
        "marks": list(marks or []),
        "detail": detail,
        "engine": f"{VISIBLE_PACKAGE}=={VISIBLE_VERSION}",
        "scope": "registered_visible_templates",
        "synthid_status": "unverified",
        **extra,
    }


def _failure(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (ImportError, importlib.metadata.PackageNotFoundError)):
        return _report("unavailable", f"가시 워터마크 엔진 설치가 필요합니다. {exc}", install=VISIBLE_INSTALL)
    return _report("error", f"{type(exc).__name__}: {exc}")


def _confidence(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _region(value):
    return [int(v) for v in value] if value is not None else None


def app_scan_visible(path: str | os.PathLike) -> dict[str, Any]:
    """Run actual registered template detectors; never turn a failure into absent.

    Strict thresholds and no metadata-based threshold relaxation are used.
    ``not_detected`` means no supported automatic template matched this image.
    """
    try:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        image_io, _, registry = _engine()
        image = image_io.load_image_bgr(source)
        detections = registry.detect_marks(
            image, include_explicit=False, provenance=frozenset(),
        )
        marks = [
            {
                "key": item.key,
                "label": item.label,
                "location": item.location,
                "confidence": _confidence(item.confidence),
                "region": _region(item.region),
            }
            for item in detections if item.detected
        ]
        return _report(
            "detected" if marks else "not_detected",
            "등록된 가시 워터마크 패턴이 검출되었습니다." if marks
            else "검사한 등록 패턴에서 검출되지 않았습니다. 미지원 패턴은 확인할 수 없습니다.",
            marks,
            checked_templates=[item.key for item in detections],
        )
    except Exception as exc:
        return _failure(exc)


def app_remove_visible(input_path: str | os.PathLike,
                       output_path: str | os.PathLike) -> dict[str, Any]:
    """Fill detected regions with OpenCV and retain upstream post-fill statuses.

    Statuses: cleaned / no_watermark / partial / unvalidated / unavailable / error.
    ``cleaned`` records the same detector's post-fill check; it is not an
    independent oracle. Partial/unvalidated candidates are saved for review and
    explicitly have ``verified=False``. No unsupported SynthID claim is made.
    Source is never overwritten. Metadata cleanup belongs to the caller.
    """
    try:
        source = Path(input_path).expanduser().absolute()
        target = Path(output_path).expanduser().absolute()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == target.resolve():
            raise ValueError("원본과 다른 출력 경로를 지정해 주세요.")
        if target.exists() or target.is_symlink():
            raise FileExistsError("같은 이름의 파일이 있습니다. 새 이름으로 저장해 주세요.")
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("출력은 PNG, JPEG, WebP만 지원합니다.")
        image_io, remove_detailed, _ = _engine()
        original, original_alpha = image_io.read_bgr_and_alpha(source)
        if original is None:
            raise ValueError("원본 이미지를 읽을 수 없습니다.")
        from PIL import Image
        import numpy as np
        with Image.open(source) as source_image:
            original_profile = source_image.info.get("icc_profile") or None
        if original_alpha is not None and target.suffix.lower() in {".jpg", ".jpeg"}:
            raise ValueError("투명도가 있는 원본의 출력은 PNG 또는 WebP로 지정해 주세요.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="unmark-visible-", dir=target.parent) as directory:
            staged = Path(directory) / ("result" + target.suffix.lower())
            result = remove_detailed(
                source, staged, sensitivity="strict", backend="cv2",
                strip_metadata=False, write_noop=True,
            )
            status = result.status
            if status not in {"cleaned", "no_watermark", "partial", "unvalidated"}:
                raise RuntimeError(f"알 수 없는 엔진 결과: {status}")
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError("엔진이 출력 파일을 저장하지 못했습니다.")
            saved, saved_alpha = image_io.read_bgr_and_alpha(staged)
            if saved is None or saved.shape[:2] != original.shape[:2]:
                raise RuntimeError("저장된 이미지의 크기 검증에 실패했습니다.")
            with Image.open(staged) as saved_image:
                saved_profile = saved_image.info.get("icc_profile") or None
                saved_exif = saved_image.info.get("exif")
            # The engine normally carries alpha/ICC itself. Recover only these
            # display properties when an engine output drops or changes them;
            # wrong geometry and other failures remain errors.
            repaired = saved_profile != original_profile
            if original_alpha is None:
                repaired = repaired or saved_alpha is not None
            else:
                transparent = original_alpha == 0
                repaired = (repaired or saved_alpha is None
                            or not np.array_equal(saved_alpha, original_alpha)
                            or not np.array_equal(saved[transparent], original[transparent]))
                if repaired:
                    saved = saved.copy()
                    saved[transparent] = original[transparent]
            if repaired:
                if staged.suffix.lower() == ".webp":
                    # WebP must preserve the RGB behind alpha=0 too. Pillow's
                    # exact lossless mode explicitly retains those samples.
                    restored = Image.fromarray(np.ascontiguousarray(saved[..., ::-1]))
                    if original_alpha is not None:
                        restored.putalpha(Image.fromarray(original_alpha))
                    save_options = {"format": "WEBP", "lossless": True, "exact": True}
                    if original_profile:
                        save_options["icc_profile"] = original_profile
                    if saved_exif:
                        save_options["exif"] = saved_exif
                    restored.save(staged, **save_options)
                elif not image_io.write_bgr_with_alpha(
                        staged, saved, original_alpha, display_tags_from=source,
                        orientation_applied=False):
                    raise RuntimeError("투명도와 색상 프로필을 보존한 결과를 저장하지 못했습니다.")
                saved, saved_alpha = image_io.read_bgr_and_alpha(staged)
                if saved is None or saved.shape[:2] != original.shape[:2]:
                    raise RuntimeError("저장된 이미지의 크기 검증에 실패했습니다.")
            if original_alpha is not None:
                if (saved_alpha is None or not np.array_equal(saved_alpha, original_alpha)
                        or not np.array_equal(saved[original_alpha == 0], original[original_alpha == 0])):
                    raise RuntimeError("저장된 이미지의 투명도 검증에 실패했습니다.")
            elif saved_alpha is not None:
                raise RuntimeError("결과에 원본에 없는 투명도가 추가되었습니다.")
            with Image.open(staged) as checked_image:
                if (checked_image.info.get("icc_profile") or None) != original_profile:
                    raise RuntimeError("저장된 이미지의 색상 프로필 검증에 실패했습니다.")
            if repaired and status == "cleaned":
                check = app_scan_visible(staged)
                if check["status"] == "detected":
                    status = "partial"
                elif check["status"] != "not_detected":
                    status = "unvalidated"
            marks = [
                {
                    "key": item.key, "label": item.label,
                    "location": item.location, "status": item.status,
                    "region": _region(item.region),
                    "mask_bbox": _region(item.mask_bbox), "backend": item.backend,
                    "confidence_before": _confidence(item.confidence_before),
                    "confidence_after": _confidence(item.confidence_after),
                    "residual_region": _region(item.residual_region),
                }
                for item in result.marks
            ]
            publish_new_file(staged, target)
        details = {
            "cleaned": "검출 영역을 복원했고 동일 검출기의 재검사에서 해당 패턴이 검출되지 않았습니다.",
            "no_watermark": "자동 제거 대상으로 선택된 등록 패턴이 없어 복원은 적용하지 않았습니다.",
            "partial": "복원 후에도 해당 영역에서 패턴이 검출됩니다. 결과 확인이 필요합니다.",
            "unvalidated": "복원은 실행했지만 재검사에 실패했습니다. 결과 확인이 필요합니다.",
        }
        return _report(status, details[status], marks, output=str(target),
                       applied=bool(marks), verified=(status == "cleaned"))
    except Exception as exc:
        result = _failure(exc)
        result.update(output=None, applied=False, verified=False)
        return result


# model_runtime.py
# ------------------------------------------------------------------------
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


# detail_restore.py
# ------------------------------------------------------------------------
import math


DETAIL_RESTORE_POLICY = {
    "modes": ("gaussian", "edge"),
    "default_mode": "gaussian", "default_sigma": 1.95,
    "default_unsharp": 0.0, "gentle_unsharp": 0.2, "maximum_unsharp": 1.0,
    "edge_sigma": 1.0, "edge_mask_sigma": 2.0, "edge_percentile": 99,
    "normalization_epsilon": 1e-6, "unsharp_sigma": 1.0, "strip_rows": 128,
}


def _detail_edge_weights(source, check_cancel):
    """원본 윤곽 강도를 구하고 이미지 전체의 99백분위로 정규화합니다."""
    import cv2
    import numpy as np
    policy = DETAIL_RESTORE_POLICY
    height, width = source.shape[:2]
    weights = np.empty((height, width), dtype=np.float32)
    # Sobel 한 픽셀과 Gaussian 필터 주변부를 함께 포함해 줄 경계를 없앱니다.
    halo = math.ceil(policy["edge_mask_sigma"] * 4) + 3
    for top in range(0, height, policy["strip_rows"]):
        check_cancel()
        bottom = min(height, top + policy["strip_rows"])
        start, stop = max(0, top - halo), min(height, bottom + halo)
        gray = cv2.cvtColor(source[start:stop], cv2.COLOR_RGB2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.GaussianBlur(np.sqrt(dx * dx + dy * dy), (0, 0),
                                     sigmaX=policy["edge_mask_sigma"], sigmaY=policy["edge_mask_sigma"])
        weights[top:bottom] = magnitude[top - start:bottom - start]
    check_cancel()
    scale = np.percentile(weights, policy["edge_percentile"]) + policy["normalization_epsilon"]
    weights /= scale
    np.clip(weights, 0, 1, out=weights)
    return weights


def restore_image_detail(clean, original, *, sigma, mode="gaussian", unsharp=0.0,
                         check_cancel=lambda: None):
    """복원 → 윤곽 혼합 → 선명화 → uint8 변환 순서를 유지합니다.

선명화 이전에 정수로 변환하면 미세한 오차가 커집니다. 복원할 때 원본의
성분을 다시 합치므로 이 결과가 워터마크를 제거했다고 가정하지 않습니다.
투명도·색상 프로필·파일 저장은 호출하는 Qwen 처리 흐름이 담당합니다.
"""
    import cv2
    import numpy as np
    from PIL import Image
    policy = DETAIL_RESTORE_POLICY
    if (mode not in policy["modes"] or not .1 <= sigma <= 4
            or not 0 <= unsharp <= policy["maximum_unsharp"]):
        raise ValueError("디테일 복원 모드·폭·선명화 강도를 확인해 주세요.")
    check_cancel()
    source = np.asarray(original.convert("RGB"))
    cleaned = np.asarray(clean.convert("RGB"))
    if clean.size != original.size:
        cleaned = cv2.resize(cleaned, original.size, interpolation=cv2.INTER_LANCZOS4)
    weights = _detail_edge_weights(source, check_cancel) if mode == "edge" else None
    result = np.empty_like(source)
    halo = math.ceil(max(sigma, policy["edge_sigma"] if weights is not None else sigma) * 4) + 2
    if unsharp:
        halo += math.ceil(policy["unsharp_sigma"] * 4) + 2

    def restore_band(first, second, width):
        low_source = cv2.GaussianBlur(first, (0, 0), sigmaX=width, sigmaY=width)
        low_clean = cv2.GaussianBlur(second, (0, 0), sigmaX=width, sigmaY=width)
        return np.clip(low_clean + (first - low_source), 0, 255)

    for top in range(0, original.height, policy["strip_rows"]):
        check_cancel()
        bottom = min(original.height, top + policy["strip_rows"])
        start, stop = max(0, top - halo), min(original.height, bottom + halo)
        first, second = source[start:stop].astype(np.float32), cleaned[start:stop].astype(np.float32)
        band = restore_band(first, second, sigma)
        if weights is not None:
            edge_band = restore_band(first, second, policy["edge_sigma"])
            mask = weights[start:stop, :, None]
            band = np.clip(mask * edge_band + (1 - mask) * band, 0, 255)
        if unsharp:
            blurred = cv2.GaussianBlur(band, (0, 0), sigmaX=policy["unsharp_sigma"], sigmaY=policy["unsharp_sigma"])
            band = np.clip(band + unsharp * (band - blurred), 0, 255)
        if not np.isfinite(band).all():
            raise ValueError("이미지 복원 결과에 유효하지 않은 수치가 있습니다.")
        # 마지막에만 절삭합니다. 기본 모드의 기존 픽셀 결과도 유지합니다.
        result[top:bottom] = band[top - start:bottom - start].astype(np.uint8)
    check_cancel()
    return Image.fromarray(result)


# qwen_inference.py
# ------------------------------------------------------------------------
import contextlib
import gc
import hashlib
import math
import os
import secrets
from pathlib import Path
import sys
import tempfile
import time

# Source modules can also be imported individually for diagnostics. The
# distributable assembles this file after model_runtime.py and file_publish.py.
if "ModelRuntimeError" not in globals():
    from model_runtime import (ModelRuntimeError, ModelCancelled, _check_cancel,
                               _notify, _emit_model_progress, _MODEL_LOCK,
                               prepare_models, model_runtime_status, MODEL_SPECS)
if "publish_new_file" not in globals():
    from file_publish import publish_new_file


if "DETAIL_RESTORE_POLICY" not in globals():
    from detail_restore import DETAIL_RESTORE_POLICY, restore_image_detail


MODEL_EXECUTION_POLICY = {
    "engine": "desynth_qwen_gguf",
    "max_working_side": 1024,
    "resize_mode": "pipeline_default",
    "batch_size": 1,
    "threads": 2,
    "passes": 1,
    "steps": 8,
    "strength": .25,
    "true_cfg_scale": 1.0,
    "cpu_dtype": "bfloat16_if_supported",
    "cpu_memory_policy": "mmap_no_offload",
    "mps_memory_policy": "block_offload_preserve_mmap",
    "cuda_memory_policy": "block_offload_preserve_mmap",
    "accelerator_resident_blocks": 1,
    "mps_environment_modified": False,
    "mps_per_process_memory_fraction": .9,
    "conditioning": "pinned_precomputed_embeddings",
}
# 빌드 도구는 위 메모리 제한을 실행 없이 검사합니다. 복원 기본값은 별도 명세에서 가져옵니다.
MODEL_EXECUTION_POLICY.update(
    restore_sigma=DETAIL_RESTORE_POLICY["default_sigma"],
    restore_mode=DETAIL_RESTORE_POLICY["default_mode"],
    unsharp=DETAIL_RESTORE_POLICY["default_unsharp"],
)
_QWEN_EMBEDS_SHA256 = "1d05bbe05797ed9ec1653e935ba257a016c017b50f0ba206394c04782da8cd4c"
_QWEN_GGUF_FILENAME = MODEL_SPECS["qwen"]["files"][0]
_QWEN_LORA_FILENAME = "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
_QWEN_GIB = 1024 ** 3
_QWEN_CPU_DTYPE = None


def _qwen_memory_snapshot():
    """Available RAM, including reclaimable file-backed GGUF pages.

    Linux cgroup limits must take precedence over the much larger host RAM.
    Dirty pages and shared anonymous memory are not counted as reclaimable.
    """
    import psutil
    memory = psutil.virtual_memory()
    result = {"limit_bytes": int(memory.total), "available_bytes": int(memory.available)}
    if sys.platform.startswith("linux"):
        # In a nested PID namespace psutil(os.getpid()) can address a DIFFERENT
        # visible process without raising. /proc/self always names this one.
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    result["process_rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            pass
    else:
        try:
            result["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
        except (psutil.Error, OSError):
            pass
    root = Path("/sys/fs/cgroup")
    try:
        limit = int((root / "memory.max").read_text().strip())
        current = int((root / "memory.current").read_text().strip())
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        reclaimable = max(0, int(stats.get("file", 0)) - int(stats.get("shmem", 0))
                          - int(stats.get("file_dirty", 0)) - int(stats.get("file_writeback", 0)))
        result.update(limit_bytes=min(result["limit_bytes"], limit),
                      available_bytes=min(result["available_bytes"], max(0, limit - current + reclaimable)),
                      cgroup_current_bytes=current, reclaimable_file_bytes=reclaimable)
    except (OSError, ValueError):
        pass
    return result


def _qwen_peak_rss_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, ValueError):
        return None


def _qwen_memory_guard(torch, device, *, loading=False):
    snapshot = _qwen_memory_snapshot()
    required = int((2.5 if loading else .75) * _QWEN_GIB)
    if snapshot["available_bytes"] < required:
        raise ModelRuntimeError(
            "DeSynth 처리에 사용할 여유 메모리가 부족합니다. "
            "다른 앱을 종료한 뒤 다시 시도해 주세요. "
            f"(여유 {snapshot['available_bytes'] / _QWEN_GIB:.1f} GiB)")
    if device == "mps":
        recommended = int(torch.mps.recommended_max_memory())
        limit = int(recommended * MODEL_EXECUTION_POLICY["mps_per_process_memory_fraction"])
        allocated = int(torch.mps.current_allocated_memory())
        driver = int(torch.mps.driver_allocated_memory())
        if limit - driver < int(.75 * _QWEN_GIB):
            torch.mps.empty_cache()
            driver = int(torch.mps.driver_allocated_memory())
        snapshot.update(accelerator_limit_bytes=limit, accelerator_allocated_bytes=allocated,
                        accelerator_driver_allocated_bytes=driver)
        if limit - max(driver, allocated) < int(.75 * _QWEN_GIB):
            raise ModelRuntimeError("Metal 메모리 한도에 가까워 처리를 중단했습니다. 다른 앱을 종료해 주세요.")
    elif device == "cuda":
        free, total = torch.cuda.mem_get_info()
        # Reusable allocator cache is not an active model allocation.
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        snapshot.update(accelerator_limit_bytes=int(total), accelerator_available_bytes=int(free + reusable))
        if free + reusable < int(.75 * _QWEN_GIB):
            raise ModelRuntimeError("그래픽 메모리가 부족해 처리를 중단했습니다. 다른 앱을 종료해 주세요.")
    return snapshot


def _qwen_apply_device_limits(torch, device):
    """Set a worker-local hard limit, even if inherited MPS env disables one.

    Public API: https://docs.pytorch.org/docs/2.14/generated/
    torch.mps.set_per_process_memory_fraction.html . A fraction of zero disables
    the allocator cap; this application always installs a nonzero bound before
    its first MPS tensor allocation. The user's environment is left untouched.
    """
    if device == "mps":
        try:
            torch.mps.set_per_process_memory_fraction(MODEL_EXECUTION_POLICY["mps_per_process_memory_fraction"])
        except (AttributeError, RuntimeError) as exc:
            raise ModelRuntimeError("Metal 처리의 메모리 상한을 설정하지 못했습니다.") from exc


def _qwen_device(torch, requested):
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ModelRuntimeError("처리 장치는 auto, cpu, mps, cuda 중 하나여야 합니다.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelRuntimeError("CUDA 장치를 사용할 수 없습니다.")
    if requested == "mps" and not has_mps:
        raise ModelRuntimeError("Metal 장치를 사용할 수 없습니다.")
    return requested


def _qwen_operator_probe(torch, dtype, device="cpu"):
    """Tiny real convolution, attention and native Q4 dequantization operations."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
    with torch.inference_mode():
        x = torch.ones((1, 3, 1, 8, 8), dtype=dtype, device=device)
        convolution = torch.nn.functional.conv3d(
            x, torch.ones((4, 3, 1, 3, 3), dtype=dtype, device=device))
        q = torch.ones((1, 2, 4, 16), dtype=dtype, device=device)
        attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_K]
        linear = GGUFLinear(block_size, 1, compute_dtype=dtype)
        # A valid all-zero untrained Q4 block; not an image/watermark fixture.
        linear.weight = GGUFParameter(torch.zeros((1, type_size), dtype=torch.uint8, device=device),
                                      quant_type=gguf.GGMLQuantizationType.Q4_K)
        predicted = linear(torch.ones((1, block_size), dtype=dtype, device=device))
        if not all(bool(torch.isfinite(value).all()) for value in (convolution, attention, predicted)):
            raise RuntimeError("Non-finite output from the Qwen operator probe")
    return {"dtype": str(dtype).removeprefix("torch."), "q2_dequantized": True,
            "convolution_executed": True, "attention_executed": True}


def _qwen_compute_dtype(torch, device):
    global _QWEN_CPU_DTYPE
    if device == "cpu" and _QWEN_CPU_DTYPE is not None:
        return getattr(torch, _QWEN_CPU_DTYPE)
    failures = []
    candidates = (torch.bfloat16, torch.float32) if device == "cpu" else (torch.bfloat16, torch.float16)
    for dtype in candidates:
        try:
            _qwen_operator_probe(torch, dtype, device)
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            failures.append(f"{dtype}: {type(exc).__name__}: {exc}")
            continue
        if dtype == torch.float32 and _qwen_memory_snapshot()["available_bytes"] < 5 * _QWEN_GIB:
            raise ModelRuntimeError("이 CPU는 BF16 연산을 지원하지 않습니다. FP32 처리에는 여유 메모리 5 GiB가 필요합니다.")
        if device == "cpu":
            _QWEN_CPU_DTYPE = str(dtype).removeprefix("torch.")
        return dtype
    raise ModelRuntimeError(f"{device.upper()}에서 Qwen 필수 연산을 실행하지 못했습니다. " + " | ".join(failures))


def _qwen_read_mapped_checkpoint(torch, path, readers):
    """Supply a mapped state dict through Diffusers' public single-file API.

    Copy-on-write mapping keeps the cache immutable and gives PyTorch writable
    views without allocating weight-sized heap buffers. Passing the dictionary
    bypasses Diffusers 0.36's default tensor.data.copy(), without a global patch.
    """
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter, SUPPORTED_GGUF_QUANT_TYPES
    reader = gguf.GGUFReader(path, mode="c")
    readers.append(reader)
    result = {}
    for tensor in reader.tensors:
        quantized = tensor.tensor_type not in (gguf.GGMLQuantizationType.F32,
                                               gguf.GGMLQuantizationType.F16)
        if quantized and tensor.tensor_type not in SUPPORTED_GGUF_QUANT_TYPES:
            raise ModelRuntimeError(f"지원하지 않는 GGUF 형식: {tensor.name} / {tensor.tensor_type}")
        value = torch.from_numpy(tensor.data)
        result[tensor.name] = GGUFParameter(value, quant_type=tensor.tensor_type) if quantized else value
    return result


def _qwen_check_mmap_binding(model, readers):
    ranges = [(int(reader.data.__array_interface__["data"][0]), int(reader.data.nbytes)) for reader in readers]
    mapped_bytes = count = 0
    for parameter in model.parameters():
        if not hasattr(parameter, "quant_type"):
            continue
        pointer = parameter.data_ptr()
        size = parameter.numel() * parameter.element_size()
        if parameter.device.type != "cpu" or not any(start <= pointer and pointer + size <= start + length
                                                    for start, length in ranges):
            raise ModelRuntimeError("양자화 모델의 메모리 매핑을 유지하지 못해 실행을 중단했습니다.")
        mapped_bytes += size
        count += 1
    if not count:
        raise ModelRuntimeError("Qwen GGUF 모델에서 양자화 가중치를 읽지 못했습니다.")
    return {"verified": True, "parameter_tensors": count, "mapped_weight_bytes": mapped_bytes}


def _qwen_load_embeddings(path, torch, dtype, device, *, hidden_size=3584, require_pinned=True):
    path = Path(path)
    if not 0 < path.stat().st_size <= 1024 * 1024:
        raise ModelRuntimeError("이미지 처리 임베딩 파일 크기가 올바르지 않습니다.")
    if require_pinned and hashlib.sha256(path.read_bytes()).hexdigest() != _QWEN_EMBEDS_SHA256:
        raise ModelRuntimeError("이미지 처리 임베딩 파일의 무결성을 확인하지 못했습니다.")
    cache = torch.load(str(path), map_location="cpu", weights_only=True)
    expected = {"prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds", "negative_prompt_embeds_mask"}
    if not isinstance(cache, dict) or not expected <= set(cache) or set(cache) - expected - {"meta"}:
        raise ModelRuntimeError("이미지 처리 임베딩 형식이 올바르지 않습니다.")
    result = {}
    for prefix in ("prompt", "negative_prompt"):
        value, mask = cache[prefix + "_embeds"], cache[prefix + "_embeds_mask"]
        if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ModelRuntimeError("이미지 처리 임베딩이 텐서가 아닙니다.")
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if (value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != hidden_size
                or not 1 <= value.shape[1] <= 1024 or tuple(mask.shape) != tuple(value.shape[:2])
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())
                or not bool(((mask == 0) | (mask == 1)).all()) or int(mask.sum()) < 1):
            raise ModelRuntimeError("이미지 처리 임베딩의 크기 또는 수치가 올바르지 않습니다.")
        result[prefix + "_embeds"] = value.to(device=device, dtype=dtype)
        result[prefix + "_embeds_mask"] = mask.to(device=device, dtype=torch.long)
    return result


class _QwenResidentGroup:
    """Move one inference group, then restore its original CPU tensor views.

    Ordinary non-streamed offload copies accelerator weights back onto the CPU
    heap and loses the GGUF mmap. This inference-only hook retains the source
    storage instead. Parameters themselves (including GGUF quant_type) are
    never replaced; exceptions and cancellation restore them too.
    """

    def __init__(self, module, device, before=None, after=None):
        self.module, self.device = module, device
        self.before, self.after = before, after
        unique = dict.fromkeys([*module.parameters(), *module.buffers()])
        self.sources = [(tensor, tensor.data) for tensor in unique]
        if any(source.device.type != "cpu" for _, source in self.sources):
            raise ModelRuntimeError("블록 오프로딩은 CPU 가중치에서 시작해야 합니다.")
        self.active = False
        self.handles = [module.register_forward_pre_hook(self._enter, with_kwargs=True),
                        module.register_forward_hook(self._leave, with_kwargs=True, always_call=True)]

    def _enter(self, module, args, kwargs):
        if self.before is not None:
            self.before()
        if self.active:
            raise ModelRuntimeError("같은 Qwen 블록의 중복 실행을 차단했습니다.")
        self.active = True
        try:
            if self.device != "cpu":
                for parameter, source in self.sources:
                    parameter.data = source.to(device=self.device)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        if self.active and self.device != "cpu":
            for parameter, source in self.sources:
                parameter.data = source
        self.active = False

    def _leave(self, module, args, kwargs, output):
        self.restore()
        if output is not None and self.after is not None:
            self.after()
        return output

    def close(self):
        self.restore()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.sources.clear()
        self.module = None
        self.before = self.after = None


def _qwen_bind_groups(pipe, torch, device, cancel_event, progress_callback, effective_steps):
    groups = []
    blocks = list(pipe.transformer.transformer_blocks)
    counters = {"blocks_completed": 0, "vae_group_calls": 0, "last_emit": 0.0}

    def before():
        _check_cancel(cancel_event)
        _qwen_memory_guard(torch, device)

    def after_block():
        counters["blocks_completed"] += 1
        now = time.monotonic()
        if now - counters["last_emit"] >= 1:
            counters["last_emit"] = now
            total = len(blocks) * effective_steps
            count = counters["blocks_completed"]
            _emit_model_progress(progress_callback, {
                "phase": "inference", "title": "이미지 처리 중",
                "detail": f"Qwen 이미지 재생성 · {count}/{total} 블록",
                "progress": min(1, count / max(1, total)),
                "blocks_completed": count, "total_blocks": total})

    def after_vae():
        counters["vae_group_calls"] += 1

    def before_vae(stage):
        before()
        # Tiled VAE calls repeat the same encoder/decoder. Announce only the
        # actual stage transition and leave its duration indeterminate.
        if counters.get("vae_phase") != stage:
            counters["vae_phase"] = stage
            _emit_model_progress(progress_callback, {
                "phase": stage,
                "title": "원본 이미지 변환 중" if stage == "encode" else "결과 이미지 만드는 중",
                "detail": ("원본 이미지를 처리할 수 있는 형태로 변환합니다."
                           if stage == "encode" else "처리한 결과를 이미지로 변환합니다.")})

    try:
        for block in blocks:
            groups.append(_QwenResidentGroup(block, device, before, after_block))
        for name, child in pipe.transformer.named_children():
            if name != "transformer_blocks":
                groups.append(_QwenResidentGroup(child, device, before))
        # Pipeline calls vae.encode()/decode(), bypassing root forward hooks.
        # These invoked children cover both methods, including tiled calls.
        for name, child in pipe.vae.named_children():
            stage = {"encoder": "encode", "decoder": "decode"}.get(name)
            before_child = (lambda stage=stage: before_vae(stage)) if stage is not None else before
            groups.append(_QwenResidentGroup(child, device, before_child, after_vae))
    except BaseException:
        for group in groups:
            group.close()
        raise
    return groups, counters


def _qwen_pipeline_class():
    from diffusers import QwenImageImg2ImgPipeline

    class BoundedQwenPipeline(QwenImageImg2ImgPipeline):
        @property
        def _execution_device(self):
            # Diffusers 0.36 otherwise infers CPU from offloaded parameters.
            # Its inputs must use the actual device chosen for our hooks.
            import torch
            return torch.device(self._qwen_device)

    return BoundedQwenPipeline


def _qwen_load_pipeline(paths, torch, device, dtype, progress_callback, cancel_event, lora_scale):
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                           GGUFQuantizationConfig, QwenImageTransformer2DModel)
    from safetensors.torch import load_file
    readers = []
    gguf_path = Path(paths["qwen"]) / _QWEN_GGUF_FILENAME
    config_root = Path(paths["vae"])
    _notify(progress_callback, "Qwen 모델 불러오는 중", "양자화 가중치를 디스크에서 메모리 매핑합니다.")
    _check_cancel(cancel_event)
    checkpoint = _qwen_read_mapped_checkpoint(torch, gguf_path, readers)
    transformer = QwenImageTransformer2DModel.from_single_file(
        checkpoint, config=str(config_root), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    del checkpoint
    mapped = _qwen_check_mmap_binding(transformer, readers)
    _check_cancel(cancel_event)
    _qwen_memory_guard(torch, device)
    _notify(progress_callback, "이미지 변환기 불러오는 중", "Qwen VAE와 Lightning 어댑터를 준비합니다.")
    vae = AutoencoderKLQwenImage.from_pretrained(str(config_root), subfolder="vae",
                                                torch_dtype=dtype, local_files_only=True,
                                                low_cpu_mem_usage=True, use_safetensors=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config_root), subfolder="scheduler", local_files_only=True)
    pipe = _qwen_pipeline_class()(scheduler=scheduler, vae=vae, text_encoder=None,
                                  tokenizer=None, transformer=transformer)
    pipe._qwen_device = device
    pipe._qwen_mmaps = readers
    pipe._qwen_mmap_report = mapped
    _check_cancel(cancel_event)
    lora_state = load_file(str(Path(paths["lightning"]) / _QWEN_LORA_FILENAME), device="cpu")
    # Convert only the small adapter, never the packed base-model parameters.
    for name in list(lora_state):
        _check_cancel(cancel_event)
        lora_state[name] = lora_state[name].to(dtype=dtype)
    pipe.load_lora_weights(lora_state, adapter_name="lightning", low_cpu_mem_usage=True)
    del lora_state
    for name, parameter in transformer.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            parameter.data = parameter.data.to(dtype=dtype)
    pipe.set_adapters(["lightning"], adapter_weights=[lora_scale])
    transformer.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    _qwen_check_mmap_binding(transformer, readers)
    return pipe


def _qwen_prepare_image(image, max_side):
    from PIL import Image
    import numpy as np
    scale = min(1., max_side / max(image.size))
    size = tuple(max(1, round(value * scale)) for value in image.size)
    resized = image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image.copy()
    padded_size = tuple(max(32, math.ceil(value / 16) * 16) for value in size)
    array = np.asarray(resized)
    padded = np.pad(array, ((0, padded_size[1] - size[1]), (0, padded_size[0] - size[0]), (0, 0)), mode="edge")
    return Image.fromarray(padded), size


def _qwen_restore_detail(clean, original, sigma, cancel_event=None, *, mode="gaussian", unsharp=0.0):
    """순수 이미지 복원과 처리 중단·앱 오류 표시를 연결합니다."""
    try:
        return restore_image_detail(clean, original, sigma=sigma, mode=mode, unsharp=unsharp,
                                    check_cancel=lambda: _check_cancel(cancel_event))
    except ValueError as exc:
        raise ModelRuntimeError(str(exc)) from exc


def _qwen_compare_pixels(before, after):
    import numpy as np
    first, second = before.convert("RGB"), after.convert("RGB")
    alpha = before.getchannel("A") if "A" in before.getbands() else None
    changed = count = largest = 0
    absolute_sum = square_sum = 0.
    for y in range(0, first.height, 128):
        box = (0, y, first.width, min(y + 128, first.height))
        delta = np.asarray(second.crop(box), dtype=np.int16) - np.asarray(first.crop(box), dtype=np.int16)
        if alpha is not None:
            delta = delta[np.asarray(alpha.crop(box)) > 0]
        if not delta.size:
            continue
        changed += int(np.count_nonzero(np.any(delta != 0, axis=-1)))
        count += delta.size // 3
        absolute_sum += float(np.abs(delta).sum(dtype=np.float64))
        square_sum += float((delta.astype(np.float64) ** 2).sum())
        largest = max(largest, int(np.abs(delta).max()))
    samples = count * 3
    mse = square_sum / samples if samples else 0.
    return {"scope": "rgb_nontransparent_pixels", "compared_pixels": count, "changed_pixels": changed,
            "changed_pixel_ratio": changed / count if count else 0., "identical": changed == 0,
            "mean_absolute_error": absolute_sum / samples if samples else 0.,
            "root_mean_squared_error": math.sqrt(mse), "max_absolute_error": largest,
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "visual_quality_verified": False, "watermark_verified": False}


def _qwen_empty_device_cache(torch, device):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _qwen_preservation_check(comparison):
    """Flag large pixel changes for review; never infer watermark removal.

    30 dB is a conservative product review budget, not a perceptual guarantee
    or a detector threshold. The comparison is against this run's input, so
    it cannot establish cumulative quality against an unavailable original.
    It does not blend in original pixels, rerun inference, or block exporting.
    """
    minimum_psnr_db = 30.0
    maximum_rmse = 255.0 * 10.0 ** (-minimum_psnr_db / 20.0)
    result = {"status": "within_budget", "scope": "rgb_nontransparent_pixels",
              "reference": "current_input", "minimum_psnr_db": minimum_psnr_db,
              "maximum_root_mean_squared_error": maximum_rmse,
              "criterion": "product_pixel_change_review_budget",
              "visual_quality_verified": False, "watermark_verified": False,
              "reasons": []}
    try:
        compared_pixels = int(comparison["compared_pixels"])
        rmse = float(comparison["root_mean_squared_error"])
        if compared_pixels < 0 or not math.isfinite(rmse) or not 0 <= rmse <= 255:
            raise ValueError("Invalid pixel comparison")
    except (KeyError, TypeError, ValueError, OverflowError):
        result.update(status="review_required", reasons=["pixel_comparison_unavailable"])
        return result
    if not compared_pixels:
        result.update(status="not_applicable", reasons=["no_visible_pixels"])
        return result
    result["root_mean_squared_error"] = rmse
    result["psnr_db"] = 20.0 * math.log10(255.0 / rmse) if rmse else None
    if rmse > maximum_rmse:
        result.update(status="review_required", reasons=["pixel_change_exceeds_budget"])
    return result


def _qwen_options(options):
    supplied = dict(options or {})
    try:
        parsed = {"device": supplied.get("device", "auto"),
                  "max_working_side": (int(supplied["max_working_side"])
                                       if supplied.get("max_working_side") is not None else None),
                  "steps": int(supplied.get("steps", 8)), "strength": float(supplied.get("strength", .25)),
                  "lora_scale": float(supplied.get("lora_scale", .8)),
                  "restore_sigma": float(supplied.get("restore_sigma", DETAIL_RESTORE_POLICY["default_sigma"])),
                  "restore_mode": supplied.get("restore_mode", DETAIL_RESTORE_POLICY["default_mode"]),
                  "unsharp": float(supplied.get("unsharp", DETAIL_RESTORE_POLICY["default_unsharp"])),
                  "seed": (int(supplied["seed"]) if supplied.get("seed") is not None else secrets.randbits(63)),
                  "restore": supplied.get("restore", True)}
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelRuntimeError("Qwen 처리 설정의 숫자 형식이 올바르지 않습니다.") from exc
    side = parsed["max_working_side"]
    if ((side is not None and (not 256 <= side <= MODEL_EXECUTION_POLICY["max_working_side"] or side % 16))
            or not isinstance(parsed["restore"], bool) or not 4 <= parsed["steps"] <= 32
            or not .125 <= parsed["strength"] <= .5 or int(parsed["steps"] * parsed["strength"]) < 1
            or not 0 < parsed["lora_scale"] <= 1.5 or not .1 <= parsed["restore_sigma"] <= 4
            or not 0 <= parsed["seed"] < 2 ** 63):
        raise ModelRuntimeError("Qwen 처리 설정 범위를 확인해주세요. 크기 256~1024, 강도 .125~.5, 단계 4~32를 지원합니다.")
    if (parsed["restore_mode"] not in DETAIL_RESTORE_POLICY["modes"]
            or not 0 <= parsed["unsharp"] <= DETAIL_RESTORE_POLICY["maximum_unsharp"]):
        raise ModelRuntimeError("복원 모드는 gaussian 또는 edge, 선명화 강도는 0~1이어야 합니다.")
    if not parsed["restore"] and (parsed["restore_mode"] != "gaussian" or parsed["unsharp"]):
        raise ModelRuntimeError("윤곽 보존·선명화는 디테일 복원을 켠 상태에서 사용해 주세요.")
    return parsed


def regenerate_image(source_png, output_png, options=None, progress_callback=None, cancel_event=None):
    """Execute one real Qwen/Lightning img2img pass, restore detail, publish PNG.

    All models are local verified files. No remote image request, detector
    substitute, automatic repeated inference, or persistent model cache is used.
    """
    started = time.monotonic()
    options = dict(options or {})
    settings = _qwen_options(options)
    _check_cancel(cancel_event)
    source, target = Path(source_png).expanduser().resolve(strict=True), Path(output_png).expanduser().absolute()
    if target.suffix.lower() != ".png" or target.is_symlink() or source == target.resolve():
        raise ModelRuntimeError("원본과 다른 새 PNG 출력 경로를 지정해주세요.")
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    from PIL import Image, ImageOps
    import cv2
    import numpy as np
    with Image.open(source) as raw:
        if (getattr(raw, "n_frames", 1) != 1 or raw.mode in {"CMYK", "I", "F"}
                or raw.mode.startswith("I;16") or raw.width * raw.height > 40_000_000):
            raise ModelRuntimeError("4천만 화소 이하의 8비트 단일 프레임 RGB 이미지가 필요합니다.")
        raw.load()
        icc = raw.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(raw)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        original = oriented.convert("RGBA" if has_alpha else "RGB")
    missing = model_runtime_status().get("missing", [])
    if missing:
        raise ModelRuntimeError("AI 실행 라이브러리가 없습니다: " + ", ".join(missing))
    import torch
    torch.set_num_threads(MODEL_EXECUTION_POLICY["threads"])
    cv2.setNumThreads(MODEL_EXECUTION_POLICY["threads"])
    device = _qwen_device(torch, settings["device"])
    _qwen_apply_device_limits(torch, device)
    _qwen_memory_guard(torch, device, loading=True)
    dtype = _qwen_compute_dtype(torch, device)
    rgb = original.convert("RGB")
    # The reference passes the original PIL image without width/height. Let
    # the pinned pipeline resize it itself; aspect-preserving padding was a
    # different algorithm. A size limit is now an explicit user override.
    if settings["max_working_side"] is None:
        work_image, cropped_size, dimensions = rgb, None, {}
    else:
        work_image, cropped_size = _qwen_prepare_image(rgb, settings["max_working_side"])
        dimensions = {"width": work_image.width, "height": work_image.height}
    effective_steps = int(settings["steps"] * settings["strength"])
    pipe, embeddings, groups = None, None, []
    sample_count = 0
    with _MODEL_LOCK:
        try:
            paths = prepare_models(progress_callback, cancel_event)
            _check_cancel(cancel_event)
            pipe = _qwen_load_pipeline(paths, torch, device, dtype, progress_callback, cancel_event,
                                       settings["lora_scale"])
            embeddings = _qwen_load_embeddings(Path(paths["embeddings"]) / "embeds_cache.pt", torch, dtype, device)
            groups, counters = _qwen_bind_groups(pipe, torch, device, cancel_event, progress_callback, effective_steps)
            mmap_report = dict(pipe._qwen_mmap_report)

            def after_step(pipeline, step, timestep, callback_kwargs):
                _check_cancel(cancel_event)
                _qwen_memory_guard(torch, device)
                if not bool(torch.isfinite(callback_kwargs["latents"]).all()):
                    raise ModelRuntimeError("Qwen 이미지 처리 결과에 유효하지 않은 수치가 있습니다.")
                return callback_kwargs

            default_side = pipe.default_sample_size * pipe.vae_scale_factor
            inference_size = (dimensions.get("width", default_side), dimensions.get("height", default_side))
            _notify(progress_callback, "이미지 처리 중", f"Qwen Q4 이미지 재생성 · {inference_size[0]} × {inference_size[1]}")
            with torch.inference_mode():
                sample_count += 1
                result = pipe(image=work_image, **dimensions,
                              **embeddings, num_inference_steps=settings["steps"], strength=settings["strength"],
                              true_cfg_scale=1., num_images_per_prompt=1,
                              generator=torch.Generator(device="cpu").manual_seed(settings["seed"]),
                              output_type="np", callback_on_step_end=after_step,
                              callback_on_step_end_tensor_inputs=["latents"])
            _check_cancel(cancel_event)
            pixels = np.asarray(result.images[0])
            if (pixels.ndim != 3 or pixels.shape != (inference_size[1], inference_size[0], 3)
                    or not np.issubdtype(pixels.dtype, np.floating) or not np.isfinite(pixels).all()):
                raise ModelRuntimeError("Qwen 출력 이미지의 크기 또는 수치를 확인하지 못했습니다.")
            cleaned = Image.fromarray(np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
            if cropped_size is not None:
                cleaned = cleaned.crop((0, 0, *cropped_size))
            del result, pixels
            completed_blocks = counters["blocks_completed"]
        except (RuntimeError, MemoryError) as exc:
            if isinstance(exc, ModelRuntimeError):
                raise
            text = str(exc).lower()
            if isinstance(exc, MemoryError) or any(token in text for token in ("out of memory", "can't allocate", "cannot allocate")):
                raise ModelRuntimeError("Qwen 처리 중 메모리가 부족했습니다. 다른 앱을 종료하거나 처리 크기를 줄여 주세요.") from exc
            raise
        finally:
            for group in groups:
                group.close()
            groups.clear()
            # Drop every pipeline reference before GC/cache release. The GUI
            # also runs processing in its own child process for final cleanup.
            pipe = None
            embeddings = None
            _qwen_empty_device_cache(torch, device)
    _emit_model_progress(progress_callback, {"phase": "detail_restore", "title": "이미지 디테일 복원 중",
                                            "detail": "원본의 형태·질감과 출력 크기를 복원합니다."})
    if settings["restore"]:
        restored = _qwen_restore_detail(cleaned, rgb, settings["restore_sigma"], cancel_event,
                                        mode=settings["restore_mode"], unsharp=settings["unsharp"])
    else:
        # Keep output geometry consistent even when detail restoration is off.
        restored = Image.fromarray(cv2.resize(np.asarray(cleaned), rgb.size, interpolation=cv2.INTER_LANCZOS4))
    if has_alpha:
        alpha = original.getchannel("A")
        restored.paste(rgb, (0, 0), alpha.point(lambda value: 255 if value == 0 else 0))
        restored.putalpha(alpha)
    comparison = _qwen_compare_pixels(original, restored)
    _check_cancel(cancel_event)
    with tempfile.TemporaryDirectory(prefix=".unmarker-qwen-", dir=target.parent) as directory:
        staged = Path(directory) / "output.png"
        save = {"format": "PNG"}
        if icc:
            save["icc_profile"] = icc
        restored.save(staged, **save)
        with Image.open(staged) as decoded:
            decoded.load()
            if decoded.size != original.size or (decoded.info.get("icc_profile") or None) != (icc or None):
                raise ModelRuntimeError("출력 크기 또는 색상 프로필 저장 검증에 실패했습니다.")
            if has_alpha and not np.array_equal(np.asarray(decoded.getchannel("A")), np.asarray(original.getchannel("A"))):
                raise ModelRuntimeError("투명도 저장 검증에 실패했습니다.")
        _check_cancel(cancel_event)
        publish_new_file(staged, target)
    changed = comparison["changed_pixels"]
    return {"processed": True, "verified": False, "status": "processed" if changed else "unchanged",
            "pixels_changed": bool(changed), "changed_pixels": changed,
            "engine": "DeSynth reference recipe · Qwen-Image-2512 GGUF Q4_K_M + Lightning",
            "modelids": {key: value["repo_id"] for key, value in MODEL_SPECS.items()},
            "device": device, "dtype": str(dtype).removeprefix("torch."),
            "execution_policy": dict(MODEL_EXECUTION_POLICY), "parameters": settings,
            "working_size": list(inference_size), "original_size": list(original.size),
            "resize_mode": "pipeline_default" if not dimensions else "explicit_aspect_preserving_limit",
            "inference_calls": sample_count, "effective_denoising_steps": effective_steps,
            "transformer_blocks_completed": completed_blocks,
            "detail_restore": {"performed": settings["restore"], "method": ("edge_blended_frequency_restore" if settings["restore_mode"] == "edge"
                                          else "gaussian_low_clean_high_original") if settings["restore"] else "none",
                               "mode": settings["restore_mode"] if settings["restore"] else "none",
                               "sigma": settings["restore_sigma"], "unsharp": settings["unsharp"],
                               "sharpening_performed": bool(settings["restore"] and settings["unsharp"])},
            "mmap": mmap_report, "weights_loaded": True, "pixel_comparison": comparison,
            "preservation_check": _qwen_preservation_check(comparison),
            "output_path": str(target), "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_process_rss_bytes": _qwen_peak_rss_bytes(),
            "synthid": {"status": "not_tested", "removal_verified": False},
            "detail": ("이미지 처리 및 디테일 복원 완료" if settings["restore"] else "이미지 재생성 완료 · 디테일 복원 끔") if changed else "Qwen 실행 완료 · 출력 픽셀 변경 없음"}


def _qwen_tiny_engine_probe(torch, dtype):
    """Real, randomly initialized tiny Qwen/GGUF/LoRA/VAE execution, offline."""
    import gguf
    import numpy as np
    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                           GGUFQuantizationConfig, QwenImageTransformer2DModel)
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix="unmarker-qwen-probe-") as directory:
        root = Path(directory)
        transformer = QwenImageTransformer2DModel(
            patch_size=2, in_channels=64, out_channels=16, num_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
            axes_dims_rope=(4, 6, 6))
        transformer.save_config(root / "transformer")
        checkpoint = root / "tiny.gguf"
        writer = gguf.GGUFWriter(str(checkpoint), "qwen_image")
        for name, value in transformer.state_dict().items():
            array = value.detach().float().cpu().numpy()
            if name == "img_in.weight" or name.endswith("attn.to_q.weight"):
                array = gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(name, array, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                writer.add_tensor(name, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        transformer.add_adapter(LoraConfig(r=2, lora_alpha=2, init_lora_weights=False,
                                            target_modules=["to_q", "to_k", "to_v"]), adapter_name="probe")
        lora = {"transformer." + key: value.detach().to(dtype=dtype)
                for key, value in get_peft_model_state_dict(transformer, adapter_name="probe").items()}
        del transformer
        readers = []
        mapped_state = _qwen_read_mapped_checkpoint(torch, checkpoint, readers)
        transformer = QwenImageTransformer2DModel.from_single_file(
            mapped_state, config=str(root), subfolder="transformer", local_files_only=True,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype, low_cpu_mem_usage=True)
        del mapped_state
        mapped = _qwen_check_mmap_binding(transformer, readers)
        vae = AutoencoderKLQwenImage(base_dim=8, z_dim=16, dim_mult=[1, 2, 2, 2],
                                     num_res_blocks=1, temperal_downsample=[False, True, True],
                                     latents_mean=[0.] * 16, latents_std=[1.] * 16).to(dtype=dtype)
        pipe = _qwen_pipeline_class()(vae=vae, transformer=transformer,
                                      scheduler=FlowMatchEulerDiscreteScheduler(), text_encoder=None, tokenizer=None)
        pipe._qwen_device = "cpu"
        pipe.set_progress_bar_config(disable=True)
        pipe.load_lora_weights(lora, adapter_name="lightning", low_cpu_mem_usage=True)
        for name, value in transformer.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                value.data = value.data.to(dtype=dtype)
        transformer.eval().requires_grad_(False)
        vae.eval().requires_grad_(False)
        context = torch.randn((1, 4, 32), dtype=dtype)
        mask = torch.ones((1, 4), dtype=torch.long)
        probe_embeddings = {"prompt_embeds": context[0], "prompt_embeds_mask": mask[0],
                            "negative_prompt_embeds": context[0], "negative_prompt_embeds_mask": mask[0]}
        embed_path = root / "probe-embeddings.pt"
        torch.save(probe_embeddings, embed_path)
        embeddings = _qwen_load_embeddings(embed_path, torch, dtype, "cpu", hidden_size=32, require_pinned=False)
        latent = vae.encode(torch.randn((1, 3, 1, 32, 32), dtype=dtype)).latent_dist.mode()
        decoded = vae.decode(latent).sample
        arguments = {"hidden_states": torch.randn((1, 4, 64), dtype=dtype),
                     "encoder_hidden_states": context, "encoder_hidden_states_mask": mask,
                     "timestep": torch.ones(1, dtype=dtype), "img_shapes": [(1, 2, 2)], "txt_seq_lens": [4]}
        pipe.set_adapters(["lightning"], adapter_weights=[.8])
        positive = transformer(**arguments).sample
        pipe.set_adapters(["lightning"], adapter_weights=[0.])
        disabled = transformer(**arguments).sample
        effect = float((positive.float() - disabled.float()).abs().max())
        pipe.set_adapters(["lightning"], adapter_weights=[.8])
        groups, counters = _qwen_bind_groups(pipe, torch, "cpu", None, None, 2)
        try:
            picture = pipe(image=Image.new("RGB", (32, 32), (83, 117, 151)), width=32, height=32,
                           **embeddings, num_inference_steps=4, strength=.5, true_cfg_scale=1.,
                           generator=torch.Generator(device="cpu").manual_seed(2718), output_type="np").images[0]
        finally:
            for group in groups:
                group.close()
        if counters["vae_group_calls"] < 4 or counters["blocks_completed"] != 2:
            raise RuntimeError("Tiny pipeline did not execute the expected VAE and transformer staging hooks")
        if (not all(bool(torch.isfinite(value).all()) for value in (positive, disabled, decoded))
                or effect <= 0 or picture.shape != (32, 32, 3) or not np.isfinite(picture).all()):
            raise RuntimeError("Tiny Qwen pipeline or Lightning scale produced invalid output")
        restored = _qwen_restore_detail(Image.fromarray(np.uint8(picture * 255)),
                                        Image.new("RGB", (32, 32), (83, 117, 151)), 1.95)
        result = {"dtype": str(dtype).removeprefix("torch."), "finite": True,
                  "output_shape": list(picture.shape), "mmap": mapped,
                  "lora_scale_effect": effect, "vae_decoded_shape": list(decoded.shape),
                  "vae_staging_calls": counters["vae_group_calls"],
                  "transformer_staging_calls": counters["blocks_completed"],
                  "embeddings_weights_only": True, "restored_size": list(restored.size),
                  "parameters": sum(p.numel() for model in (transformer, vae) for p in model.parameters())}
        del pipe, transformer, vae, readers, embeddings, lora
        return result


def model_engine_preflight():
    """Offline CPU engine check with tiny untrained weights; not detection."""
    started = time.monotonic()
    report = {"ready": False, "device": "cpu", "weights_loaded": False,
              "probe_weights": "random_untrained", "checked": [], "bounded_memory": True,
              "watermark_verified": False, "execution_policy": dict(MODEL_EXECUTION_POLICY)}
    try:
        import torch
        from torchvision.ops import nms
        with _MODEL_LOCK, torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.set_num_threads(MODEL_EXECUTION_POLICY["threads"])
            torch.random.default_generator.manual_seed(1729)
            dtype = _qwen_compute_dtype(torch, "cpu")
            report["operators"] = _qwen_operator_probe(torch, dtype)
            report["pipeline"] = _qwen_tiny_engine_probe(torch, dtype)
            selected = nms(torch.tensor([[0., 0., 4., 4.], [.1, .1, 3.9, 3.9], [8., 8., 9., 9.]]),
                           torch.tensor([.9, .8, .7]), .3).tolist()
            if selected != [0, 2]:
                raise RuntimeError("torchvision native NMS produced an unexpected result")
            report["torchvision_nms"] = {"executed": True, "selected_indices": selected}
            report.update(ready=True, checked=["qwen", "gguf", "lightning", "vae", "embeddings", "detail_restore"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


# pixel_engine.py
# ------------------------------------------------------------------------
import math
from pathlib import Path
import tempfile
import time



class PixelProcessingCancelled(RuntimeError):
    pass


def _pixel_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise PixelProcessingCancelled("이미지 처리를 취소했습니다.")


def _pixel_emit(callback, title, detail, completed=0, total=1):
    if callback is not None:
        callback({"phase": "pixel_process", "title": title, "detail": detail,
                  "completed": completed, "total": total,
                  "progress": min(1.0, max(0.0, completed / max(1, total)))})


def scan_frequency_signature(path):
    """Measure local spectral peaks; never report an AI detection percentage.

    A centered crop is windowed to reduce the spectrum introduced by crop
    boundaries. Amplitudes are compared to their radial-band mean. Repeated
    texture, typography, grids and compression can all produce candidates.
    The thresholds describe this heuristic only; they are not a trained
    classifier or a calibrated SynthID test.
    """
    import numpy as np
    from PIL import Image
    started = time.perf_counter()
    source = Path(path).expanduser().resolve(strict=True)
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("단일 프레임 이미지가 필요합니다.")
        width, height = image.size
        crop_width, crop_height = min(512, width), min(512, height)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        crop = image.crop((left, top, left + crop_width, top + crop_height))
        # Do not measure hidden RGB values as if they were visible texture.
        if "A" in crop.getbands() or "transparency" in crop.info:
            rgba = crop.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
            background.alpha_composite(rgba)
            crop = background.convert("RGB")
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
    base = {
        "status": "not_detected", "scope": "spectral_anomaly_heuristic",
        "verified_watermark": False, "peak_count": 0,
        "peak_energy_fraction": 0.0, "high_frequency_energy_fraction": 0.0,
        "maximum_peak_ratio": 0.0, "sample_size": [crop_width, crop_height],
        "sample_box": [left, top, left + crop_width, top + crop_height],
        "algorithm": "hann-radial-peaks-v1", "threshold_ratio": 8.0,
        "detail": "설정한 기준의 반복 주파수 패턴이 없습니다.",
    }
    if min(crop_width, crop_height) < 32:
        base.update(status="insufficient_data", detail="주파수 패턴을 비교하기에는 이미지가 작습니다.")
    else:
        centered = gray - np.mean(gray, dtype=np.float64)
        if float(np.std(centered, dtype=np.float64)) >= 0.05:
            window = np.hanning(crop_height)[:, None] * np.hanning(crop_width)[None, :]
            magnitude = np.abs(np.fft.fftshift(np.fft.fft2(centered * window)))
            yy, xx = np.ogrid[:crop_height, :crop_width]
            radius = np.sqrt((yy - crop_height // 2) ** 2 + (xx - crop_width // 2) ** 2)
            rings = np.floor(radius).astype(np.int32)
            ring_count = np.bincount(rings.ravel())
            ring_mean = np.bincount(rings.ravel(), weights=magnitude.ravel()) / np.maximum(ring_count, 1)
            ratio = magnitude / np.maximum(ring_mean[rings], 1e-10)
            band = (radius >= max(6.0, min(crop_width, crop_height) * .08)) & (radius <= min(crop_width, crop_height) * .48)
            power = magnitude * magnitude
            total_power = float(np.sum(power[radius >= 2], dtype=np.float64))
            # Count local maxima, rather than every neighboring FFT bin of one peak.
            peak = band & (ratio >= 8.0)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                peak &= magnitude > np.roll(magnitude, (dy, dx), axis=(0, 1))
            count = int(np.count_nonzero(peak))
            peak_energy = float(np.sum(power[peak], dtype=np.float64)) / max(total_power, 1e-20)
            high_energy = float(np.sum(power[radius >= min(crop_width, crop_height) * .25], dtype=np.float64)) / max(total_power, 1e-20)
            maximum = float(np.max(ratio[band])) if np.any(band) else 0.0
            # Conjugate peaks appear in pairs; a minimum energy excludes
            # numerical/quantization spikes far below the visible structure.
            candidate = count >= 2 and peak_energy >= .015
            base.update(peak_count=count, peak_energy_fraction=min(1.0, peak_energy),
                        high_frequency_energy_fraction=min(1.0, high_energy),
                        maximum_peak_ratio=maximum,
                        status="pattern_candidate" if candidate else "not_detected",
                        detail=("반복 주파수 패턴이 있습니다. 워터마크 여부를 단정하는 결과는 아닙니다."
                                if candidate else "설정한 기준의 반복 주파수 패턴이 없습니다."))
    base["elapsed_seconds"] = time.perf_counter() - started
    return base


def _pixel_noise(y0, y1, x0, x1, image_width, seed):
    """Coordinate-hashed noise is identical across tile sizes and traversal."""
    import numpy as np
    ys = np.arange(y0, y1, dtype=np.uint32)[:, None]
    xs = np.arange(x0, x1, dtype=np.uint32)[None, :]
    counter = (ys * np.uint32(image_width) + xs)[..., None] * np.uint32(3)
    counter = counter + np.arange(3, dtype=np.uint32)
    value = counter ^ np.uint32(seed & 0xffffffff)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7feb352d)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846ca68b)
    value ^= value >> np.uint32(16)
    uniform = (value >> np.uint32(8)).astype(np.float32) / np.float32(1 << 24)
    return (uniform - np.float32(.5)) * np.float32(.012)


def _pixel_pad(array, padding):
    # A one-pixel dimension has no reflected neighbor; replicating it is the
    # appropriate boundary condition, and remains valid on older NumPy builds.
    import numpy as np
    result = array
    for axis, amount in enumerate(padding):
        if not amount:
            continue
        pad = [(0, 0)] * result.ndim
        pad[axis] = (amount, amount)
        result = np.pad(result, pad, mode="reflect" if result.shape[axis] > 1 else "edge")
    return result


def _pixel_filtered_tile(rgb, noise, alpha=None, max_delta=4):
    import numpy as np
    values = np.asarray(rgb, dtype=np.float32) / np.float32(255.)
    padded = _pixel_pad(values, (1, 1, 0))
    laplacian = (np.float32(4.) * values - padded[:-2, 1:-1] - padded[2:, 1:-1]
                 - padded[1:-1, :-2] - padded[1:-1, 2:])
    cleaned = values - np.float32(.22) * laplacian + noise
    horizontal = _pixel_pad(cleaned, (0, 1, 0))
    horizontal = (horizontal[:, :-2] + np.float32(2.) * horizontal[:, 1:-1]
                  + horizontal[:, 2:]) * np.float32(.25)
    vertical = _pixel_pad(horizontal, (1, 0, 0))
    smooth = (vertical[:-2] + np.float32(2.) * vertical[1:-1] + vertical[2:]) * np.float32(.25)
    mask = np.clip(np.abs(cleaned - smooth) * np.float32(8.), 0., 1.)
    candidate = np.clip(smooth * (np.float32(1.) - mask) + cleaned * mask, 0., 1.)
    delta = np.clip(candidate * np.float32(255.) - rgb, -max_delta, max_delta)
    if alpha is not None:
        opaque = alpha == 255
        safe = opaque.copy()
        pad = _pixel_pad(opaque, (2, 2))
        for dy in range(5):
            for dx in range(5):
                safe &= pad[dy:dy + opaque.shape[0], dx:dx + opaque.shape[1]]
        delta *= safe[..., None]
    return np.rint(np.clip(rgb + delta, 0., 255.)).astype(np.uint8)


def _pixel_comparison(source, result):
    import numpy as np
    squared, absolute, maximum, changed, compared, samples = 0., 0., 0, 0, 0, 0
    height = source.shape[0]
    for y in range(0, height, 256):
        original = source[y:y + 256]
        output = result[y:y + 256]
        valid = original[:, :, 3] > 0 if original.shape[2] == 4 else np.ones(original.shape[:2], dtype=bool)
        difference = output[:, :, :3].astype(np.int16) - original[:, :, :3].astype(np.int16)
        selected = difference[valid]
        if selected.size:
            square = selected.astype(np.float64)
            squared += float(np.sum(square * square, dtype=np.float64))
            absolute += float(np.sum(np.abs(selected), dtype=np.float64))
            maximum = max(maximum, int(np.max(np.abs(selected))))
            changed += int(np.count_nonzero(np.any(difference != 0, axis=2) & valid))
            compared += int(np.count_nonzero(valid))
            samples += int(selected.size)
    mse = squared / samples if samples else 0.0
    return {"scope": "rgb_nontransparent_pixels",
            "changed_pixels": changed, "compared_pixels": compared,
            "changed_pixel_ratio": changed / compared if compared else 0.0,
            "root_mean_squared_error": math.sqrt(mse), "max_absolute_error": maximum,
            "identical": not changed, "visual_quality_verified": False, "watermark_verified": False,
            "changed_fraction": changed / compared if compared else 0.0,
            "mean_absolute_error": absolute / samples if samples else 0.0,
            "mse": mse, "psnr_db": 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else None,
            "maximum_channel_change": maximum,
            "fully_transparent_pixels_excluded": source.shape[2] == 4}


def process_pixel_image(source_png, output_png, options=None, progress_callback=None, cancel_event=None):
    """Filter in bounded tiles, preserve image structure, and publish a new PNG.

    Options used internally: seed (42), tile_rows (512), tile_columns (512),
    max_delta (4; integer 1..4), minimum_psnr (40 dB; may only be increased).
    Metadata stripping/visible-template processing are managed by the caller.
    """
    import numpy as np
    from PIL import Image
    started = time.perf_counter()
    options = dict(options or {})
    _pixel_cancel(cancel_event)
    source = Path(source_png).expanduser().resolve(strict=True)
    target = Path(output_png).expanduser().absolute()
    if target.suffix.lower() != ".png" or target.is_symlink() or source == target.resolve():
        raise ValueError("원본과 다른 새 PNG 출력 경로를 지정해주세요.")
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    with source.open("rb") as stream:
        png_header = stream.read(29)
    if png_header[:8] == b"\x89PNG\r\n\x1a\n" and len(png_header) >= 25 and png_header[24] > 8:
        raise ValueError("16비트 PNG는 픽셀 정보를 보존하기 위해 이 처리에서 제외합니다.")
    seed = int(options.get("seed", 42))
    rows, columns = int(options.get("tile_rows", 512)), int(options.get("tile_columns", 512))
    max_delta = int(options.get("max_delta", 4))
    minimum_psnr = float(options.get("minimum_psnr", 40.))
    if not (1 <= rows <= 1024 and 1 <= columns <= 1024 and 1 <= max_delta <= 4
            and math.isfinite(minimum_psnr) and 40. <= minimum_psnr <= 80.):
        raise ValueError("픽셀 처리의 크기 또는 품질 설정이 올바르지 않습니다.")
    _pixel_emit(progress_callback, "이미지 처리 중", "원본 크기와 투명도를 확인합니다.")
    with Image.open(source) as raw:
        if (raw.format != "PNG" or getattr(raw, "n_frames", 1) != 1
                or raw.mode in {"I", "F", "CMYK"} or raw.mode.startswith("I;16")):
            raise ValueError("8비트 단일 프레임 PNG 이미지가 필요합니다.")
        raw.load()
        icc = raw.info.get("icc_profile")
        has_alpha = "A" in raw.getbands() or "transparency" in raw.info
        image = raw.convert("RGBA" if has_alpha else "RGB")
        original = np.asarray(image)
    height, width = original.shape[:2]
    result = np.array(original, copy=True)
    total = math.ceil(height / rows) * math.ceil(width / columns)
    frequency_before = scan_frequency_signature(source)
    index = 0
    for y in range(0, height, rows):
        for x in range(0, width, columns):
            _pixel_cancel(cancel_event)
            y2, x2 = min(height, y + rows), min(width, x + columns)
            top, bottom = max(0, y - 2), min(height, y2 + 2)
            left, right = max(0, x - 2), min(width, x2 + 2)
            block = original[top:bottom, left:right]
            noise = _pixel_noise(top, bottom, left, right, width, seed)
            filtered = _pixel_filtered_tile(block[:, :, :3], noise,
                                             block[:, :, 3] if has_alpha else None, max_delta)
            result[y:y2, x:x2, :3] = filtered[y - top:y2 - top, x - left:x2 - left]
            index += 1
            _pixel_emit(progress_callback, "이미지 처리 중", "원본 형태를 유지하며 픽셀을 처리합니다.", index, total + 2)
    _pixel_cancel(cancel_event)
    comparison = _pixel_comparison(original, result)
    mse_limit = 255.0 ** 2 / 10.0 ** (minimum_psnr / 10.0)
    quality_scale = 1.0
    if comparison["mse"] > mse_limit:
        quality_scale = math.sqrt(mse_limit / comparison["mse"])
        for y in range(0, height, 256):
            _pixel_cancel(cancel_event)
            rgb = original[y:y + 256, :, :3]
            delta = result[y:y + 256, :, :3].astype(np.int16) - rgb.astype(np.int16)
            # Rounding toward zero guarantees the measured squared error cannot
            # exceed the limit due to quantization after scaling.
            delta = np.trunc(delta.astype(np.float64) * quality_scale).astype(np.int16)
            result[y:y + 256, :, :3] = (rgb.astype(np.int16) + delta).astype(np.uint8)
        comparison = _pixel_comparison(original, result)
    _pixel_emit(progress_callback, "결과 확인 중", "저장된 픽셀과 투명도, 주파수 변화를 확인합니다.", total + 1, total + 2)
    _pixel_cancel(cancel_event)
    with tempfile.TemporaryDirectory(prefix=".unmarker-pixel-", dir=target.parent) as directory:
        staged = Path(directory) / "result.png"
        save = {"format": "PNG", "compress_level": 4}
        if icc:
            save["icc_profile"] = icc
        Image.fromarray(result).save(staged, **save)
        with Image.open(staged) as saved:
            saved.load()
            if (saved.size != image.size or saved.mode != image.mode
                    or (saved.info.get("icc_profile") or None) != (icc or None)
                    or not np.array_equal(np.asarray(saved), result)):
                raise RuntimeError("처리 결과의 픽셀 또는 투명도 저장 검증에 실패했습니다.")
        frequency_after = scan_frequency_signature(staged)
        _pixel_cancel(cancel_event)
        publish_new_file(staged, target)
    changed = comparison["changed_pixels"]
    return {"processed": True, "status": "processed" if changed else "unchanged",
            "verified": False, "removal_verified": False,
            "pixels_changed": bool(changed), "changed_pixels": changed,
            "device": "cpu", "engine": "local-laplacian-v1", "models_required": False,
            "tiles": total, "tile_size": [columns, rows], "seed": seed,
            "max_delta": max_delta, "minimum_psnr_db": minimum_psnr,
            "laplacian_strength": .22, "noise_amplitude": .012,
            "quality_scale": quality_scale, "pixel_comparison": comparison,
            "alpha_preserved": has_alpha, "alpha_boundary_protection_px": 2,
            "icc_preserved": bool(icc), "geometry_preserved": True,
            "frequency_before": frequency_before, "frequency_after": frequency_after,
            "elapsed_seconds": time.perf_counter() - started,
            "output_path": str(target),
            "detail": "픽셀 처리와 저장을 마쳤습니다." if changed else "처리와 저장을 마쳤습니다. 변경된 픽셀은 없습니다."}


# open_watermark_bridge.py
# ------------------------------------------------------------------------
from pathlib import Path
import time


_OPEN_WATERMARK_BITS = {
    "Stable Diffusion XL": (48, 0b101100111110110010010000011110111011000110011110, 44),
    # The official encoder uses bin(payload)[2:], so its two leading zeroes
    # are omitted: the literal's effective length is 46, despite its comment.
    # Also recognize explicitly zero-padded 48-bit exports used by wrappers.
    "FLUX.2": (46, 0b001010101111111010000111100111001111010100101110, 43),
    "FLUX.2 (48-bit compatibility)": (48, 0b001010101111111010000111100111001111010100101110, 44),
    "Stable Diffusion 1.x / 2.x": (136, int.from_bytes(b"StableDiffusionV1", "big"), 126),
}


def scan_open_watermarks(path):
    """Decode stored pixels before an optional, metadata-directed orientation.

    DWT block order belongs to the embedded raster. Applying EXIF orientation
    first can erase a valid payload even though no stored RGB value changed.
    The fallback is limited to the image's actual EXIF transform, and only runs
    after the stored raster misses. There is no search over arbitrary rotations,
    resizes, alpha backgrounds or thresholds.

    A partial alpha channel does not prevent reading its uncomposited RGB data.
    Such results explicitly record that basis; they are not a claim about a
    displayed composite. Fully transparent images remain unsupported.
    """
    started = time.perf_counter()
    result = {"status": "unavailable", "available": False,
              "scope": "open_dwt_dct", "engine": "remove-ai-watermarks==0.46.0/dwt_dct",
              "supported_schemes": list(_OPEN_WATERMARK_BITS), "matches": [],
              "comparisons": [], "synthid_supported": False,
              "basis": None, "attempts": [], "exif_orientation": 1,
              "alpha_handling": "opaque_rgb",
              "detail": "공개 워터마크 검사 구성요소를 준비해야 합니다."}
    try:
        import importlib.metadata
        import numpy as np
        from PIL import Image, ImageOps
        if importlib.metadata.version("remove-ai-watermarks") != "0.46.0":
            raise ImportError("remove-ai-watermarks==0.46.0이 필요합니다.")
        from remove_ai_watermarks.dwt_dct import decode_dwt_dct_lengths

        def decode_pixels(rgb, basis):
            attempt_started = time.perf_counter()
            attempt = {"basis": basis, "status": "error", "comparisons": [], "matches": []}
            result["attempts"].append(attempt)
            try:
                bgr = np.asarray(rgb)[:, :, ::-1].copy()
                decoded = decode_dwt_dct_lengths(bgr, (46, 48, 136))
                for name, (width, payload, threshold) in _OPEN_WATERMARK_BITS.items():
                    bits = np.asarray(decoded[width]).reshape(-1)
                    if bits.size != width:
                        raise ValueError("공개 워터마크 디코더가 잘못된 길이를 반환했습니다.")
                    value = 0
                    for bit in bits:
                        value = (value << 1) | int(bool(bit))
                    matching = width - (value ^ payload).bit_count()
                    comparison = {"name": name, "matched_bits": matching, "total_bits": width,
                                  "threshold_bits": threshold, "matched": matching >= threshold}
                    attempt["comparisons"].append(comparison)
                    if comparison["matched"]:
                        attempt["matches"].append(dict(comparison))
                attempt["status"] = "detected" if attempt["matches"] else "not_detected"
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                attempt["elapsed_seconds"] = round(time.perf_counter() - attempt_started, 5)
            return attempt

        result["available"] = True
        source = Path(path).expanduser().resolve(strict=True)
        with source.open("rb") as stream:
            header = stream.read(29)
        if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 25 and header[24] > 8:
            result.update(status="unsupported", detail="16비트 PNG는 공개 워터마크 검사 대상이 아닙니다.")
            return result
        with Image.open(source) as raw:
            if (getattr(raw, "n_frames", 1) != 1 or raw.mode in {"I", "F", "CMYK"}
                    or raw.mode.startswith("I;16")):
                result.update(status="unsupported", detail="이 형식은 공개 워터마크 검사 대상이 아닙니다.")
                return result
            raw.load()
            if min(raw.size) < 8 or raw.width * raw.height < 256 * 256:
                result.update(status="insufficient_data", detail="공개 워터마크를 판별하기에는 이미지가 작습니다.")
                return result
            uncomposited = False
            if "A" in raw.getbands() or "transparency" in raw.info:
                alpha_min, alpha_max = raw.convert("RGBA").getchannel("A").getextrema()
                if alpha_max == 0:
                    result.update(status="unsupported", detail="투명 이미지의 공개 워터마크는 판별하지 않습니다.")
                    result["alpha_handling"] = "fully_transparent"
                    return result
                uncomposited = alpha_min < 255
                if uncomposited:
                    result["alpha_handling"] = "uncomposited_rgb"
            orientation = raw.getexif().get(0x0112, 1)
            orientation = orientation if orientation in (2, 3, 4, 5, 6, 7, 8) else 1
            result["exif_orientation"] = orientation
            stored_rgb = raw.convert("RGB")
            suffix = "_uncomposited" if uncomposited else ""
            selected = decode_pixels(stored_rgb, "stored_rgb" + suffix)
            if not selected["matches"] and orientation != 1:
                oriented_rgb = ImageOps.exif_transpose(raw).convert("RGB")
                if (stored_rgb.size != oriented_rgb.size or
                        not np.array_equal(np.asarray(stored_rgb), np.asarray(oriented_rgb))):
                    oriented = decode_pixels(oriented_rgb, "exif_oriented_rgb" + suffix)
                    if oriented["matches"]:
                        selected = oriented
            # Preserve the existing one-comparison-per-scheme result contract.
            # All fallback scores remain available in attempts. A negative
            # result keeps the primary raster's scores rather than selecting
            # whichever non-matching score happened to be higher.
            result.update(basis=selected["basis"], comparisons=selected["comparisons"],
                          matches=selected["matches"])
        result.update(status="detected" if result["matches"] else "not_detected",
                      detail=("공개 DWT-DCT 워터마크의 비트 패턴과 일치합니다." if result["matches"]
                              else "검사한 공개 DWT-DCT 워터마크와 일치하지 않습니다."))
    except (ImportError, ModuleNotFoundError) as exc:
        result.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        # A failed decoder must not masquerade as an image with no watermark.
        result.update(status="error", detail="공개 워터마크 검사를 마치지 못했습니다.",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 5)
    return result


def _app_detection_capabilities(metadata, open_result):
    capabilities = metadata.setdefault("capabilities", {})
    capabilities.update(invisible_watermark_detection=bool(open_result.get("available")),
                        invisible_watermark_detection_scope="open_dwt_dct",
                        supported_invisible_watermarks=list(_OPEN_WATERMARK_BITS),
                        synthid_detection=False, frequency_measurement=True)
    return metadata


# worker_runtime.py
# ------------------------------------------------------------------------
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class _JobCancellation:
    def __init__(self, path):
        self.path = Path(path)

    def is_set(self):
        return self.path.exists()


def _app_process_worker(request_path):
    """Internal local IPC entry; the desktop does not send images to a service."""
    request_path = Path(request_path).resolve(strict=True)
    request = _rs_read_json(request_path)
    if not isinstance(request, dict) or request.get("schema") != 1:
        raise ValueError("이미지 처리 요청을 읽지 못했습니다.")
    directory = request_path.parent
    report_path = directory / "result.json"
    cancel = _JobCancellation(directory / "cancel")
    try:
        if hasattr(os, "nice"):
            try:
                os.nice(5)
            except OSError:
                pass
        with (directory / "progress.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
            def progress(title, detail=""):
                payload = title if isinstance(title, dict) else {"title": str(title), "detail": str(detail)}
                stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            result = app_process(request["source"], request["output"], request.get("options"), progress, cancel)
        _rs_atomic_json(report_path, {"ok": True, "result": result})
        return 0
    except (ProcessingCancelled, ModelCancelled, PixelProcessingCancelled) as exc:
        _rs_atomic_json(report_path, {"ok": False, "cancelled": True, "error": str(exc)})
        return 130
    except Exception as exc:
        error = app_error_payload(exc, "로컬 이미지 처리")
        _rs_atomic_json(report_path, {"ok": False, "error": str(exc), "details": error})
        return 1


def app_process_isolated(source, output, options=None, progress_callback=None, cancel_event=None):
    """Let one worker own the models while the UI background thread waits.

    The worker exits after every job, including failure and cancellation, so
    Torch/MPS allocations cannot remain attached to the next GUI operation.
    """
    _app_cancel(cancel_event)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("같은 이름의 파일이 있습니다. 다른 이름을 선택해 주세요.")
    with tempfile.TemporaryDirectory(prefix="unmarker-job-") as temporary:
        directory = Path(temporary)
        request = directory / "request.json"
        _rs_atomic_json(request, {"schema": 1, "source": str(Path(source).resolve(strict=True)),
                                  "output": str(output), "options": dict(options or {})})
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).resolve()))
        command.extend(("--process-job", str(request)))
        scratch = directory / "tmp"
        scratch.mkdir()
        environment = os.environ.copy()
        environment.update({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                            "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch)})
        position, pending, cancel_started, process = 0, "", None, None
        try:
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        if cancel_started is None:
                            (directory / "cancel").touch()
                            cancel_started = time.monotonic()
                        if process.poll() is None and time.monotonic() - cancel_started > 3.0:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                    progress_file = directory / "progress.jsonl"
                    if progress_file.exists():
                        with progress_file.open("r", encoding="utf-8") as reader:
                            reader.seek(position)
                            pending += reader.read()
                            position = reader.tell()
                        lines = pending.split("\n")
                        pending = lines.pop()
                        for line in lines:
                            if line and progress_callback is not None:
                                progress_callback(json.loads(line))
                    if process.poll() is not None:
                        break
                    time.sleep(0.08)
                process.wait()
            if cancel_started is not None or (cancel_event is not None and cancel_event.is_set()):
                raise ProcessingCancelled("작업을 취소했습니다.")
            report = directory / "result.json"
            if report.is_file():
                payload = _rs_read_json(report)
                if payload.get("ok") is True and process.returncode == 0:
                    result = payload["result"]
                    result["execution"] = {"isolation": "process", "worker_exit_code": 0,
                                           "model_memory_released": True}
                    return result
                if payload.get("cancelled"):
                    raise ProcessingCancelled(payload.get("error") or "작업을 취소했습니다.")
                detail = payload.get("error") or "이미지 처리에 실패했습니다."
                original_log = (payload.get("details") or {}).get("log_path")
                if original_log:
                    detail += "\n오류 기록: " + str(original_log)
                raise RuntimeError(detail)
            log = directory / "worker.log"
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                detail = _rs_safe_text(stream.read().decode("utf-8", errors="replace")).strip()
            raise RuntimeError(f"이미지 처리 작업이 종료되었습니다 (종료 코드 {process.returncode})."
                               + ("\n" + detail if detail else ""))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


# provenance.py
# ------------------------------------------------------------------------
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import secrets
import ssl
import stat


PROVENANCE_SERVICES = {
    "openai": {"name": "OpenAI Verify", "url": "https://openai.com/ko-KR/research/verify/",
               "scope": "OpenAI 생성 콘텐츠", "api_available": True},
    "google": {"name": "Google SynthID Detector", "url": "https://synthid.com/",
               "scope": "Google 및 지원 협력사의 콘텐츠", "api_available": False},
}
_PROVENANCE_HOST = "api.openai.com"
_PROVENANCE_PATH = "/v1/content_provenance_checks"
_PROVENANCE_MAX_FILE = 50 * 1024 * 1024
_PROVENANCE_MAX_RESPONSE = 1024 * 1024


class ProvenanceError(RuntimeError):
    """키·요청 본문·서버의 원시 오류를 포함하지 않는 사용자용 오류입니다."""


def _provenance_signals(payload):
    """누락된 검사나 새 응답 형식을 미검출로 해석하지 않습니다."""
    if not isinstance(payload, dict) or payload.get("object") != "content_provenance_check":
        raise ProvenanceError("공식 검사 응답 형식을 확인하지 못했습니다.")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ProvenanceError("공식 검사 응답에 결과 목록이 없습니다.")
    signals = {}
    for name in ("synthid", "c2pa"):
        matches = [row for row in rows if isinstance(row, dict) and row.get("type") == name]
        if len(matches) != 1 or matches[0].get("outcome") not in ("detected", "not_detected"):
            signals[name] = {"status": "not_tested", "detected": None}
            continue
        row = matches[0]
        signals[name] = {"status": row["outcome"], "detected": row["outcome"] == "detected"}
        for key in ("validation_state", "issuer", "model", "generated_at"):
            value = row.get(key)
            if value is None or isinstance(value, str):
                signals[name][key] = value[:512] if isinstance(value, str) else None
    return signals


def verify_openai_file(path, *, allow_upload=False):
    """선택한 파일 한 개만 공식 API에 전송합니다. 키는 환경변수에서만 읽습니다.

    파일의 실제 형식·크기와 업로드한 바이트의 해시를 기록합니다. 고정 HTTPS
    목적지, 인증서 검증, 제한 시간, 응답 크기 제한을 유지하며 리디렉션을 따르지
    않습니다. 오류가 나더라도 자동 재시도 또는 다른 서비스 전송은 하지 않습니다.
    """
    if allow_upload is not True:
        raise ProvenanceError("외부 검사에는 선택한 이미지의 OpenAI 업로드 동의가 필요합니다.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ProvenanceError("OPENAI_API_KEY가 설정되지 않았습니다. 공식 웹 검사도 이용하실 수 있습니다.")
    if any(ord(char) < 33 or ord(char) > 126 for char in api_key):
        raise ProvenanceError("API 키 형식이 올바르지 않습니다.")
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or not 0 < source.stat().st_size <= _PROVENANCE_MAX_FILE:
        raise ProvenanceError("50 MiB 이하의 PNG, JPEG, WebP 파일을 선택해 주세요.")
    with source.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ProvenanceError("일반 이미지 파일을 선택해 주세요.")
        data = stream.read(_PROVENANCE_MAX_FILE + 1)
    if len(data) > _PROVENANCE_MAX_FILE:
        raise ProvenanceError("파일 크기가 50 MiB를 초과했습니다.")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.verify()
    except Exception:
        raise ProvenanceError("이미지 파일의 내용을 읽지 못했습니다.") from None
    media = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"), "WEBP": ("webp", "image/webp")}
    if image_format not in media:
        raise ProvenanceError("공식 이미지 검사는 PNG, JPEG, WebP만 지원합니다.")
    extension, content_type = media[image_format]
    boundary = "unmark-" + secrets.token_hex(24)
    # 로컬 파일명이나 전체 경로는 외부로 보내지 않습니다.
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
              f"filename=\"image.{extension}\"\r\nContent-Type: {content_type}\r\n\r\n").encode("ascii")
    body = prefix + data + f"\r\n--{boundary}--\r\n".encode("ascii")
    try:
        import truststore
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        connection = http.client.HTTPSConnection(_PROVENANCE_HOST, timeout=90, context=context)
        try:
            connection.request("POST", _PROVENANCE_PATH, body=body, headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "multipart/form-data; boundary=" + boundary,
                "Accept": "application/json", "User-Agent": "AI-Unmarker-Pro"})
            response = connection.getresponse()
            status = response.status
            raw = response.read(_PROVENANCE_MAX_RESPONSE + 1)
        finally:
            connection.close()
    except Exception:
        raise ProvenanceError("OpenAI 검사 서버에 안전하게 연결하지 못했습니다. 네트워크와 인증서를 확인해 주세요.") from None
    if status != 200:
        messages = {400: "파일이 지원되지 않거나 검사할 수 없습니다.",
                    401: "API 키를 확인해 주세요.", 403: "이 API를 사용할 권한이 없습니다.",
                    404: "이 계정의 Content Provenance API 접근 권한을 확인해 주세요.",
                    429: "검사 요청 한도에 도달했습니다. 나중에 다시 시도해 주세요."}
        raise ProvenanceError(f"OpenAI 검사 오류 (HTTP {status}): " + messages.get(status, "검사를 완료하지 못했습니다."))
    if len(raw) > _PROVENANCE_MAX_RESPONSE:
        raise ProvenanceError("검사 서버의 응답 크기가 허용 범위를 초과했습니다.")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProvenanceError("검사 서버의 응답을 읽지 못했습니다.") from None
    return {"provider": "openai", "method": "official_api", "scope": "openai_content",
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "signals": _provenance_signals(payload),
            "detail": "미검출은 AI로 생성되지 않았다는 뜻이 아닙니다. Google 검사는 별도로 수행하십시오."}


# app_backend.py
# ------------------------------------------------------------------------
import argparse
import hashlib
import json
import re
import shutil
import sys
import traceback
import time



class ProcessingCancelled(Exception):
    pass


def app_identity():
    """Identify the actual running release, including its original build source."""
    receipt = getattr(sys, "_ai_unmarker_build", {})
    frozen = bool(getattr(sys, "frozen", False))
    digest = receipt.get("source_sha256", "") if frozen else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"app_version": APP_VERSION, "engine_id": ENGINE_ID,
            "release_id": RELEASE_ID, "source_sha256": digest,
            "frozen": frozen, "executable": sys.executable}


def _app_numeric_limits():
    try:
        import cv2
        cv2.setNumThreads(2)
    except ImportError:
        pass


def app_export_result(processed_path, output_path):
    """Export the checked PNG bytes without re-encoding or replacing any file."""
    source = Path(processed_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().absolute()
    if output.suffix.lower() != ".png":
        raise ValueError("PNG 파일로 저장해 주세요.")
    if output.is_symlink() or output.exists() or source == output.resolve():
        raise FileExistsError("같은 이름의 파일이 있습니다. 다른 이름으로 저장해 주세요.")
    if not output.parent.is_dir():
        raise FileNotFoundError("저장할 폴더를 찾을 수 없습니다.")
    original_stat = source.stat()
    _require_pillow()
    with Image.open(source) as img:
        if img.format != "PNG":
            raise ValueError("처리 결과가 PNG 파일이 아닙니다.")
        img.verify()
    with tempfile.TemporaryDirectory(prefix=".unmark-export-", dir=output.parent) as directory:
        stage = Path(directory) / "result.png"
        digest = hashlib.sha256()
        with source.open("rb") as src, stage.open("xb") as dst:
            for block in iter(lambda: src.read(2 * 1024 * 1024), b""):
                dst.write(block)
                digest.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        current_stat = source.stat()
        if (current_stat.st_dev, current_stat.st_ino, current_stat.st_size,
                current_stat.st_mtime_ns, current_stat.st_ctime_ns) != (
                original_stat.st_dev, original_stat.st_ino, original_stat.st_size,
                original_stat.st_mtime_ns, original_stat.st_ctime_ns):
            raise RuntimeError("저장하는 동안 처리 결과가 변경되었습니다. 다시 시도해 주세요.")
        copied = hashlib.sha256()
        with stage.open("rb") as src:
            for block in iter(lambda: src.read(2 * 1024 * 1024), b""):
                copied.update(block)
        if digest.digest() != copied.digest() or stage.stat().st_size != original_stat.st_size:
            raise RuntimeError("저장 파일을 검증하지 못했습니다. 다시 시도해 주세요.")
        publish_new_file(stage, output)
    return {"output_path": str(output), "bytes": original_stat.st_size, "sha256": digest.hexdigest()}


def _app_synthid_observation(metadata):
    """Report an actual provenance declaration separately from pixel detection.

    The local SDK can read signed manifest assertions without GPU models.
    Finding such a declaration never substitutes for a SynthID pixel decoder,
    and a missing declaration never means that the pixels lack a watermark.
    """
    c2pa = (metadata.get("metadata") or {}).get("c2pa") or {}
    observed = [dict(item) for item in c2pa.get("watermark_declarations", [])
                if isinstance(item, dict) and
                str(item.get("name", "")).casefold() == "synthid"]
    declarations = [item for item in observed if item.get("scope") == "active_manifest"]
    related = [item for item in observed if item.get("scope") != "active_manifest"]
    return {"status": "metadata_declared" if declarations else "not_tested",
            "detected": None, "pixel_detection_supported": False,
            "verification_services": {name: dict(service) for name, service in PROVENANCE_SERVICES.items()},
            "verification_method": "official_service_user_upload",
            "scope": "c2pa_declaration", "metadata_declared": bool(declarations),
            "declarations": declarations, "related_declarations": related,
            "c2pa_status": c2pa.get("status", "unknown"),
            "detail": "C2PA에 기록된 SynthID 삽입 선언입니다." if declarations else
                      "주파수 패턴 수치는 SynthID 판정값이 아닙니다."}


def _app_metadata_observation(metadata):
    """One CPU metadata result for import, processing and the desktop summary.

    An ICC colour profile alone is not an AI/source trace. Software names are
    literal records, so preferring a recorded generator over a later editor
    does not establish that the complete image was generated by that tool.
    """
    fields = metadata.get("metadata") or {}
    signal_keys = ("c2pa", "exif", "xmp", "text", "jumbf", "comment", "other")
    categories = [key for key in signal_keys if (fields.get(key) or {}).get("present") is True]
    credentials = fields.get("c2pa") or {}
    apps = [dict(item) for item in metadata.get("recorded_apps", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()]
    named_ai_software = re.compile(
        r"\b(?:openai|chatgpt|dall[-· ]?e|midjourney|firefly|gemini|imagen|"
        r"stable[ _-]?diffusion|comfyui|automatic1111|grok|bing image creator)\b", re.IGNORECASE)

    def source_order(item):
        group = 0 if item.get("role") == "ai_generator" else (
            1 if named_ai_software.search(item["name"]) else 2)
        scope = {"active_manifest": 0, "ingredient": 1, "file_metadata": 2}.get(item.get("scope"), 3)
        return group, item.get("signature_verified") is not True, scope

    apps.sort(key=source_order)
    declarations = [dict(item) for item in credentials.get("ai_assertions", []) if isinstance(item, dict)]
    incomplete = []
    if credentials.get("status") in ("invalid", "unavailable", "error", "external_reference"):
        incomplete.append({"category": "c2pa", "status": credentials["status"]})
    xmp_errors = (fields.get("xmp") or {}).get("container_errors") or []
    if xmp_errors:
        incomplete.append({"category": "xmp", "status": "incomplete", "errors": xmp_errors})
    return {"status": "incomplete" if incomplete else ("detected" if categories else "not_detected"),
            "detected": True if categories else (None if incomplete else False),
            "categories": categories, "count": len(categories),
            "scope": "file_metadata", "pixel_detection": False,
            "color_profile_present": (fields.get("icc") or {}).get("present") is True,
            "recorded_apps": apps, "primary_app": apps[0] if apps else None,
            "ai_declared": bool(declarations), "ai_declarations": declarations,
            "complete": not incomplete, "incomplete_checks": incomplete}


def app_scan_file(path):
    started = time.perf_counter()
    _app_numeric_limits()
    metadata = scan_image(path)
    # Neither a filename nor a text mention is evidence of a pixel watermark.
    if metadata["animated"] or metadata["bit_depth"] > 8 or metadata["mode"] in ("CMYK", "I", "F"):
        visible = {"status": "unavailable", "marks": [],
                   "detail": "이 이미지 형식은 가시 워터마크 검사 대상이 아닙니다."}
        frequency = {"status": "unavailable", "peak_count": None,
                     "scope": "spectral_anomaly_heuristic", "verified_watermark": False,
                     "detail": "이 이미지 형식은 픽셀 검사 대상이 아닙니다."}
    else:
        # A measured image statistic is not a probability of AI authorship.
        frequency = scan_frequency_signature(path)
        visible = app_scan_visible(path)
    open_watermarks = scan_open_watermarks(path)
    _app_detection_capabilities(metadata, open_watermarks)
    return {"app": app_identity(), "metadata": metadata,
            "metadata_observation": _app_metadata_observation(metadata), "visible": visible, "frequency": frequency,
            "open_watermarks": open_watermarks,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "synthid": _app_synthid_observation(metadata)}


def _app_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("작업을 취소했습니다.")


def app_process(input_path, output_path, options=None, progress_callback=None, cancel_event=None):
    """Run local Qwen processing and rescan the actual encoded output."""
    started = time.perf_counter()
    _app_numeric_limits()
    options = dict(options or {})
    regenerate = bool(options.get("regenerate", not options.get("metadata_only", False)))
    pixel_process = bool(options.get("pixel_process", False))
    remove_visible = bool(options.get("remove_visible", True))
    progress = progress_callback or (lambda status, detail="": None)
    source = Path(input_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().absolute()
    if output.suffix.lower() != ".png":
        raise ValueError("결과 파일은 .png로 저장해주세요.")
    if source == output.resolve() or (output.exists() and os.path.samefile(source, output)):
        raise ValueError("원본 파일과 다른 경로를 선택해주세요.")
    if output.exists():
        raise FileExistsError("같은 이름의 파일이 있습니다. 새 이름으로 저장해주세요.")
    if output.is_symlink() or not output.parent.is_dir():
        raise ValueError("저장 경로를 확인해주세요.")
    _app_cancel(cancel_event)
    progress("파일 확인 중", source.name)
    original_stat = source.stat()
    visible_result = {"status": "not_requested", "marks": [], "detail": "처리하지 않음"}
    regeneration = {"processed": False, "verified": False, "detail": "처리하지 않음"}
    pixel_result = {"processed": False, "verified": False, "detail": "처리하지 않음"}
    preserve_profile = bool(options.get("preserve_color_profile", True))
    with tempfile.TemporaryDirectory(prefix="unmarker-") as directory:
        workdir = Path(directory)
        snapshot = workdir / ("source" + source.suffix)
        shutil.copyfile(source, snapshot)
        source_stat = source.stat()
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) != (
                original_stat.st_dev, original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns):
            raise RuntimeError("읽는 도중 원본 파일이 변경되었습니다. 다시 선택해주세요.")
        prepared = workdir / "prepared.png"
        progress("메타데이터 삭제 중", "EXIF·XMP·텍스트·C2PA 정보를 삭제하고 결과를 확인합니다.")
        initial = clean_image(snapshot, prepared, preserve_profile)
        initial["before"]["path"] = str(source)
        input_frequency = scan_frequency_signature(prepared)
        input_visible = app_scan_visible(str(prepared)) if remove_visible else {
            "status": "not_requested", "marks": [], "detail": "검사하지 않음"}
        # The before-result must inspect the original stored raster. Preparing
        # a display-oriented PNG can change a DWT payload's block order.
        input_open = scan_open_watermarks(snapshot)
        _app_detection_capabilities(initial["before"], input_open)
        input_scan = {"metadata": initial["before"], "visible": input_visible,
                      "metadata_observation": _app_metadata_observation(initial["before"]),
                      "frequency": input_frequency, "open_watermarks": input_open,
                      "synthid": _app_synthid_observation(initial["before"])}
        current = prepared
        _app_cancel(cancel_event)
        if remove_visible and input_visible.get("status") == "detected":
            progress("가시 워터마크 처리 중", "지원하는 표식을 검사한 뒤 해당 영역만 처리합니다.")
            visible_path = workdir / "visible.png"
            visible_result = app_remove_visible(str(current), str(visible_path))
            if visible_result.get("status") in ("error", "unavailable"):
                raise RuntimeError(visible_result.get("detail", "가시 워터마크 처리에 실패했습니다."))
            if not visible_path.is_file():
                raise RuntimeError("가시 워터마크 처리 결과 파일이 생성되지 않았습니다.")
            # Restore the prepared alpha/profile explicitly, checking geometry.
            with Image.open(prepared) as base, Image.open(visible_path) as transformed:
                base.load()
                transformed.load()
                if transformed.size != base.size:
                    raise RuntimeError("가시 워터마크 처리 과정에서 이미지 크기가 바뀌었습니다.")
                adjusted = transformed.convert("RGB")
                if "A" in base.getbands():
                    adjusted.putalpha(base.getchannel("A"))
                adjusted.info.clear()
                save_options = {"format": "PNG"}
                if preserve_profile and base.info.get("icc_profile"):
                    save_options["icc_profile"] = base.info["icc_profile"]
                adjusted.save(visible_path, **save_options)
            current = visible_path
        elif remove_visible:
            # A negative visible-template scan does not suppress pixel work.
            # No-op file copies and a second identical detection pass are not
            # necessary when no registered mark was selected for restoration.
            visible_result = dict(input_visible)
            visible_result.update(applied=False, verified=False)
            if input_visible.get("status") == "not_detected":
                visible_result.update(status="no_watermark",
                                      detail="자동 복원 대상으로 선택된 가시 표식이 없습니다.")
        _app_cancel(cancel_event)
        if pixel_process:
            processed = workdir / "pixels.png"
            try:
                pixel_result = process_pixel_image(current, processed, options, progress, cancel_event)
            except PixelProcessingCancelled as exc:
                raise ProcessingCancelled(str(exc)) from exc
            current = processed
        _app_cancel(cancel_event)
        if regenerate:
            regenerated = workdir / "regenerated.png"
            try:
                regeneration = regenerate_image(current, regenerated, options, progress, cancel_event)
            except ModelCancelled as exc:
                raise ProcessingCancelled(str(exc)) from exc
            current = regenerated
        _app_cancel(cancel_event)
        progress("결과 검증 중", "실제 출력 파일의 메타데이터와 이미지 데이터를 다시 확인합니다.")
        # Stage and verify first. Publication never replaces a file created by
        # another process, including on filesystems without hard-link support.
        with tempfile.TemporaryDirectory(prefix=".unmarker-", dir=output.parent) as stage_dir:
            staged_output = Path(stage_dir) / "verified.png"
            final = clean_image(current, staged_output, preserve_profile)
            with Image.open(prepared) as before_img, Image.open(staged_output) as after_img:
                before_rgba, after_rgba = before_img.convert("RGBA"), after_img.convert("RGBA")
                if before_rgba.size != after_rgba.size:
                    raise RuntimeError("결과 이미지 크기가 원본과 달라 저장하지 않았습니다.")
                if not _same_pixels(before_rgba.getchannel("A"), after_rgba.getchannel("A")):
                    raise RuntimeError("결과 이미지 투명도가 원본과 달라 저장하지 않았습니다.")
                if (before_img.info.get("icc_profile") or None) != (after_img.info.get("icc_profile") or None):
                    raise RuntimeError("결과의 색상 프로필 검증에 실패하여 저장하지 않았습니다.")
                original_pixels_unchanged = _same_pixels(before_rgba, after_rgba)
            # Recheck the exact encoded output before publishing it. A failed
            # check remains unvalidated; it must never inherit verified=True
            # from the earlier in-memory, pre-encoding removal report.
            frequency_after = scan_frequency_signature(staged_output)
            open_after = scan_open_watermarks(staged_output)
            _app_detection_capabilities(final["after"], open_after)
            check = app_scan_visible(str(staged_output)) if remove_visible else {
                "status": "not_requested", "marks": [], "detail": "검사하지 않음"}
            if remove_visible:
                visible_result["output_check"] = check
                if check.get("status") == "detected":
                    visible_result.update(
                        status="partial", verified=False,
                        detail="일부 가시 워터마크가 결과에서 다시 검출되었습니다.")
                elif check.get("status") != "not_detected":
                    visible_result.update(
                        status="unvalidated", verified=False,
                        detail="가시 워터마크 결과를 재검사하지 못했습니다.")
            _app_cancel(cancel_event)
            publish_new_file(staged_output, output)
        final["before"] = initial["before"]
        final["after"]["path"] = str(output)
        final["output_path"] = str(output)
        final["removed_metadata"] = list(dict.fromkeys(
            initial["removed_metadata"] + final["removed_metadata"]))
        final["original_pixels_unchanged"] = original_pixels_unchanged
        final["image_regenerated"] = bool(regeneration["processed"])
        if regeneration["processed"]:
            # Report the published file, not the intermediate PNG removed
            # when this function's temporary workspace is cleaned up.
            regeneration["output_path"] = str(output)
        if pixel_result["processed"]:
            pixel_result["output_path"] = str(output)
            pixel_result["frequency_after"] = frequency_after
        # Completion of the requested pixel operation and confirmation that a
        # proprietary watermark was removed are different facts. Never replace
        # either one with an unconditional success/failure flag.
        final["pixel_processing_completed"] = bool(pixel_result["processed"] or regeneration["processed"])
        final["invisible_watermark_processed"] = final["pixel_processing_completed"]
        final["invisible_watermark_removal_verified"] = False
        if remove_visible:
            visible_result["output"] = str(output)
        output_scan = {"metadata": final["after"], "visible": check, "frequency": frequency_after,
                       "metadata_observation": _app_metadata_observation(final["after"]),
                       "open_watermarks": open_after,
                       "synthid": _app_synthid_observation(final["after"])}
        # This flag describes the selected local checks, never SynthID removal.
        verified = not (regeneration["processed"] or pixel_result["processed"]) and visible_result["status"] in ("not_requested", "no_watermark", "cleaned", "not_detected")
        # verified retains its legacy conservative meaning. The file's checked
        # completion and the actual before/after measurements have their own
        # fields, so the UI can show a saved result without inventing a detector.
        legacy_cleanup = {"status": "not_requested", "removed_bytes": 0, "removed_files": []}
        if regeneration.get("processed") is True and regeneration.get("weights_loaded") is True:
            # A completed new-engine output is the migration prerequisite.
            # Cache maintenance must never invalidate the published image.
            try:
                legacy_cleanup = prune_legacy_model_cache(cancel_event)
            except Exception as exc:
                legacy_cleanup = {"status": "deferred", "removed_bytes": 0,
                                  "removed_files": [], "error": str(exc)}
        return {"app": app_identity(), "output_path": str(output), "metadata": final, "visible": visible_result,
                "pixel_processing": pixel_result, "regeneration": regeneration,
                "legacy_model_cache_cleanup": legacy_cleanup,
                "verified": verified, "input_scan": input_scan, "output_scan": output_scan,
                "processing_completed": True, "elapsed_seconds": round(time.perf_counter() - started, 4),
                "validation": {"output_decoded": True, "metadata_checked": True,
                               "frequency_rechecked": True,
                               "open_watermarks_rechecked": open_after.get("status") in ("detected", "not_detected"),
                               "alpha_preserved": True, "dimensions_preserved": True,
                               "original_pixels_unchanged": original_pixels_unchanged,
                               "synthid_verified": False},
                "synthid": {"status": "not_tested", "detail": "주파수 패턴 수치는 SynthID 판정값이 아닙니다."}}


def app_engine_probe():
    """Test executable identity and local model APIs without downloading weights."""
    identity = app_identity()
    try:
        _app_numeric_limits()
        model_check = model_engine_preflight()
        return dict(identity, ready=model_check.get("ready") is True,
                    model_preflight=model_check)
    except Exception as exc:
        return dict(identity, ready=False, model_preflight={"ready": False},
                    error=f"{type(exc).__name__}: {exc}")


def app_cli_main():
    parser = argparse.ArgumentParser(description="AI Unmarker Pro — 로컬 워터마크 처리 및 메타데이터 삭제")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scan", metavar="IMAGE", help="메타데이터·주파수 패턴·지원 워터마크 검사")
    group.add_argument("--verify-openai", metavar="IMAGE", help="공식 OpenAI API로 C2PA와 SynthID 검사 (키와 업로드 동의 필요)")
    group.add_argument("--clean", metavar="IMAGE", help="Qwen·Lightning으로 처리하고 원본 디테일을 복원한 뒤 PNG 저장")
    group.add_argument("--prepare-models", action="store_true", help="이미지 처리에 필요한 파일을 외부 캐시에 미리 다운로드")
    group.add_argument("--engine-probe", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--process-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help="검사·처리 정보를 JSON으로 저장")
    parser.add_argument("--ignore-overlay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", help="새 출력 .png 경로")
    parser.add_argument("--remove-visible", action="store_true")
    parser.add_argument("--regenerate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-only", action="store_true", help="픽셀을 유지하고 메타데이터만 삭제")
    parser.add_argument("--pixel-only", action="store_true", help="모델 없이 미세 픽셀 처리만 실행")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-working-side", type=int, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--restore-sigma", type=float, default=DETAIL_RESTORE_POLICY["default_sigma"])
    parser.add_argument("--restore-mode", choices=DETAIL_RESTORE_POLICY["modes"],
                        default=DETAIL_RESTORE_POLICY["default_mode"], help="gaussian: 기본 복원, edge: 윤곽 보존")
    parser.add_argument("--unsharp", type=float, default=DETAIL_RESTORE_POLICY["default_unsharp"],
                        help="복원 후 선명화 강도 0~1. 약한 선명화 권장값 0.2")
    parser.add_argument("--seed", type=int, default=None, help="재현용 시드. 생략하면 무작위 시드를 결과에 기록합니다.")
    parser.add_argument("--no-restore", action="store_true", help="원본 고주파 디테일을 다시 합치지 않습니다.")
    parser.add_argument("--remove-icc", action="store_true")
    parser.add_argument("--allow-upload", action="store_true", help="--verify-openai의 선택 이미지 OpenAI 전송에 동의")
    args = parser.parse_args()
    if args.process_job:
        raise SystemExit(_app_process_worker(args.process_job))
    if args.engine_probe:
        result = app_engine_probe()
    elif args.prepare_models:
        def setup_progress(payload, detail=""):
            if isinstance(payload, dict):
                print(payload.get("title", "준비 중"), payload.get("detail", ""), file=sys.stderr)
            else:
                print(payload, detail, file=sys.stderr)
        dependencies = ensure_model_dependencies(setup_progress)
        if not dependencies.get("ready") and not dependencies.get("restart_required"):
            message = dependencies.get("error") or "실행 구성요소를 준비하지 못했습니다."
            if dependencies.get("log_path"):
                message += "\n설치 기록: " + dependencies["log_path"]
            raise RuntimeError(message)
        paths = prepare_models(setup_progress)
        result = {"app": app_identity(), "models_ready": True, "paths": paths}
    elif args.verify_openai:
        result = verify_openai_file(args.verify_openai, allow_upload=args.allow_upload)
    elif args.scan:
        result = app_scan_file(args.scan)
    elif args.clean:
        if not args.output:
            parser.error("--clean에는 --output 경로가 필요합니다.")
        if args.metadata_only and (args.regenerate or args.pixel_only):
            parser.error("--metadata-only는 다른 이미지 처리 옵션과 함께 사용할 수 없습니다.")
        result = app_process(args.clean, args.output, {
            "preserve_color_profile": not args.remove_icc,
            "remove_visible": args.remove_visible or not args.metadata_only,
            "pixel_process": args.pixel_only,
            "regenerate": not (args.metadata_only or args.pixel_only),
            "device": args.device, "strength": args.strength, "steps": args.steps,
            **({"max_working_side": args.max_working_side} if args.max_working_side is not None else {}),
            "lora_scale": args.lora_scale, "restore_sigma": args.restore_sigma, "restore": not args.no_restore,
            "restore_mode": args.restore_mode, "unsharp": args.unsharp,
            "seed": args.seed})
    else:
        launch_gui()
        return
    if args.result_file:
        _rs_atomic_json(args.result_file, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.engine_probe:
        raise SystemExit(0 if result["ready"] else 1)



def app_write_crash(exc):
    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for location in (Path.home() / "AI_Unmarker_Crash.log", Path(tempfile.gettempdir()) / "AI_Unmarker_Crash.log"):
        try:
            location.write_text(message, encoding="utf-8")
            return str(location)
        except OSError:
            continue
    return "로그를 저장할 수 없습니다."


def app_error_payload(exc, stage="이미지 처리"):
    """Retain the failing stage and exception chain for the UI and support log.

    The preview's input scan is not a processing error report. Capture the
    exception here, in the worker that caught it, before its traceback is lost.
    Logging is best effort: a full or unwritable disk must not hide the error.
    """
    sanitize = globals().get("_rs_safe_text", str)
    error = sanitize(f"{type(exc).__name__}: {exc}")
    trace = sanitize("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    causes, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(sanitize(f"{type(current).__name__}: {current}"))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__)
    # The outer exception provides context; the innermost cause identifies the
    # missing module, bad argument or device failure that actually stopped work.
    detail = " ".join((causes[-1] if causes else error).split())
    if len(detail) > 280:
        detail = detail[:277] + "…"
    payload = {"stage": sanitize(stage or "이미지 처리"), "error": error,
               "detail": detail, "causes": causes, "traceback": trace, "log_path": ""}
    directories = []
    try:
        directories.append(_rs_root() / "logs")
    except Exception:
        pass
    directories.append(Path(tempfile.gettempdir()) / "AI-Unmarker-Pro-logs")
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, filename = tempfile.mkstemp(
                prefix="processing-error-", suffix=".log", dir=directory)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write("AI Unmarker Pro\n단계: " + payload["stage"] + "\n\n" + trace)
            payload["log_path"] = filename
            break
        except OSError:
            continue
    return payload


# ui_assets.py
# ------------------------------------------------------------------------
import ctypes
import ctypes.util
import os
from pathlib import Path
import struct
import sys

_UA_STATE = None
_UA_LIBRARIES = {}
_UA_ICONS = {}
_UA_GLYPHS = {
    "upload-simple": 0xE4C0,
    "image": 0xE2CA,
    "download-simple": 0xE20C,
    "arrow-counter-clockwise": 0xE038,
    "x": 0xE4F6,
    "check": 0xE182,
    "info": 0xE2CE,
    "magnifying-glass": 0xE30C,
    "eye": 0xE220,
    "check-circle": 0xE184,
    "warning-circle": 0xE4E2,
    "circle-notch": 0xEB44,
}


def ui_resource_roots():
    roots = []
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS))
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        roots.append(executable_dir)
        if sys.platform == "darwin" and executable_dir.name == "MacOS":
            roots.append(executable_dir.parent / "Resources")
    roots.append(Path(__file__).resolve().parent)
    return list(dict.fromkeys(path.resolve() for path in roots))


def ui_app_icon_candidates():
    """Prefer every project icon before the bundled Phosphor fallback."""
    names = ("icon.ico", "icon.png", "icon.icns") if sys.platform == "win32" else ("icon.png", "icon.icns", "icon.ico")
    candidates = []
    for root in ui_resource_roots():
        for directory in (root / "icons", root):
            for name in names:
                candidate = directory / name
                if candidate not in candidates:
                    candidates.append(candidate)
    for root in ui_resource_roots():
        for name in names:
            candidate = root / "assets" / "icons" / name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def ui_apply_app_identity(window, title="AI Unmarker Pro"):
    """Apply the original app name/icon through Tk's public window APIs."""
    import tkinter as tk
    window.title(title)
    window.tk.call("tk", "appname", "AI Unmarker Pro")
    candidates = ui_app_icon_candidates()
    result = {"name": "AI Unmarker Pro", "title": title, "icon_applied": False,
              "icon_path": None, "searched": [str(path) for path in candidates], "errors": []}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if sys.platform == "win32" and path.suffix.lower() == ".ico":
                window.iconbitmap(str(path))
            else:
                if path.suffix.lower() == ".png":
                    photo = tk.PhotoImage(master=window, file=str(path))
                else:
                    # Pillow is already present in the main desktop UI. A
                    # minimal first-run bootstrap can continue to a PNG when
                    # Pillow has not been installed yet.
                    from PIL import Image, ImageTk
                    with Image.open(path) as source:
                        photo = ImageTk.PhotoImage(source.convert("RGBA"), master=window)
                window.iconphoto(True, photo)
                window._ua_app_icon_photo = photo  # keep the Tk image alive
                if sys.platform == "win32":
                    # CTk installs its default Windows icon after 200 ms when
                    # no .ico was used. Reapply the user's PNG through public
                    # Tk methods after that scheduled initialization.
                    def retain_project_icon():
                        try:
                            if window.winfo_exists():
                                window.iconphoto(True, photo)
                        except tk.TclError:
                            pass
                    window.after(250, retain_project_icon)
            result.update(icon_applied=True, icon_path=str(path))
            return result
        except (ImportError, OSError, ValueError, tk.TclError) as exc:
            result["errors"].append(f"{path}: {type(exc).__name__}: {exc}")
    return result


def _ua_font_table(path, tag):
    """Read a bounded SFNT table without depending on Pillow during first setup."""
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        header = stream.read(12)
        if len(header) != 12 or header[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
            return b""
        count = struct.unpack_from(">H", header, 4)[0]
        if count > 256:
            return b""
        records = stream.read(count * 16)
        if len(records) != count * 16:
            return b""
        for position in range(0, len(records), 16):
            table, _checksum, offset, length = struct.unpack_from(">4sIII", records, position)
            if table == tag and length <= 8 * 1024 * 1024 and offset + length <= size:
                stream.seek(offset)
                return stream.read(length)
    return b""


def _ua_font_family(path):
    data = _ua_font_table(path, b"name")
    if len(data) < 6:
        return ""
    _format, count, storage = struct.unpack_from(">HHH", data)
    names = []
    for index in range(min(count, (len(data) - 6) // 12)):
        platform_id, _encoding, language, name_id, length, offset = struct.unpack_from(">HHHHHH", data, 6 + index * 12)
        if name_id not in (1, 16) or storage + offset + length > len(data):
            continue
        encoding = "utf-16-be" if platform_id in (0, 3) else "mac_roman"
        try:
            value = data[storage + offset:storage + offset + length].decode(encoding).strip("\x00 ")
        except UnicodeError:
            continue
        if value:
            priority = (name_id == 16, language in (0, 0x409), platform_id in (0, 3))
            names.append((priority, value))
    return max(names)[1] if names else ""


def _ua_has_glyph(path, codepoint):
    data = _ua_font_table(path, b"cmap")
    if len(data) < 4:
        return False
    count = struct.unpack_from(">H", data, 2)[0]
    for index in range(min(count, (len(data) - 4) // 8)):
        platform_id, encoding, offset = struct.unpack_from(">HHI", data, 4 + index * 8)
        if platform_id not in (0, 3) or (platform_id == 3 and encoding not in (1, 10)) or offset + 4 > len(data):
            continue
        form = struct.unpack_from(">H", data, offset)[0]
        if form == 12 and offset + 16 <= len(data):
            length, _language, groups = struct.unpack_from(">III", data, offset + 4)
            if offset + length > len(data) or 16 + groups * 12 > length:
                continue
            for group in range(groups):
                start, end, glyph = struct.unpack_from(">III", data, offset + 16 + group * 12)
                if start <= codepoint <= end:
                    return glyph + codepoint - start != 0
        elif form == 4 and offset + 16 <= len(data) and codepoint <= 0xFFFF:
            length = struct.unpack_from(">H", data, offset + 2)[0]
            segments = struct.unpack_from(">H", data, offset + 6)[0] // 2
            if offset + length > len(data) or 16 + segments * 8 > length:
                continue
            end_base = offset + 14
            start_base = end_base + segments * 2 + 2
            delta_base = start_base + segments * 2
            range_base = delta_base + segments * 2
            for segment in range(segments):
                start = struct.unpack_from(">H", data, start_base + segment * 2)[0]
                end = struct.unpack_from(">H", data, end_base + segment * 2)[0]
                if not start <= codepoint <= end:
                    continue
                delta = struct.unpack_from(">h", data, delta_base + segment * 2)[0]
                range_position = range_base + segment * 2
                distance = struct.unpack_from(">H", data, range_position)[0]
                if not distance:
                    return (codepoint + delta) & 0xFFFF != 0
                glyph_position = range_position + distance + 2 * (codepoint - start)
                if glyph_position + 2 > offset + length:
                    return False
                glyph = struct.unpack_from(">H", data, glyph_position)[0]
                return bool(glyph and (glyph + delta) & 0xFFFF)
    return False


def _ua_register_font(path):
    """Register fonts for this process; never require a system-wide install."""
    if sys.platform == "darwin":
        if "coretext" not in _UA_LIBRARIES:
            core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            text = ctypes.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
            core.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
            core.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
            core.CFRelease.argtypes = [ctypes.c_void_p]
            core.CFRelease.restype = None
            text.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            text.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
            _UA_LIBRARIES["coretext"] = (core, text)
        core, text = _UA_LIBRARIES["coretext"]
        encoded = os.fsencode(path)
        url = core.CFURLCreateFromFileSystemRepresentation(None, encoded, len(encoded), False)
        if not url:
            return False
        error = ctypes.c_void_p()
        try:
            return bool(text.CTFontManagerRegisterFontsForURL(url, 1, ctypes.byref(error)))
        finally:
            if error.value:
                core.CFRelease(error)
            core.CFRelease(url)
    if sys.platform == "win32":
        gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi.AddFontResourceExW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
        gdi.AddFontResourceExW.restype = ctypes.c_int
        return bool(gdi.AddFontResourceExW(str(path), 0x10, None))
    if sys.platform.startswith("linux"):
        if "fontconfig" not in _UA_LIBRARIES:
            library = ctypes.util.find_library("fontconfig")
            if not library:
                return False
            fontconfig = ctypes.CDLL(library)
            fontconfig.FcConfigGetCurrent.restype = ctypes.c_void_p
            fontconfig.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            fontconfig.FcConfigAppFontAddFile.restype = ctypes.c_int
            _UA_LIBRARIES["fontconfig"] = fontconfig
        fontconfig = _UA_LIBRARIES["fontconfig"]
        return bool(fontconfig.FcConfigAppFontAddFile(fontconfig.FcConfigGetCurrent(), os.fsencode(path)))
    return False


def ui_load_fonts():
    global _UA_STATE
    if _UA_STATE is not None:
        return _UA_STATE
    records, errors = [], []
    seen = set()
    roots = ui_resource_roots()
    # Preserve the original fonts/ folder. The shipped fallback is used only
    # when the corresponding font family was not supplied by the project.
    directories = [(root / "fonts", 0) for root in roots]
    directories += [(root / "assets" / "fonts", 1) for root in roots]
    project_families = set()
    for directory, priority in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf") or not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                family = _ua_font_family(path)
                category = next((name for name in ("pretendard", "phosphor") if name in family.lower()), None)
                if not category or (priority and category in project_families):
                    continue
                registered = _ua_register_font(path)
                records.append({"path": str(path), "family": family, "registered": registered,
                                "priority": priority})
                if priority == 0:
                    project_families.add(category)
            except (OSError, ValueError, struct.error) as exc:
                errors.append(f"{path.name}: {exc}")
    text_fonts = [item for item in records if "pretendard" in item["family"].lower()]
    text_fonts.sort(key=lambda item: (item["priority"], item["family"].lower() != "pretendard",
                                      "regular" not in Path(item["path"]).stem.lower()))
    icon_fonts = [item for item in records if "phosphor" in item["family"].lower()]
    icon_fonts.sort(key=lambda item: (item["priority"], item["family"].lower() != "phosphor",
                                      "regular" not in item["path"].lower(), item["path"]))
    _UA_STATE = {"family": text_fonts[0]["family"] if text_fonts else "Pretendard",
                 "fonts": records, "icon_fonts": icon_fonts, "errors": errors}
    return _UA_STATE


def ui_font_family():
    return ui_load_fonts()["family"]


def ui_icon_image(name, size=24, color="#687180"):
    """Render the real Phosphor glyph from the local font, including U+E4C0."""
    codepoint = _UA_GLYPHS.get(name)
    if codepoint is None:
        return None
    size = max(8, int(size))
    key = (name, size, color)
    if key in _UA_ICONS:
        return _UA_ICONS[key].copy()
    from PIL import Image, ImageDraw, ImageFont
    for record in ui_load_fonts()["icon_fonts"]:
        try:
            if not _ua_has_glyph(record["path"], codepoint):
                continue
            scale = 4
            edge = size * scale
            font = ImageFont.truetype(record["path"], edge)
            glyph = chr(codepoint)
            left, top, right, bottom = font.getbbox(glyph)
            if right <= left or bottom <= top:
                continue
            result = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
            ImageDraw.Draw(result).text(((edge - right - left) / 2, (edge - bottom - top) / 2), glyph, font=font, fill=color)
            result = result.resize((size, size), Image.Resampling.LANCZOS)
            _UA_ICONS[key] = result
            return result.copy()
        except (OSError, ValueError, struct.error):
            continue
    return None


# bootstrap_ui.py
# ------------------------------------------------------------------------
import os

INITIAL_SETUP_APPROVED = os.environ.pop("UNMARK_SETUP_APPROVED", "") == "1"


def bootstrap_ui_dependencies():
    """Ask once, install in the app's private runtime, then open the desktop.

    Packaged releases already contain the UI; this path also makes a source
    launch recover missing Pillow/CustomTkinter without a terminal command.
    """
    global INITIAL_SETUP_APPROVED
    import queue
    import re
    import threading
    import tkinter as tk
    from tkinter import ttk

    font_name = ui_font_family()
    window = tk.Tk()
    ui_apply_app_identity(window, f"AI Unmarker Pro v{APP_VERSION} · 처음 사용 준비")
    window.geometry("560x380")
    window.minsize(560, 320)
    window.resizable(False, False)
    window.configure(bg="#F6F7F9")
    # Native Tk uses positive font sizes as points; CTk uses pixels. Match the
    # desktop's typography explicitly so this fallback does not appear larger.
    window.option_add("*Font", (font_name, -13))
    window.option_add("*Label.padX", 0)
    window.option_add("*Label.padY", 0)
    frame = tk.Frame(window, bg="white", padx=20, pady=20,
                     highlightbackground="#E2E5EA", highlightthickness=1)
    frame.pack(fill="both", expand=True, padx=20, pady=20)
    title = tk.Label(frame, text="시작 준비를 확인하고 있습니다", bg="white", fg="#17202D",
             font=(font_name, -20, "bold"), anchor="w", justify="left", wraplength=476)
    title.pack(fill="x", pady=(0, 12))
    message = tk.StringVar(value="이미지 처리에 필요한 실행 구성요소를 확인합니다.")
    tk.Label(frame, textvariable=message, bg="white", fg="#616B78", justify="left",
             anchor="w", wraplength=476).pack(fill="x")
    estimate = tk.Label(frame, text="시작 중", bg="white", fg="#17202D",
                        font=(font_name, -16, "bold"), anchor="w")
    estimate.pack(fill="x", pady=(16, 6))
    estimate_note = tk.StringVar(value="이 단계에서는 파일을 다운로드하지 않습니다.")
    tk.Label(frame, textvariable=estimate_note,
             bg="white", fg="#78818D", justify="left", anchor="w", wraplength=476,
             font=(font_name, -11)).pack(fill="x")
    status = tk.StringVar(value="")
    tk.Label(frame, textvariable=status, bg="white", fg="#616B78", anchor="w",
             wraplength=476, justify="left").pack(fill="x", pady=(12, 8))
    bar = ttk.Progressbar(frame, mode="indeterminate")
    actions = tk.Frame(frame, bg="white")
    actions.pack(side="bottom", fill="x", pady=(16, 0))
    events = queue.Queue()
    cancel = threading.Event()
    state = {"busy": False, "ready": False, "closed": False, "checking": True}

    def fit_window():
        # Long errors need more space than progress messages. Change only the
        # height so neither buttons nor text are squeezed and position stays put.
        window.update_idletasks()
        height = max(320, window.winfo_reqheight())
        if window.winfo_height() != height:
            window.geometry(f"560x{height}")

    def dismiss():
        state["closed"] = True
        cancel.set()
        window.destroy()

    def report(payload, detail=""):
        events.put(("progress", payload if isinstance(payload, dict) else {"title": str(payload), "detail": detail}))

    def setup_error_message(payload):
        fallback = "준비를 마치지 못했습니다. 오류 기록을 확인해 주세요."
        if isinstance(payload, dict):
            cause = str(payload.get("cause") or "").strip()
            raw = cause or payload.get("error") or payload.get("detail") or fallback
        else:
            raw = payload
        lines = str(raw or "").splitlines()
        causes = [line.split(":", 1)[1].strip() for line in lines
                  if line.lstrip().startswith("원인:") and line.split(":", 1)[1].strip()]
        if causes:
            lines = causes + lines
        elif any(line.lstrip().startswith("Traceback") for line in lines):
            lines = list(reversed(lines))
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("Traceback", "File \"", "File '", "로그 파일:", "설치 기록:")):
                continue
            line = re.sub(r"^[A-Za-z_][\w.]*(?:Error|Exception):\s*", "", line)
            path = re.search(r"https?://|[A-Za-z]:[\\/]|(?<!\w)/\S+", line)
            if path:
                line = line[:path.start()].rstrip(" :")
            if line:
                return line if len(line) <= 150 else line[:147].rstrip() + "…"
        return fallback

    def start():
        if state["busy"] or state["checking"] or state["closed"]:
            return
        state["busy"] = True
        cancel.clear()
        start_button.configure(state="disabled", text="준비 중…")
        later_button.configure(text="중단하고 닫기")
        title.configure(text="실행 구성요소 준비")
        message.set("이미지 처리에 필요한 구성요소를 다운로드합니다.")
        status.set("실행 구성요소 확인 중…")
        bar.pack(fill="x", before=actions)
        bar.start(15)
        fit_window()

        def work():
            try:
                # The main desktop validates native DnD after this bootstrap
                # closes. A worker must not create a second macOS Tk app.
                result = ensure_runtime_dependencies(report, cancel, include_gui=False)
                if not isinstance(result, dict) or (not result.get("ready") and not result.get("restart_required")):
                    failure = result if isinstance(result, dict) else {
                        "error": "실행 구성요소의 준비 상태를 확인하지 못했습니다."}
                    events.put(("error", failure))
                    return
                events.put(("ready", result))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True, name="unmark-first-setup").start()

    start_button = tk.Button(actions, text="확인 중", command=start, state="disabled",
                             bg="#17202D", fg="white", activebackground="#2C3645",
                             activeforeground="white", relief="flat", bd=0,
                             font=(font_name, -13, "bold"), padx=20, pady=10)
    start_button.pack(side="right")
    later_button = tk.Button(actions, text="앱 닫기", command=dismiss,
                             bg="#F0F2F5", fg="#46505E", relief="flat", bd=0,
                             font=(font_name, -13), padx=20, pady=10)
    later_button.pack(side="right", padx=(0, 8))

    def poll():
        global INITIAL_SETUP_APPROVED
        if state["closed"]:
            return
        layout_changed = False
        try:
            while True:
                kind, payload = events.get_nowait()
                layout_changed = True
                if kind == "checked":
                    state["checking"] = False
                    title.configure(text="처음 한 번, 사용 준비가 필요합니다")
                    estimate.configure(text="실행 구성요소 · 예상 1~3분")
                    estimate_note.set("이미 설치된 파일은 다시 받지 않습니다.\n인터넷 연결 속도와 설치 환경에 따라 달라집니다.")
                    action = "다운로드 시작"
                    message.set("다운로드와 설치가 끝나면 사용할 수 있습니다.")
                    start_button.configure(state="normal", text=action)
                    if INITIAL_SETUP_APPROVED:
                        window.after_idle(start)
                elif kind == "progress":
                    detail = str(payload.get("detail", ""))
                    progress_title = str(payload.get("title", ""))
                    if "기다" in detail:
                        friendly = "다른 실행 창의 준비가 끝나기를 기다리고 있습니다."
                    elif any(word in progress_title + detail for word in ("확인", "검사", "검증", "준비했습니다")):
                        friendly = "필요한 실행 구성요소를 확인하고 있습니다."
                    else:
                        friendly = "이미지 처리에 필요한 구성요소를 다운로드하고 있습니다."
                    title.configure(text="실행 구성요소 준비")
                    current, total = payload.get("current"), payload.get("total")
                    if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                        bar.stop()
                        bar.configure(mode="determinate", maximum=total, value=min(current, total))
                        status.set(f"{friendly} · {current / 1_000_000:.1f} / {total / 1_000_000:.1f} MB")
                    else:
                        bar.configure(mode="indeterminate")
                        bar.start(15)
                        status.set(friendly)
                elif kind == "ready":
                    if cancel.is_set():
                        dismiss()
                        return
                    try:
                        if isinstance(payload, dict) and payload.get("restart_required"):
                            app_restart_after_setup()
                            state["closed"] = True
                            window.destroy()
                            return
                        activate_runtime_packages()
                        INITIAL_SETUP_APPROVED = True
                    except Exception as exc:
                        events.put(("error", str(exc)))
                        continue
                    state["ready"] = True
                    state["closed"] = True
                    window.destroy()
                    return
                elif kind == "error":
                    if cancel.is_set():
                        dismiss()
                        return
                    state["busy"] = False
                    bar.stop()
                    bar.pack_forget()
                    message.set(setup_error_message(payload))
                    status.set("설치 기록은 앱의 오류 기록에 보관됩니다.")
                    try:
                        app_write_crash(RuntimeError(str(payload)))
                    except Exception:
                        pass
                    start_button.configure(state="normal", text="다시 시도")
                    later_button.configure(text="앱 닫기")
        except queue.Empty:
            pass
        if layout_changed:
            fit_window()
        window.after(80, poll)

    fit_window()
    window.protocol("WM_DELETE_WINDOW", dismiss)
    window.after(80, poll)
    def check_after_paint():
        def work():
            try:
                snapshot = globals().get("runtime_dependency_snapshot")
                result = snapshot() if callable(snapshot) else {}
                events.put(("checked", result if isinstance(result, dict) else {}))
            except Exception:
                # UI dependencies are already known missing. An unavailable
                # estimate must not be mistaken for permission to install.
                events.put(("checked", {}))
        if not state["closed"]:
            threading.Thread(target=work, daemon=True, name="unmark-first-check").start()
    window.after_idle(lambda: window.after(20, check_after_paint))
    window.mainloop()
    return state["ready"]


def app_restart_after_setup():
    """Start a fresh interpreter only when an active native dependency changed."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    environment = os.environ.copy()
    environment["UNMARK_SETUP_APPROVED"] = "1"
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    command = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve())]
    subprocess.Popen(command, env=environment, close_fds=True)


# desktop_ui.py
# ------------------------------------------------------------------------
def _normalise_dropped_image_path(tk_interpreter, data):
    """Read a single native TkDND file-list entry without rewriting its name."""
    import os
    from pathlib import Path
    import re
    import tkinter
    from urllib.parse import unquote, urlsplit

    try:
        paths = tk_interpreter.splitlist(data)
    except (tkinter.TclError, TypeError, ValueError) as exc:
        raise ValueError("파일 경로를 읽지 못했습니다. 클릭해서 이미지를 선택해 주세요.") from exc
    if not paths:
        raise ValueError("이미지 파일을 드래그 앤 드롭해 주세요.")
    if len(paths) != 1:
        raise ValueError("이미지를 한 장씩 드래그 앤 드롭해 주세요.")
    value = str(paths[0])
    if not value or "\x00" in value:
        raise ValueError("올바른 이미지 파일을 선택해 주세요.")
    # Finder/TkDND normally sends native paths. Decode URI escapes only for an
    # actual file URI: a literal '%' or '%20' in a filename must stay literal.
    if value.lower().startswith("file:"):
        try:
            uri = urlsplit(value)
            if uri.netloc.lower() not in ("", "localhost") or uri.query or uri.fragment:
                raise ValueError("remote or ambiguous file URI")
            value = unquote(uri.path, encoding="utf-8", errors="strict")
        except (UnicodeError, ValueError) as exc:
            raise ValueError("컴퓨터에 저장된 이미지 파일을 선택해 주세요.") from exc
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
    elif re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) and not re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError("웹 주소 대신 컴퓨터에 저장된 이미지 파일을 선택해 주세요.")
    if "\x00" in value or value.startswith(("//", "\\\\")):
        raise ValueError("컴퓨터에 저장된 이미지 파일을 선택해 주세요.")
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("이미지 파일의 전체 경로를 확인하지 못했습니다.")
        if path.is_dir():
            raise ValueError("폴더 대신 이미지 파일을 선택해 주세요.")
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError("PNG, JPG 또는 WEBP 이미지 파일을 선택해 주세요.")
        if not path.is_file():
            raise ValueError("이미지 파일을 찾을 수 없습니다. 파일 위치를 확인해 주세요.")
        return str(path.resolve(strict=True))
    except OSError as exc:
        raise ValueError("이미지 파일에 접근하지 못했습니다. 파일 위치와 권한을 확인해 주세요.") from exc


def _ui_local_model_progress(payload):
    """Describe the model/file named by an actual setup event."""
    title, detail = str(payload.get("title", "")), str(payload.get("detail", ""))
    phase = str(payload.get("phase", ""))
    text = title + " " + detail
    names = (("qwen", "Qwen 이미지 모델", "Frederic75/Qwen-Image-2512-GGUF"),
             ("lightning", "Lightning 가속 모델", "lightx2v/Qwen-Image-2512-Lightning"),
             ("vae", "이미지 변환 모델", "Qwen/Qwen-Image-2512"),
             ("embeddings", "처리 설정", "0xROOTPLS/DeSynth"))
    for index, (key, name, repo_hint) in enumerate(names, 1):
        repo = (globals().get("MODEL_SPECS", {}).get(key) or {}).get("repo_id", repo_hint)
        if repo not in text and repo_hint not in text:
            continue
        filename = detail.split(" · ", 1)[1] if " · " in detail else ""
        if phase == "retry":
            activity = "연결을 다시 시도합니다. 받은 파일은 이어서 사용합니다."
            friendly = name + " 다운로드를 이어가고 있습니다"
        elif phase == "connecting":
            activity = "다운로드 서버에 연결하고 있습니다."
            friendly = name + " 다운로드 연결 중"
        elif phase in ("verify", "ready"):
            activity = "받은 파일을 확인하고 있습니다."
            friendly = name + " 파일을 확인하고 있습니다"
        else:
            activity = "필요한 파일을 다운로드하고 있습니다."
            friendly = name + "을 다운로드하고 있습니다"
        return {"title": friendly, "headline": f"구성 {index}/{len(names)} · {name}",
                "activity": filename + "\n" + activity if filename else activity}
    if "모델" in text or (payload.get("total_bytes") is not None and phase in
                             ("download", "connecting", "retry", "verify", "ready")):
        return {"title": "이미지 처리 모델을 확인하고 있습니다",
                "headline": "Qwen · Lightning",
                "activity": detail or "모델 파일의 준비와 검증을 마친 뒤 사용할 수 있습니다."}
    return None


def _ui_metadata_evidence(scan):
    """Project observed file records into the existing summary/card widgets.

    The backend owns provenance interpretation. In particular, a preferred
    recorded application is a file-record observation, not a pixel-AI verdict.
    Old saved scan reports remain readable through the original field layout.
    """
    import re

    scan = scan if isinstance(scan, dict) else {}
    record = scan.get("metadata") or {}
    record = record if isinstance(record, dict) else {}
    fields = record.get("metadata") or {}
    fields = fields if isinstance(fields, dict) else {}
    observation = scan.get("metadata_observation") or {}
    observation = observation if isinstance(observation, dict) else {}
    labels = {"c2pa": "C2PA", "exif": "EXIF", "xmp": "XMP", "text": "텍스트",
              "jumbf": "JUMBF", "comment": "코멘트", "other": "기타"}
    # Presence is still grounded in the actual per-format parser result; a
    # summary/status flag must not manufacture a missing record category.
    categories = [key for key in labels
                  if isinstance(fields.get(key), dict) and fields[key].get("present") is True]
    declared = list(dict.fromkeys(str(item["name"])
        for item in (scan.get("synthid") or {}).get("declarations", [])
        if isinstance(item, dict) and item.get("name")))
    names = list(dict.fromkeys(declared + [labels[key] for key in categories]))
    apps = record.get("recorded_apps") or []
    apps = apps if isinstance(apps, list) else []
    preferred = observation.get("primary_app")
    ordered = ([preferred] if isinstance(preferred, dict) else []) + apps
    app_names = list(dict.fromkeys(re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
        for item in ordered if isinstance(item, dict) and item.get("name")))
    c2pa = fields.get("c2pa") or {}
    incomplete = (observation.get("status") == "incomplete" or c2pa.get("status") in
                  ("error", "unavailable", "malformed", "invalid", "external_reference"))
    return {"names": names, "categories": categories, "apps": app_names,
            "present": bool(names), "incomplete": incomplete,
            "checked": bool(fields)}


def _ui_compact_evidence(values, limit=3):
    return " · ".join(values[:limit]) + (f" 외 {len(values)-limit}종" if len(values) > limit else "")


def launch_gui():
    import json
    import math
    import os
    from pathlib import Path
    import queue
    import re
    import sys
    import tempfile
    import threading
    app_title = "AI Unmarker Pro v" + APP_VERSION
    try:
        import tkinter as tk
        from tkinter import filedialog, font as tkfont
    except ImportError as exc:
        raise RuntimeError("기본 화면 구성 요소 tkinter가 없습니다. tkinter가 포함된 Python 또는 실행파일 배포본을 사용해 주세요.") from exc
    try:
        import customtkinter as ctk
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        bootstrap = globals().get("bootstrap_ui_dependencies")
        if not callable(bootstrap):
            raise RuntimeError("화면 실행에 필요한 구성 요소가 없습니다. 실행파일 배포본을 사용해 주세요.") from exc
        if not bootstrap():
            return
        try:
            import customtkinter as ctk
            from PIL import Image, ImageDraw, ImageOps
        except ImportError as retry_exc:
            raise RuntimeError("화면 구성 요소를 준비하지 못했습니다. 앱을 다시 실행해 주세요.") from retry_exc

    # TkinterDnD.require() supports an existing CustomTkinter root. Import and
    # initialise it after building the window so a missing runtime component
    # can be retried after setup, without fixing the root's class at startup.
    COPY, REFUSE_DROP = "copy", "refuse_drop"

    C = {"bg": "#F5F6F8", "surface": "#FFFFFF", "line": "#E3E6EB",
         "ink": "#101827", "sub": "#687180", "soft": "#F1F3F6",
         "green": "#179C79", "amber": "#AD650F", "red": "#B54747"}
    font_name = ui_font_family()

    def thumb(path):
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGBA")
            img.thumbnail((1400, 1200), Image.Resampling.LANCZOS)
            return img.copy()

    def size_text(value):
        value = max(0, float(value or 0))
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1000 or unit == "GB":
                return "%s %s" % (("%.1f" % value) if unit != "B" else str(int(value)), unit)
            value /= 1000

    def duration_text(seconds):
        if seconds is None:
            return None
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        if seconds < 60:
            return "약 %d초 남음" % max(1, round(seconds))
        return "약 %d분 남음" % max(1, math.ceil(seconds / 60))

    def progress_payload(*args, **kwargs):
        if args and isinstance(args[0], dict):
            return dict(args[0])
        return {"title": str(args[0]) if args else str(kwargs.get("status", "준비 중")),
                "detail": str(args[1]) if len(args) > 1 else str(kwargs.get("detail", ""))}

    def setup_error_message(payload):
        fallback = "준비를 마치지 못했습니다. 상세 로그를 확인해 주세요."
        if isinstance(payload, dict):
            cause = str(payload.get("cause") or "").strip()
            raw = cause or payload.get("error") or payload.get("detail") or fallback
        else:
            raw = payload
        lines = str(raw or "").splitlines()
        causes = [line.split(":", 1)[1].strip() for line in lines
                  if line.lstrip().startswith("원인:") and line.split(":", 1)[1].strip()]
        if causes:
            lines = causes + lines
        elif any(line.lstrip().startswith("Traceback") for line in lines):
            lines = list(reversed(lines))
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("Traceback", "File \"", "File '", "로그 파일:", "설치 기록:")):
                continue
            line = re.sub(r"^[A-Za-z_][\w.]*(?:Error|Exception):\s*", "", line)
            path = re.search(r"https?://|[A-Za-z]:[\\/]|(?<!\w)/\S+", line)
            if path:
                line = line[:path.start()].rstrip(" :")
            if line:
                return line if len(line) <= 150 else line[:147].rstrip() + "…"
        return fallback

    class AIUnmarkerApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            if (ctk.DrawEngine.preferred_drawing_method == "font_shapes"
                    and "CustomTkinter_shapes_font" not in tkfont.families(self)):
                ctk.DrawEngine.preferred_drawing_method = "polygon_shapes"
            self.app_identity = ui_apply_app_identity(self, app_title)
            self.geometry("1040x700")
            self.minsize(840, 610)
            self.configure(fg_color=C["bg"])
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)
            self.events = queue.Queue()
            self.close_event = threading.Event()
            self.cancel_event = threading.Event()
            self.setup_cancel = threading.Event()
            self.job = 0
            self.busy = False
            self.phase = "idle"
            self.input_path = None
            self.input_scan = None
            self.output_scan = None
            self.result = None
            self.result_dir = None
            self.result_path = None
            self.saved_path = None
            self.thumbnails = {}
            self.icon_images = {}
            self.preview_image = None
            transparent_pixel = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            self.empty_preview_image = ctk.CTkImage(
                light_image=transparent_pixel, dark_image=transparent_pixel, size=(1, 1))
            self.preview_timer = None
            self.poll_timer = None
            self.setup_window = None
            self.setup_check_timer = None
            self.setup_checked = False
            self.setup_ready = False
            self.setup_busy = False
            self.setup_stage = "checking"
            self.setup_check_started = False
            self.setup_estimate_data = {}
            self.setup_error = None
            self.resume_after_setup = False
            self.logs = []
            self.last_error = None
            self.work_stage = ""
            self.dnd_working = False
            self.dnd_error = ""
            self.dnd_targets = {}
            self.gauge_image = None
            self.processing_options = {
                "restore_mode": DETAIL_RESTORE_POLICY["default_mode"],
                "unsharp": DETAIL_RESTORE_POLICY["default_unsharp"],
            }
            self._build()
            self._set_state("idle")
            if not self.app_identity["icon_applied"]:
                self.logs.append("앱 아이콘 파일을 찾거나 적용하지 못했습니다. 확인 경로:\n" +
                    "\n".join(self.app_identity["searched"] + self.app_identity["errors"]))
            self.protocol("WM_DELETE_WINDOW", self._close)
            self._init_drag_and_drop()
            self.poll_timer = self.after(60, self._poll)
            # Paint first, then inspect cached models before cold native imports.
            # A quick healthy start needs no dialog; slow checks use the same
            # existing compact setup window without requesting a download.
            self.after_idle(self._check_startup)

        def _init_drag_and_drop(self):
            if self.close_event.is_set() or self.dnd_working:
                return self.dnd_working
            try:
                import importlib
                importlib.invalidate_caches()
                from tkinterdnd2 import TkinterDnD, DND_FILES
                require = getattr(TkinterDnD, "require", None) or TkinterDnD._require
                self.TkdndVersion = require(self)
                self._dnd_file_type = DND_FILES
                self._register_drop_targets()
                self.dnd_working = True
                self.dnd_error = ""
                self.logs.append("드래그 앤 드롭 준비 완료 · tkdnd " + str(self.TkdndVersion))
            except Exception as exc:
                import traceback
                detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
                if detail != self.dnd_error:
                    self.logs.append("드래그 앤 드롭 시작 오류:\n" + detail +
                        "\n앱을 다시 실행해 재확인할 수 있습니다. 계속되면 이 상세 로그를 공유해 주세요.")
                self.dnd_working = False
                self.dnd_error = detail
            prefix = "\n" if self.upload_icon_image else ""
            self.empty_preview_text = prefix + (
                "클릭해 이미지를 첨부하세요.\n드래그 앤 드롭도 가능합니다.\nPNG · JPG · WEBP"
                if self.dnd_working else
                "드래그 앤 드롭을 사용할 수 없습니다.\n클릭해 이미지를 첨부하세요.\nPNG · JPG · WEBP")
            self._sync_setup_gate()
            return self.dnd_working

        def _register_drop_targets(self):
            # CTkFrame.winfo_children() hides its native drawing canvas.
            # Enumerate Tk's actual children so the empty label, its image and
            # every painted preview surface receive the same native events.
            pending = [self.preview_surface]
            while pending:
                widget = pending.pop()
                pending.extend(tk.Misc.winfo_children(widget))
                name = str(widget)
                if name in self.dnd_targets:
                    continue
                bindings = []
                try:
                    widget.drop_target_register(self._dnd_file_type)
                    for event, handler in (("<<DropEnter>>", self._drop_enter),
                                           ("<<DropPosition>>", self._drop_enter),
                                           ("<<Drop>>", self._drop)):
                        bindings.append((event, widget.dnd_bind(event, handler)))
                except Exception:
                    for event, funcid in bindings:
                        widget.tk.call("bind", widget._w, event, "")
                        if funcid:
                            widget.deletecommand(funcid)
                    try:
                        widget.drop_target_unregister()
                    except tk.TclError:
                        pass
                    raise
                self.dnd_targets[name] = widget

        def _drop_allowed(self):
            return (self.dnd_working and not self.close_event.is_set()
                    and self.setup_ready and not self.setup_busy and not self.busy)

        def _drop_enter(self, event):
            # Enter/position payloads need not include paths. Returning the
            # native action is required; returning None makes Finder refuse it.
            if not self._drop_allowed():
                return REFUSE_DROP
            actions = getattr(event, "actions", ())
            if isinstance(actions, str):
                try:
                    actions = self.tk.splitlist(actions)
                except tk.TclError:
                    return REFUSE_DROP
            return COPY if not actions or COPY in actions else REFUSE_DROP

        def _drop_message(self, message):
            self.logs.append("드래그 앤 드롭: " + str(message))
            # The caption is visible even when the idle right-hand panel is
            # intentionally hidden. Keep an existing image/result intact.
            self.caption.configure(text=str(message))

        def _remember_error(self, data, stage="이미지 처리"):
            payload = dict(data) if isinstance(data, dict) else {
                "stage": stage, "error": str(data), "detail": str(data)}
            payload["stage"] = str(payload.get("stage") or stage)
            payload["error"] = str(payload.get("error") or payload.get("detail") or "원인을 확인하지 못했습니다.")
            payload["detail"] = str(payload.get("detail") or payload["error"])
            self.last_error = payload
            self.logs.append("작업 오류:\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            raw = (payload["error"] + " " + payload["detail"]).lower()
            if "out of memory" in raw or "memoryerror" in raw:
                return "이미지를 처리할 메모리가 부족합니다. 다른 앱을 닫고 다시 시도해 주세요."
            if "permission" in raw or "access denied" in raw:
                return "파일에 접근하지 못했습니다. 저장 폴더의 권한을 확인해 주세요."
            if "no space" in raw or "disk full" in raw:
                return "저장 공간이 부족합니다. 여유 공간을 확보해 주세요."
            message = setup_error_message(payload)
            # Keep stack traces and native-library details in the details view.
            if re.search(r"[가-힣]", message):
                return message if len(message) <= 88 else message[:85].rstrip() + "…"
            return "이미지 처리 중 오류가 발생했습니다. 오류 상세에서 원인을 확인할 수 있습니다."

        def label(self, parent, text, size=13, bold=False, color=None, **kwargs):
            # CTkLabel defaults to a 28 px minimum, even for a single 11 px line.
            # Let text determine its height; gauges and controls opt in to a size.
            kwargs.setdefault("height", 0)
            return ctk.CTkLabel(parent, text=text, font=(font_name, size, "bold" if bold else "normal"),
                text_color=color or C["ink"], **kwargs)

        def button(self, parent, text, command, primary=False, **kwargs):
            return ctk.CTkButton(parent, text=text, command=command, corner_radius=8,
                font=(font_name, 13, "bold"), fg_color=C["ink"] if primary else C["soft"],
                hover_color="#293345" if primary else C["line"],
                text_color="#FFFFFF" if primary else C["ink"], **kwargs)

        def _icon_image(self, name, size, color):
            key = (name, size, color)
            if key not in self.icon_images:
                image = ui_icon_image(name, size, color)
                self.icon_images[key] = None if image is None else ctk.CTkImage(
                    light_image=image, dark_image=image, size=(size, size))
            return self.icon_images[key]

        def _set_preview_content(self, image=None, text=""):
            # CTk 5.2.2 does not clear the native image for image=None. Keep a
            # transparent CTkImage for empty states and replace the image via
            # the public API before changing text or releasing its old owner.
            display_image = self.empty_preview_image if image is None else image
            self.preview_label.configure(image=display_image)
            self.preview_image = display_image
            self.preview_label.configure(text=text)

        def _build(self):
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 16))
            title_row = ctk.CTkFrame(header, fg_color="transparent")
            title_row.pack(anchor="w")
            self.label(title_row, "AI Unmarker Pro", 26, True, height=32).pack(side="left")
            self.label(title_row, "v" + APP_VERSION, 11, True, color=C["sub"],
                fg_color=C["line"], corner_radius=5, width=60, height=24).pack(side="left", padx=(12, 0))
            self.label(header, "이미지를 처리하고 메타데이터를 삭제합니다.", 13,
                       color=C["sub"]).pack(anchor="w", pady=(6, 0))
            body = ctk.CTkFrame(self, fg_color="transparent")
            body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
            body.grid_columnconfigure(0, weight=1, minsize=340)
            body.grid_columnconfigure(1, weight=0, minsize=320)
            body.grid_rowconfigure(0, weight=1)
            preview = ctk.CTkFrame(body, fg_color=C["surface"], border_color=C["line"], border_width=1, corner_radius=12)
            self.preview_surface = preview
            preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
            preview.grid_columnconfigure(0, weight=1)
            preview.grid_rowconfigure(1, weight=1)
            preview_tools = ctk.CTkFrame(preview, fg_color="transparent")
            preview_tools.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
            self.preview_tools = preview_tools
            self.tabs = ctk.CTkSegmentedButton(preview_tools, values=["원본"], command=self._select_tab,
                font=(font_name, 12), fg_color=C["soft"], selected_color=C["line"],
                selected_hover_color=C["line"], unselected_color=C["surface"],
                unselected_hover_color=C["soft"], text_color=C["ink"], height=28)
            self.tabs.pack(side="left")
            self.tabs.set("원본")
            self.change_button = self.button(preview_tools, "파일 변경", self._pick, width=82, height=28)
            self.change_button.pack(side="right")
            self.preview_area = ctk.CTkFrame(preview, fg_color=C["surface"], corner_radius=0)
            self.preview_area.grid(row=1, column=0, sticky="nsew", padx=16)
            self.preview_area.bind("<Configure>", self._schedule_preview)
            self.upload_icon_image = self._icon_image("upload-simple", 44, C["sub"])
            self.empty_preview_text = ("\n" if self.upload_icon_image else "") + \
                "클릭해 이미지를 첨부하세요.\n드래그 앤 드롭도 가능합니다.\nPNG · JPG · WEBP"
            self.preview_label = self.label(self.preview_area,
                self.empty_preview_text, 14, image=self.upload_icon_image or self.empty_preview_image, compound="top",
                color=C["sub"], justify="center", wraplength=300, cursor="hand2")
            self.preview_label.place(relx=.5, rely=.5, anchor="center")
            self.preview_label.bind("<Button-1>", lambda _event: self._pick() if not self.input_path else None)
            self.preview_area.bind("<Button-1>", lambda _event: self._pick() if not self.input_path else None)
            self.caption = self.label(preview, "원본 파일은 그대로 보관됩니다.", 11, color=C["sub"], wraplength=470)
            self.caption.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 12))

            panel = ctk.CTkFrame(body, width=320, fg_color=C["surface"], border_color=C["line"], border_width=1, corner_radius=12)
            panel.grid(row=0, column=1, sticky="nsew")
            panel.grid_propagate(False)
            panel.grid_columnconfigure(0, weight=1)
            panel.grid_rowconfigure(1, weight=1)
            summary = ctk.CTkFrame(panel, fg_color="transparent")
            summary.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
            self.summary_panel = summary
            self.state_icon = ctk.CTkFrame(summary, width=210, height=108, fg_color="transparent")
            self.state_icon.pack(pady=(0, 6))
            self.state_icon.pack_propagate(False)
            self.gauge_arc = self.label(self.state_icon, "", width=210, height=108)
            self.gauge_arc.place(x=0, y=0)
            self.gauge_value = self.label(self.state_icon, "대기", 29, True, height=36)
            self.gauge_value.place(relx=.5, y=64, anchor="center")
            self.gauge_caption = self.label(self.state_icon, "처리 상태", 10, color=C["sub"], height=16)
            self.gauge_caption.place(relx=.5, y=91, anchor="center")
            self.state_title = self.label(summary, "", 18, True, justify="center", wraplength=282)
            self.state_title.pack(fill="x")
            self.state_detail = self.label(summary, "", 12, color=C["sub"], justify="center", wraplength=280)
            self.state_detail.pack(fill="x", pady=(6, 0))

            self.idle_intro = ctk.CTkFrame(panel, fg_color="transparent")
            self.idle_intro.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=20, pady=(24, 0))
            self.label(self.idle_intro, "이미지를 넣고,\n한 번에 처리하세요.", 19, True,
                justify="left", anchor="w", wraplength=270).pack(fill="x")
            feature_list = ctk.CTkFrame(self.idle_intro, fg_color="transparent")
            feature_list.pack(fill="x", pady=(16, 0))
            feature_list.grid_columnconfigure(1, weight=1)
            for row, text in enumerate(("로컬 모델을 사용한 이미지 재생성", "메타데이터·C2PA 기록 확인 및 삭제",
                                        "등록된 가시 워터마크 검사 및 제거", "원본과 결과를 비교하고 저장")):
                ctk.CTkFrame(feature_list, width=4, height=4, corner_radius=2, fg_color=C["sub"]).grid(
                    row=row, column=0, sticky="n", padx=(0, 8), pady=(6, 0))
                self.label(feature_list, text, 12, color=C["sub"], anchor="w",
                    justify="left", wraplength=248).grid(row=row, column=1, sticky="ew", pady=(0, 8))

            evidence = ctk.CTkScrollableFrame(panel, width=268, height=160,
                fg_color=C["surface"], corner_radius=0, border_width=0,
                scrollbar_fg_color=C["surface"], scrollbar_button_color=C["line"],
                scrollbar_button_hover_color="#C7CDD6")
            evidence.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(12, 0))
            self.evidence_scroll = evidence
            evidence.grid_columnconfigure(0, weight=1)
            evidence_header = ctk.CTkFrame(evidence, fg_color="transparent")
            evidence_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            self.evidence_title = self.label(evidence_header, "처리 안내", 11, True,
                color=C["sub"])
            self.evidence_title.pack(side="left")
            self.info_button = self.button(evidence_header, "상세", self._show_info, width=42, height=20)
            self.info_button.pack(side="right")
            self.synthid_button = self.button(evidence_header, "SynthID 공식 검사", self._open_synthid, width=112, height=20)
            self.synthid_button.pack(side="right", padx=(0, 6))
            self.evidence_labels = {}
            self.evidence_cards = {}
            for row, key in enumerate(("regeneration", "pixel", "metadata", "open_watermark", "visible"), 1):
                card = ctk.CTkFrame(evidence, fg_color=C["soft"], corner_radius=8)
                card.grid(row=row, column=0, sticky="ew", pady=(0, 6))
                card.grid_columnconfigure(0, weight=1)
                heading = self.label(card, "", 10, color=C["sub"], anchor="w")
                heading.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
                value = self.label(card, "", 13, True, anchor="w", justify="left", wraplength=244)
                value.grid(row=1, column=0, sticky="ew", padx=12)
                detail = self.label(card, "", 10, color=C["sub"], anchor="w", justify="left", wraplength=244)
                detail.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
                self.evidence_labels[key] = (heading, value, detail)
                self.evidence_cards[key] = card

            self.file_summary = self.label(preview, "", 11, color=C["sub"],
                justify="center", wraplength=470)
            self.file_summary.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 0))
            self.caption.grid(row=3, column=0, sticky="ew", padx=16, pady=(6, 12))
            actions = ctk.CTkFrame(panel, fg_color="transparent")
            actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(12, 16))
            self.action_panel = actions
            self.small_status = self.label(actions, "", 10, color=C["sub"],
                justify="center", wraplength=280)
            self.work_progress = ctk.CTkProgressBar(actions, mode="indeterminate", height=4,
                progress_color=C["ink"], fg_color=C["soft"])
            self.primary = self.button(actions, "이미지 선택", self._primary_action, primary=True, height=40)
            self.primary.pack(fill="x")
            self.options_button = self.button(actions, "처리 옵션 · 기본", self._show_processing_options, height=30)
            self.options_button.pack(fill="x", pady=(6, 0))
            self.secondary = self.button(actions, "다른 이미지", self._reset_and_pick, height=32)
            self.error_details = self.button(actions, "오류 상세", self._show_info, height=28)

        def _show_processing_options(self):
            """옵션은 다음 처리에 적용하며 기존 결과는 명시적 재처리 전까지 유지합니다."""
            if self.busy or not self.setup_ready:
                return
            dialog = ctk.CTkToplevel(self)
            dialog.title("이미지 처리 옵션")
            dialog.geometry("440x310")
            dialog.resizable(False, False)
            dialog.transient(self)
            self.after_idle(lambda: self._grab_setup(dialog))
            edge = tk.BooleanVar(dialog, self.processing_options["restore_mode"] == "edge")
            sharpen = tk.BooleanVar(dialog, self.processing_options["unsharp"] > 0)
            self.label(dialog, "디테일 복원 옵션", 18, True).pack(anchor="w", padx=24, pady=(22, 16))
            for title, variable in (("윤곽 보존 모드", edge), ("약한 선명화", sharpen)):
                ctk.CTkCheckBox(dialog, text=title, variable=variable, font=(font_name, 14),
                    fg_color=C["ink"], hover_color=C["sub"], text_color=C["ink"]).pack(
                        anchor="w", padx=24, pady=(0, 16))
            self.label(dialog, "두 옵션을 함께 사용할 수 있습니다.\n윤곽·질감 조정과 워터마크 검출 결과는 별개입니다.",
                12, color=C["sub"], justify="left", wraplength=390).pack(anchor="w", padx=24)
            footer = ctk.CTkFrame(dialog, fg_color="transparent")
            footer.pack(fill="x", padx=24, pady=(20, 20))
            def apply_options(process=False):
                self.processing_options.update(
                    restore_mode="edge" if edge.get() else "gaussian",
                    unsharp=DETAIL_RESTORE_POLICY["gentle_unsharp"] if sharpen.get() else 0.0)
                labels = (["윤곽 보존"] if edge.get() else []) + (["약한 선명화"] if sharpen.get() else [])
                self.options_button.configure(text="처리 옵션 · " + (" + ".join(labels) or "기본"))
                dialog.destroy()
                if process:
                    self._process()
            self.button(footer, "설정 저장", apply_options, height=34).pack(side="left")
            if self.input_path:
                self.button(footer, "적용 후 처리", lambda: apply_options(True),
                            primary=True, height=34).pack(side="right")

        def _set_small_status(self, text):
            self.small_status.configure(text=text)
            if text:
                before = self.work_progress if self.work_progress.winfo_manager() else self.primary
                self.small_status.pack(fill="x", pady=(0, 6), before=before)
            else:
                self.small_status.pack_forget()

        def _open_synthid(self):
            """제공사별 검사 범위를 표시하고 사용자가 선택한 검사만 실행합니다."""
            import webbrowser
            from tkinter import messagebox
            dialog = ctk.CTkToplevel(self)
            dialog.title("SynthID 공식 검사")
            dialog.geometry("580x470")
            dialog.transient(self)
            self.label(dialog, "제공사별 공식 검사", 18, True).pack(padx=20, pady=(20, 8))
            self.label(dialog,
                "원본과 저장한 처리본을 각각 검사해 주세요.\n"
                "C2PA 메타데이터와 SynthID 픽셀 판정은 별개의 결과입니다.\n"
                "페이지를 여는 것만으로 파일이 전송되지는 않습니다.",
                12, justify="left", wraplength=530).pack(padx=20, pady=8)
            for service in PROVENANCE_SERVICES.values():
                self.button(dialog, service["name"] + " 열기 · " + service["scope"],
                    lambda url=service["url"]: webbrowser.open(url, new=2), height=36).pack(fill="x", padx=20, pady=5)
            status = self.label(dialog, "미검출은 AI로 생성되지 않았다는 뜻이 아닙니다.", 12,
                justify="left", wraplength=530)
            status.pack(fill="x", padx=20, pady=12)
            responses = queue.Queue()

            def receive_result():
                if not dialog.winfo_exists():
                    return
                try:
                    result, error = responses.get_nowait()
                except queue.Empty:
                    dialog.after(150, receive_result)
                    return
                api_button.configure(state="normal")
                if error:
                    status.configure(text=error)
                    return
                labels = {"detected": "검출", "not_detected": "미검출", "not_tested": "검사 안 됨"}
                text = "OpenAI 공식 API 결과\n" + "\n".join(
                    name.upper() + " · " + labels[value["status"]] for name, value in result["signals"].items())
                status.configure(text=text + "\nGoogle 검사 결과는 별도로 확인해 주세요.")
                destination = filedialog.asksaveasfilename(parent=dialog, title="검사 결과 저장",
                    defaultextension=".json", initialfile="openai-verification.json",
                    filetypes=[("검사 결과", "*.json")])
                if destination:
                    try:
                        # 기존 파일은 덮어쓰지 않고 새 검사 기록으로 보존합니다.
                        with open(destination, "x", encoding="utf-8") as stream:
                            json.dump(result, stream, ensure_ascii=False, indent=2)
                    except OSError:
                        messagebox.showerror("저장", "다른 이름으로 저장해 주세요. 기존 파일은 덮어쓰지 않습니다.", parent=dialog)

            def run_api():
                path = filedialog.askopenfilename(parent=dialog, title="OpenAI에서 검사할 이미지 선택",
                    filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp")])
                if not path or not messagebox.askyesno("OpenAI 이미지 전송",
                    Path(path).name + " 파일을 OpenAI 공식 검사 API에 업로드합니다.\n"
                    "계정의 API 이용 조건과 요청 한도가 적용됩니다.\n전송하여 검사하시겠습니까?", parent=dialog):
                    return
                api_button.configure(state="disabled")
                status.configure(text="선택한 파일을 OpenAI에서 검사 중입니다.")

                def worker():
                    try:
                        responses.put((verify_openai_file(path, allow_upload=True), None))
                    except Exception as exc:
                        # 알려진 안전한 메시지만 화면에 표시하며 키나 본문을 기록하지 않습니다.
                        responses.put((None, str(exc) if isinstance(exc, ProvenanceError) else "검사를 완료하지 못했습니다."))
                threading.Thread(target=worker, daemon=True, name="unmark-provenance").start()
                dialog.after(150, receive_result)

            api_button = self.button(dialog, "선택한 이미지 OpenAI API 검사", run_api, height=36)
            api_button.pack(fill="x", padx=20, pady=5)
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                api_button.configure(state="disabled")
                self.label(dialog, "앱 내 API 검사는 OPENAI_API_KEY 설정 시 사용할 수 있습니다.\n"
                    "키가 없으면 위 공식 웹 검사를 이용해 주세요.", 11,
                    color=C["sub"], wraplength=530).pack(padx=20, pady=5)

        def _draw_icon(self, state, fraction=None):
            """A state indicator; a number is shown only for measured progress."""
            color = C["green"] if state in ("done", "saved") else (C["amber"] if state in ("error", "review") else C["sub"])
            valid = (state == "processing" and isinstance(fraction, (int, float))
                     and not isinstance(fraction, bool) and math.isfinite(fraction)
                     and 0 <= fraction <= 1)
            labels = {"idle": "대기", "loading": "확인", "ready": "확인", "processing": "처리",
                      "done": "완료", "saving": "저장", "saved": "저장", "error": "확인", "review": "검토"}
            self.gauge_value.configure(text=("%d%%" % int(fraction * 100)) if valid else labels.get(state, "확인"),
                text_color=color if state in ("review", "error", "done", "saved") else C["ink"])
            self.gauge_caption.configure(text="현재 단계 진행률" if valid else "처리 상태")
            scale = 3
            image = Image.new("RGBA", (210 * scale, 108 * scale), (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            bounds = tuple(value * scale for value in (14, 10, 196, 192))
            draw.arc(bounds, 180, 360, fill=C["line"], width=11 * scale)
            if valid and fraction > 0:
                draw.arc(bounds, 180, 180 + 180 * fraction, fill=C["ink"], width=11 * scale)
            elif state in ("done", "saved", "error", "review"):
                draw.arc(bounds, 180, 360, fill=color, width=11 * scale)
            self.gauge_image = ctk.CTkImage(light_image=image, dark_image=image, size=(210, 108))
            self.gauge_arc.configure(image=self.gauge_image, text="")

        def _refresh_evidence(self):
            after = self.tabs.get() == "처리 후"
            scan = (self.output_scan if after else self.input_scan) or {}
            failure = self.last_error
            regeneration = (self.result or {}).get("regeneration") or {}

            def card(key, title, value, detail="", color=None):
                heading, primary, secondary = self.evidence_labels[key]
                heading.configure(text=title)
                primary.configure(text=value, text_color=color or C["ink"])
                secondary.configure(text=detail)

            self.info_button.configure(text="상세", width=42,
                state="normal" if scan or failure else "disabled")
            self.evidence_title.configure(text="처리 후 확인" if after else "확인한 정보")
            if self.phase == "processing":
                card("regeneration", "이미지 재생성", "처리 중", self.work_stage or "로컬 모델을 준비하고 있습니다.")
            elif failure and self.phase == "error":
                card("regeneration", "이미지 재생성", "처리를 마치지 못했습니다",
                    "오류 상세에서 원인을 확인할 수 있습니다.", C["amber"])
            elif after and regeneration.get("processed") is True:
                modelids = regeneration.get("modelids") or {}
                model_names = [name for key, name in (("qwen", "Qwen"),
                    ("lightning", "Lightning"))
                    if isinstance(modelids.get(key), str) and modelids[key]]
                unchanged = regeneration.get("status") == "unchanged" or regeneration.get("pixels_changed") is False
                value = "모델 실행 후 픽셀 변경 없음" if unchanged else "이미지 처리 완료"
                note = " · ".join(model_names) if model_names else str(regeneration.get("engine") or "이미지 재생성을 마쳤습니다.")
                card("regeneration", "이미지 재생성", value, note, C["amber"] if unchanged else C["ink"])
            elif after:
                card("regeneration", "이미지 재생성", "실행 내역 없음", "상세에서 처리 내용을 확인해 주세요.")
            else:
                card("regeneration", "이미지 처리", "처리 대기", "이미지 처리 후 원본 디테일을 복원합니다.")
            frequency = scan.get("frequency") or {}
            before_frequency = (self.input_scan or {}).get("frequency") or {}
            peak = frequency.get("peak_count")
            before_peak = before_frequency.get("peak_count")
            if isinstance(peak, int) and not isinstance(peak, bool):
                value = f"피크 {peak:,}개" if peak else "뚜렷한 주파수 피크 없음"
                note = "이미지의 주파수 특성을 측정했습니다."
                if after and isinstance(before_peak, int) and not isinstance(before_peak, bool):
                    note = f"원본 {before_peak:,} → 처리 후 {peak:,}"
                card("pixel", "주파수 패턴", value, note)
            elif frequency.get("status") in ("unavailable", "error"):
                card("pixel", "주파수 패턴", "확인하지 못했습니다", "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            else:
                card("pixel", "주파수 패턴", "확인 중" if self.phase == "loading" else "검사 정보 없음")

            names = {"c2pa": "C2PA", "exif": "EXIF", "xmp": "XMP", "text": "텍스트",
                     "jumbf": "JUMBF", "comment": "코멘트", "other": "기타"}
            metadata = (scan.get("metadata") or {}).get("metadata") or {}
            original = ((self.input_scan or {}).get("metadata") or {}).get("metadata") or {}
            metadata_evidence = _ui_metadata_evidence(scan)
            present = metadata_evidence["names"]
            removed = [name for key, name in names.items()
                       if (original.get(key) or {}).get("present") is True
                       and (metadata.get(key) or {}).get("present") is False]
            # Provenance declarations are metadata, not a pixel-verifier result.
            # Keep them in the existing metadata card, using the actual names
            # read from the file. Credentials must not disappear behind "외 N종".
            declared = list(dict.fromkeys(str(item["name"])
                for item in (scan.get("synthid") or {}).get("declarations", [])
                if isinstance(item, dict) and item.get("name")))
            before_declared = list(dict.fromkeys(str(item["name"])
                for item in ((self.input_scan or {}).get("synthid") or {}).get("declarations", [])
                if isinstance(item, dict) and item.get("name")))
            removed = list(dict.fromkeys([name for name in before_declared if name not in declared] + removed))
            metadata_incomplete = metadata_evidence["incomplete"]

            def compact(values, limit=2):
                return " · ".join(values[:limit]) + (f" 외 {len(values)-limit}종" if len(values) > limit else "")

            if metadata_incomplete and not present:
                card("metadata", "메타데이터", "확인하지 못했습니다",
                    "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif after and removed and not present:
                card("metadata", "메타데이터", compact(removed) + " 삭제 확인", "출력 파일을 다시 검사했습니다.", C["green"])
            elif present:
                app_names = metadata_evidence["apps"]
                note = ("기록된 앱: " + app_names[0][:30]) if app_names else "파일에 기록된 정보입니다."
                if after:
                    note = "출력 파일에 남아 있는 기록입니다."
                if metadata_incomplete:
                    note = "상세에서 검사 내용을 확인해 주세요."
                card("metadata", "메타데이터", _ui_compact_evidence(present), note,
                    C["amber"] if after or metadata_incomplete else None)
            else:
                card("metadata", "메타데이터", "삭제 대상 정보 없음" if metadata else "확인 중",
                    "색상 프로필 유지" if (metadata.get("icc") or {}).get("present") else
                    "EXIF · XMP · C2PA · 텍스트 확인" if metadata else "")

            public = scan.get("open_watermarks") or {}
            original_public = (self.input_scan or {}).get("open_watermarks") or {}
            public_status = public.get("status")
            was_public = original_public.get("status") == "detected"
            public_matches = public.get("matches") or original_public.get("matches") or []
            public_names = list(dict.fromkeys(str(item.get("name") or "공개 패턴")
                for item in public_matches if isinstance(item, dict)))
            if public_status == "detected":
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", compact(public_names, 1) or "공개 패턴 확인",
                    "출력에서 다시 검출됐습니다." if after else "공개 코덱의 삽입 신호를 확인했습니다.", C["amber"])
            elif after and was_public:
                self.evidence_cards["open_watermark"].grid()
                if public_status == "not_detected":
                    card("open_watermark", "공개 비가시 워터마크", "재검사에서 검출되지 않음",
                        compact(public_names, 1) + " · 동일 검사 적용", C["green"])
                else:
                    card("open_watermark", "공개 비가시 워터마크", "재검사 확인 필요",
                        "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif public_status in ("error", "unavailable", "unsupported", "insufficient_data"):
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", "검사 결과 확인 필요",
                    "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif public_status == "not_detected":
                # A completed public-codec check is useful evidence even when
                # it misses. Do not hide it or let the metadata-only summary
                # imply that the untested proprietary signal is absent.
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", "등록된 공개 패턴과 불일치",
                    "SynthID 픽셀 신호는 검사하지 않았습니다.")
            else:
                self.evidence_cards["open_watermark"].grid_remove()

            visible = scan.get("visible") or {}
            status = visible.get("status")
            result_visible = (self.result or {}).get("visible") or {}
            show_visible = status in ("detected", "error", "unavailable") or (
                after and result_visible.get("status") in ("cleaned", "partial", "unvalidated"))
            if not show_visible:
                self.evidence_cards["visible"].grid_remove()
                return
            self.evidence_cards["visible"].grid()
            labels = list(dict.fromkeys(str(mark.get("label") or mark.get("key") or "표식")
                for mark in visible.get("marks", []) if isinstance(mark, dict)))
            if status == "detected":
                card("visible", "가시 워터마크", (compact(labels, 1) or "등록된 표식")[:32],
                    "출력에서 다시 확인됐습니다." if after else "처리할 때 해당 영역을 복원합니다.", C["amber"])
            elif after and status == "not_detected" and result_visible.get("status") == "cleaned":
                card("visible", "가시 워터마크", "제거 후 재검사 완료", "등록된 패턴이 다시 검출되지 않았습니다.", C["green"])
            else:
                card("visible", "가시 워터마크", "검사 결과 확인 필요", "상세에서 검사 내용을 확인해 주세요.", C["amber"])

        def _show_work_progress(self, payload):
            # Do not infer completion from elapsed time, tile labels, scores,
            # or a step index that has no reported total.
            fraction = payload.get("progress")
            if not (isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
                    and math.isfinite(fraction) and 0 <= fraction <= 1):
                completed, total = payload.get("completed"), payload.get("total")
                if (isinstance(completed, (int, float)) and not isinstance(completed, bool)
                        and isinstance(total, (int, float)) and not isinstance(total, bool)
                        and math.isfinite(completed) and math.isfinite(total) and 0 <= completed <= total and total > 0):
                    fraction = completed / total
                else:
                    fraction = None
            self._draw_icon("processing", fraction)
            self.work_progress.stop()
            if fraction is None:
                self.work_progress.configure(mode="indeterminate")
                self.work_progress.start()
            else:
                self.work_progress.configure(mode="determinate")
                self.work_progress.set(fraction)

        def _result_notice(self):
            if not self.result:
                return ""
            regeneration = self.result.get("regeneration") or {}
            preservation = regeneration.get("preservation_check") or {}
            if preservation.get("status") in ("review_required", "failed"):
                return "원본과 처리 결과를 비교해 주세요." + "\n" + "상세에서 검사 내용을 확인해 주세요."
            if regeneration.get("processed") is True and (regeneration.get("status") == "unchanged"
                    or regeneration.get("pixels_changed") is False):
                return "모델은 실행됐지만 픽셀 변경이 없습니다.\n처리 정보를 확인해 주세요."
            visible_status = self.result.get("visible", {}).get("status")
            if visible_status == "partial":
                return "가시 표식 일부가 남아 있습니다.\n원본과 결과를 비교해 주세요."
            if visible_status in ("unvalidated", "error"):
                return "가시 표식의 재검사를 마치지 못했습니다.\n상세에서 검사 내용을 확인해 주세요."
            # Rendering/encoding success is separate from the actual output
            # checks. Inspect their named result fields, never an arbitrary
            # detected=True flag or a missing legacy quality field.
            output_scan = self.result.get("output_scan") or getattr(self, "output_scan", None) or {}
            public_status = (output_scan.get("open_watermarks") or {}).get("status")
            if public_status == "detected":
                return ("공개 비가시 워터마크" + " · " + "출력에서 다시 검출됐습니다." + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            if public_status in ("error", "unavailable", "unsupported", "insufficient_data"):
                return ("공개 비가시 워터마크" + " · " + "재검사 확인 필요" + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            if (output_scan.get("metadata_observation") or {}).get("status") == "incomplete":
                return ("메타데이터" + " · " + "확인하지 못했습니다" + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            return ""

        def _set_state(self, state, detail=None):
            self.phase = state
            self.busy = state in ("loading", "processing", "saving")
            texts = {
                "idle": ("이미지를 선택해 주세요", "왼쪽에 이미지를 놓으면\n검사 결과를 확인할 수 있습니다.", "이미지 선택 후 처리"),
                "loading": ("이미지 확인 중", "선택한 이미지를 불러오고 있습니다.", "이미지 확인 중"),
                "ready": ("이미지 확인 완료", "이미지 재생성과 메타데이터 삭제를\n한 번에 실행합니다.", "이미지 처리"),
                "processing": ("이미지 처리 중", "로컬 모델로 처리하고 저장 파일을 확인합니다.", "이미지 처리 중"),
                "done": ("이미지 처리 완료", "원본과 결과를 비교한 뒤\n저장해 주세요.", "이미지 저장"),
                "saving": ("이미지 저장 중", "처리한 이미지를 저장하고 있습니다.", "이미지 저장 중"),
                "saved": ("저장 완료", "원본과 처리 결과를 비교해 주세요.", "이미지 저장"),
                "error": ("처리하지 못했습니다", "다시 시도하거나 처리 정보를\n확인해 주세요.", "다시 처리" if self.input_path else "이미지 선택"),
            }
            title, body, action = texts[state]
            notice = self._result_notice() if state in ("done", "saved") else ""
            visible_incomplete = (self.input_scan or {}).get("visible", {}).get("status") in ("unavailable", "error")
            public_incomplete = (self.input_scan or {}).get("open_watermarks", {}).get("status") in (
                "error", "unavailable", "unsupported", "insufficient_data")
            metadata_evidence = _ui_metadata_evidence(self.input_scan)
            credentials_incomplete = metadata_evidence["incomplete"]
            incomplete_scan = state == "ready" and (visible_incomplete or public_incomplete or credentials_incomplete)
            if incomplete_scan:
                title = "일부 검사가 필요합니다"
                body = ("메타데이터는 확인했습니다.\n처리 시 가시 워터마크를 다시 검사합니다."
                        if visible_incomplete and not credentials_incomplete else
                        "상세에서 검사 내용을 확인해 주세요.")
            if state == "ready" and metadata_evidence["present"]:
                # Keep the original heading, typography and actions. The
                # existing detail area now shows the actual scan evidence,
                # instead of repeating a generic processing instruction.
                observed = _ui_compact_evidence(metadata_evidence["names"])
                note = ("기록된 앱: " + metadata_evidence["apps"][0][:30]
                        if metadata_evidence["apps"] else "파일에 기록된 정보입니다.")
                if incomplete_scan:
                    note = "상세에서 검사 내용을 확인해 주세요."
                body = " · ".join(("메타데이터", observed)) + "\n" + note
            elif state == "ready" and metadata_evidence["checked"] and not incomplete_scan:
                public = (self.input_scan or {}).get("open_watermarks") or {}
                visible = (self.input_scan or {}).get("visible") or {}
                if public.get("status") != "detected" and visible.get("status") != "detected":
                    body = "메타데이터 · 삭제 대상 정보 없음" + "\n" + "SynthID · 검사 안 됨"
            if state == "done" and self.result:
                elapsed = self.result.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
                    body = f"{elapsed:.1f}초 만에 처리와 재검사를 마쳤습니다.\n원본과 결과를 비교해 주세요."
            if notice:
                if state == "done":
                    title = "결과 확인이 필요합니다"
                body = notice
            self.state_title.configure(text=title)
            self.state_detail.configure(text=detail or body)
            self.primary.configure(text=action, state="disabled" if self.busy or state == "idle" else "normal")
            self.change_button.configure(state="disabled" if self.busy else "normal")
            if self.input_path:
                self.preview_tools.grid()
                self.change_button.pack(side="right")
            else:
                self.preview_tools.grid_remove()
                self.change_button.pack_forget()
            self._draw_icon("review" if notice or incomplete_scan else state)
            self._refresh_evidence()
            if state in ("idle", "loading"):
                self._reset_evidence_scroll()
            self.secondary.pack_forget()
            self.error_details.pack_forget()
            self.work_progress.stop()
            self.work_progress.pack_forget()
            if self.busy:
                self.work_progress.pack(fill="x", pady=(0, 10), before=self.primary)
                self.work_progress.configure(mode="indeterminate")
                self.work_progress.start()
                if state != "saving":
                    self.secondary.configure(text="작업 중단", command=self._cancel, state="normal")
                    self.secondary.pack(fill="x", pady=(6, 0))
            elif state in ("done", "saved", "error"):
                self.secondary.configure(text="다른 이미지", command=self._reset_and_pick, state="normal")
                self.secondary.pack(fill="x", pady=(6, 0))
            if self.last_error and not self.busy:
                self.error_details.pack(fill="x", pady=(6, 0))
            self._set_small_status(Path(self.saved_path).name if state == "saved" and self.saved_path else "")
            self._sync_setup_gate()
            self._sync_idle_view()

        def _reset_evidence_scroll(self):
            parent = self.evidence_scroll.master
            while parent is not None and parent is not self:
                if isinstance(parent, tk.Canvas):
                    parent.yview_moveto(0)
                    return
                parent = getattr(parent, "master", None)

        def _sync_idle_view(self):
            if not self.input_scan:
                self.evidence_scroll.grid_remove()
                if self.phase in ("loading", "error"):
                    self.summary_panel.grid()
                    self.idle_intro.grid_remove()
                    self.action_panel.grid()
                else:
                    self.summary_panel.grid_remove()
                    self.idle_intro.grid()
                    self.action_panel.grid_remove()
            else:
                self.idle_intro.grid_remove()
                self.summary_panel.grid()
                self.evidence_scroll.grid()
                self.action_panel.grid()

        def _sync_setup_gate(self):
            locked = not self.setup_ready or self.setup_busy
            self.tabs.configure(state="disabled" if locked or self.busy else "normal")
            self.change_button.configure(state="disabled" if locked or self.busy else "normal")
            self.options_button.configure(state="disabled" if locked or self.busy else "normal")
            self.preview_label.configure(cursor="arrow" if locked else "hand2")
            if locked:
                checking = self.setup_stage == "checking"
                self.primary.configure(state="disabled", text="시작 준비 확인 중" if checking else "사용 준비 필요")
                self.state_title.configure(text="시작 준비 중" if checking else "사용 준비가 필요합니다")
                self.state_detail.configure(text="모델과 실행 구성요소 준비가 끝나면\n이미지를 선택할 수 있습니다.")
                self.secondary.configure(state="disabled")
                if not self.input_path:
                    self.preview_label.configure(text="사용 준비가 끝나면\n이미지를 선택할 수 있습니다.")
            elif not self.input_path and self.phase == "idle":
                self._set_preview_content(image=self.upload_icon_image, text=self.empty_preview_text)
            elif not self.input_path and self.phase == "error":
                self._set_preview_content(text="이미지를 열지 못했습니다.\n다른 파일을 선택해 주세요.")

        def _require_setup_ready(self):
            if self.close_event.is_set():
                return False
            if self.setup_ready and not self.setup_busy:
                return True
            self._sync_setup_gate()
            if self.setup_stage != "checking":
                self._open_setup()
            return False

        def _emit(self, job, kind, data=None):
            if self.close_event.is_set():
                if kind == "done" and data:
                    data[0].cleanup()
                return
            self.events.put((job, kind, data))

        def _check_startup(self):
            if self.close_event.is_set() or self.setup_check_started:
                return
            self.setup_check_started = True
            self.setup_stage = "checking"
            self.setup_ready = False
            self._set_small_status("앱 준비 상태 확인 중…")
            self._sync_setup_gate()
            # Cache presence does not mean cold imports finish immediately.
            # Show the existing progress window if checking takes long enough
            # to be visible; cancel this callback as soon as checking finishes.
            self.setup_check_timer = self.after(250, self._show_setup_checking)
            def worker():
                try:
                    # These checks read package metadata/cache headers only.
                    # A known missing model must show consent immediately,
                    # without waiting for a cold native-library import probe.
                    models = model_runtime_status()
                    estimate = setup_estimate()
                    snapshot = globals().get("runtime_dependency_snapshot")
                    dependencies = snapshot() if callable(snapshot) else {"metadata_only": True, "missing": []}
                    if models.get("cached") is not True or dependencies.get("missing") or dependencies.get("error_code"):
                        self._emit("setup", "checked", {"ready": False, "dependencies": dependencies,
                                   "models": models, "estimate": estimate})
                        return
                    if self.close_event.is_set() or self.setup_cancel.is_set():
                        return
                    dependencies = runtime_dependency_status(cancel_event=self.setup_cancel, include_gui=False)
                    ready = dependencies.get("ready") is True and models.get("cached") is True
                    self._emit("setup", "checked", {"ready": ready, "dependencies": dependencies,
                               "models": models, "estimate": estimate})
                except Exception as exc:
                    self._emit("setup", "checked", {"ready": False, "error": str(exc), "estimate": {}})
            def begin_check():
                if not self.close_event.is_set():
                    threading.Thread(target=worker, daemon=True, name="unmark-readiness").start()
            self.after(20, begin_check)

        def _show_setup_checking(self):
            self.setup_check_timer = None
            if (self.close_event.is_set() or self.setup_checked or self.setup_ready
                    or self.setup_stage != "checking"):
                return
            self._open_setup()

        def _cancel_setup_check_dialog(self):
            timer = self.setup_check_timer
            self.setup_check_timer = None
            if timer:
                try:
                    self.after_cancel(timer)
                except tk.TclError:
                    pass

        def _open_setup(self):
            if self.close_event.is_set() or self.setup_ready:
                return
            if self.setup_window and self.setup_window.winfo_exists():
                self.setup_window.lift()
                self._grab_setup(self.setup_window)
                return
            dialog = ctk.CTkToplevel(self)
            self.setup_window = dialog
            dialog.title(app_title + " · 시작 준비")
            dialog.geometry("560x380")
            dialog.minsize(560, 300)
            dialog.resizable(False, False)
            dialog.configure(fg_color=C["surface"])
            # A transient window is attached to its parent on macOS. Keep this
            # dialog independently movable; grab_set still enforces setup.
            dialog.grid_columnconfigure(0, weight=1)
            self.setup_title = self.label(dialog, "시작 준비를 확인하고 있습니다", 20, True,
                       wraplength=520, justify="left")
            self.setup_title.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 16))
            content = ctk.CTkFrame(dialog, fg_color=C["soft"], corner_radius=10)
            content.grid(row=1, column=0, sticky="ew", padx=20)
            self.setup_headline = self.label(content, "시작 중", 16, True, wraplength=488, justify="left")
            self.setup_headline.pack(anchor="w", padx=16, pady=(16, 6))
            self.setup_note = self.label(content, "", 12, color=C["sub"], wraplength=488, justify="left")
            self.setup_note.pack(anchor="w", padx=16)
            self.setup_activity = self.label(content, "", 12,
                color=C["sub"], wraplength=488, justify="left")
            self.setup_activity.pack(anchor="w", padx=16, pady=(12, 12))
            self.setup_bar = ctk.CTkProgressBar(content, height=6, mode="determinate",
                progress_color=C["ink"], fg_color=C["line"])
            self.setup_bar.set(0)
            self.setup_bar.pack(fill="x", padx=16, pady=(0, 8))
            self.setup_metrics = self.label(content, "", 11,
                color=C["sub"], wraplength=488, justify="left")
            self.setup_metrics.pack(anchor="w", padx=16, pady=(0, 16))
            footer = ctk.CTkFrame(dialog, fg_color="transparent")
            footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(16, 20))
            footer.grid_columnconfigure((0, 1), weight=1)
            self.setup_later = self.button(footer, "앱 닫기", self._defer_setup, height=40)
            self.setup_later.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            self.setup_start = self.button(footer, "확인 중", self._start_setup, primary=True, height=40, state="disabled")
            self.setup_start.grid(row=0, column=1, sticky="ew", padx=(6, 0))
            self.setup_log = self.button(footer, "상세 로그", self._show_logs, height=28)
            dialog.protocol("WM_DELETE_WINDOW", self._close)
            self.after_idle(lambda: self._grab_setup(dialog))
            if self.setup_stage == "checking":
                self._setup_checking_controls()
            elif self.setup_busy:
                self._setup_running_controls()
            else:
                self._setup_consent_controls()
            # Size and center after the initial text is laid out. Subsequent
            # progress and parent movement must never reset this position.
            self._fit_setup_window()
            self._center_dialog(dialog)
            self._sync_setup_gate()

        def _setup_checking_controls(self):
            self.setup_title.configure(text="시작 준비를 확인하고 있습니다")
            self.setup_headline.configure(text="시작 중")
            self.setup_note.configure(text="실행 구성요소와 저장된 모델 파일을 확인합니다.")
            self.setup_activity.configure(text="확인이 끝나면 다음 단계를 진행합니다.", text_color=C["sub"])
            self.setup_metrics.configure(text="이 단계에서는 파일을 다운로드하지 않습니다.")
            self.setup_start.configure(text="확인 중", state="disabled")
            self.setup_later.configure(text="앱 닫기", state="normal")
            self.setup_log.grid_remove()
            self.setup_bar.configure(mode="indeterminate")
            self.setup_bar.start()
            self._fit_setup_window()

        def _setup_consent_controls(self):
            self.setup_title.configure(text="처음 한 번, 사용 준비가 필요합니다")
            amount = self.setup_estimate_data.get("remaining_bytes")
            amount_text = "약 9.0 GB" if amount is None else size_text(amount)
            if amount == 0:
                headline = "기존 모델 파일을 확인합니다."
                note = "실행 구성요소와 파일을 확인한 뒤 사용할 수 있습니다.\n부족한 구성요소는 추가로 다운로드합니다."
                action = "준비 시작"
            else:
                headline = "필요한 다운로드 · " + amount_text
                estimate_range = self.setup_estimate_data.get("estimated_seconds_range")
                time_text = "예상 전송 약 5~20분"
                if isinstance(estimate_range, (list, tuple)) and len(estimate_range) == 2:
                    low, high = estimate_range
                    if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) and value >= 0 for value in (low, high)):
                        lower, upper = max(1, math.ceil(low / 60)), max(1, math.ceil(high / 60))
                        time_text = f"예상 전송 약 {lower}~{upper}분" if lower != upper else f"예상 전송 약 {upper}분"
                note = time_text + " · 100~300Mbps 기준 가정\n설치·파일 확인 시간은 별도이며 연결 속도에 따라 달라집니다."
                action = "다운로드 시작"
            self.setup_headline.configure(text=headline)
            self.setup_note.configure(text=note)
            activity = "Qwen · Lightning 모델을 컴퓨터에 저장합니다."
            legacy_bytes = self.setup_estimate_data.get("legacy_cache_bytes", 0)
            if isinstance(legacy_bytes, (int, float)) and legacy_bytes > 0:
                activity += "\n첫 처리 성공 후 이전 모델 파일 " + size_text(legacy_bytes) + "를 삭제합니다."
            self.setup_activity.configure(text=activity, text_color=C["sub"])
            self.setup_metrics.configure(text="다운로드와 파일 확인이 끝나면 사용할 수 있습니다.")
            self.setup_bar.stop()
            self.setup_bar.configure(mode="determinate")
            self.setup_bar.set(0)
            self.setup_start.configure(text=action, state="normal")
            self.setup_later.configure(text="앱 닫기", state="normal")
            self.setup_log.grid_remove()
            if self.setup_stage == "interrupted":
                self.setup_title.configure(text="사용 준비가 중단되었습니다")
                self.setup_headline.configure(text="사용 준비를 이어서 진행합니다")
                self.setup_note.configure(text="받은 파일과 이어받기 정보를 확인한 뒤\n필요한 준비를 계속합니다.")
                self.setup_activity.configure(text="준비를 마쳐야 이미지를 열고 처리할 수 있습니다.", text_color=C["sub"])
                self.setup_metrics.configure(text="다시 시작하면 확인된 파일과 이어받기 정보를 사용합니다.")
                self.setup_start.configure(text="준비 계속하기")
            if self.setup_error:
                self.setup_title.configure(text="사용 준비를 마치지 못했습니다")
                self.setup_activity.configure(text=setup_error_message(self.setup_error), text_color=C["red"])
                self.setup_metrics.configure(text="자세한 내용은 상세 로그에서 확인할 수 있습니다.")
                self.setup_start.configure(text="다시 시도")
                self.setup_log.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            self._fit_setup_window()

        def _fit_setup_window(self):
            """Fit changing setup text without moving the user's window."""
            dialog = self.setup_window
            if not dialog or not dialog.winfo_exists():
                return
            dialog.update_idletasks()
            dimensions = re.match(r"(\d+)x(\d+)", dialog.geometry())
            if dimensions is None:
                return
            width, height = map(int, dimensions.groups())
            # The public CTk geometry is logical; Tk requested/actual sizes are
            # physical. Convert only the measured content height back to CTk.
            scale = dialog.winfo_height() / max(1, height)
            fitted_height = max(300, math.ceil(dialog.winfo_reqheight() / scale))
            if fitted_height != height:
                dialog.geometry(f"{width}x{fitted_height}")

        def _center_dialog(self, dialog, width=None, height=None):
            # CTk scales requested dimensions, while winfo reports the actual
            # window pixels. Read the size after applying geometry so HiDPI
            # windows center correctly without any private scaling methods.
            if width is not None and height is not None:
                dialog.geometry(f"{width}x{height}")
            dialog.update_idletasks()
            parent = dialog.master or self
            x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
            # Tk accepts '+-N' as an absolute negative coordinate. Do not clamp
            # to the primary monitor: secondary monitors may be to its left.
            dialog.geometry(f"+{x}+{y}")

        def _grab_setup(self, dialog):
            if not self.close_event.is_set() and dialog.winfo_exists():
                try:
                    dialog.lift()
                    dialog.grab_set()
                except tk.TclError:
                    pass

        def _setup_running_controls(self):
            self.setup_title.configure(text="실행 구성요소를 확인하고 있습니다")
            self.setup_headline.configure(text="실행 구성요소")
            self.setup_note.configure(text="확인이 끝나면 필요한 모델 파일을 다운로드합니다.")
            self.setup_start.configure(state="disabled", text="준비 중")
            self.setup_later.configure(text="준비 중단", state="normal")
            self.setup_log.grid_remove()
            self.setup_activity.configure(text="실행 환경과 설치된 파일을 확인합니다.", text_color=C["sub"])
            self.setup_metrics.configure(text="다운로드와 파일 확인이 끝나면 사용할 수 있습니다.")
            self.setup_bar.configure(mode="indeterminate")
            self.setup_bar.start()
            self._fit_setup_window()

        def _start_setup(self):
            if self.close_event.is_set() or self.setup_busy or not self.setup_checked or self.setup_stage == "checking":
                return
            self.setup_busy = True
            self.setup_stage = "running"
            self.setup_ready = False
            self.setup_error = None
            self.setup_cancel = threading.Event()
            cancel = self.setup_cancel
            self._setup_running_controls()
            self._sync_setup_gate()
            def progress(*args, **kwargs):
                payload = progress_payload(*args, **kwargs)
                self._emit("setup", "progress", payload)
            def worker():
                try:
                    dependency_result = ensure_runtime_dependencies(progress, cancel, include_gui=False)
                    if cancel.is_set() or (isinstance(dependency_result, dict) and dependency_result.get("cancelled")):
                        detail = dependency_result.get("error", "") if isinstance(dependency_result, dict) else ""
                        log_path = dependency_result.get("log_path", "") if isinstance(dependency_result, dict) else ""
                        self._emit("setup", "cancelled", detail + ("\n로그 파일: " + str(log_path) if log_path else ""))
                        return
                    if isinstance(dependency_result, dict) and dependency_result.get("restart_required"):
                        self._emit("setup", "restart", dependency_result)
                        return
                    if not isinstance(dependency_result, dict) or dependency_result.get("ready") is not True:
                        failure = dependency_result if isinstance(dependency_result, dict) else {
                            "error": "실행에 필요한 구성 요소를 준비하지 못했습니다. 다시 시도해 주세요."}
                        self._emit("setup", "error", failure)
                        return
                    prepare_models(progress, cancel)
                    if cancel.is_set():
                        self._emit("setup", "cancelled")
                        return
                    # ensure_runtime_dependencies already validated real imports
                    # in a new process. Do not repeat that slow native probe.
                    models = model_runtime_status()
                    if dependency_result.get("ready") is not True or models.get("cached") is not True:
                        raise RuntimeError("사용 준비 후 확인에 실패했습니다.\n" + json.dumps(
                            {"dependencies": dependency_result, "models": models}, ensure_ascii=False))
                    if cancel.is_set():
                        self._emit("setup", "cancelled")
                        return
                    self._emit("setup", "ready")
                except Exception as exc:
                    self._emit("setup", "cancelled" if cancel.is_set() else "error", str(exc))
            threading.Thread(target=worker, daemon=True, name="unmark-setup").start()

        def _defer_setup(self):
            self.resume_after_setup = False
            if self.setup_busy:
                self.setup_stage = "stopping"
                self.setup_cancel.set()
                self.setup_later.configure(state="disabled", text="중단 요청됨")
                self.setup_activity.configure(text="준비를 중단하고 있습니다. 잠시만 기다려 주세요.")
                return
            self._close()

        def _destroy_setup(self):
            self._cancel_setup_check_dialog()
            if self.setup_window and self.setup_window.winfo_exists():
                self.setup_bar.stop()
                try:
                    self.setup_window.grab_release()
                except tk.TclError:
                    pass
                self.setup_window.destroy()
            self.setup_window = None

        def _setup_event(self, kind, data):
            if self.close_event.is_set():
                return
            if kind == "checked":
                self._cancel_setup_check_dialog()
                self.setup_checked = True
                ready = data.get("ready") is True and not self.setup_cancel.is_set()
                self.setup_ready = False
                self.setup_estimate_data = data.get("estimate") or {}
                self.logs.append("준비 상태: " + json.dumps(data, ensure_ascii=False, default=str))
                if not self.busy:
                    self._set_small_status("")
                if not ready:
                    self.setup_stage = "consent"
                    self.setup_error = data.get("error")
                    self._open_setup()
                    self._setup_consent_controls()
                    self._sync_setup_gate()
                    if globals().get("INITIAL_SETUP_APPROVED"):
                        self._start_setup()
                else:
                    # The dependency worker deliberately creates no Tk root.
                    # Native TkDND must succeed on this actual desktop root
                    # before input/processing is unlocked.
                    if not self._init_drag_and_drop():
                        self._setup_event("error", self.dnd_error)
                        return
                    self.setup_ready = True
                    self.setup_stage = "ready"
                    self._destroy_setup()
                    self._set_state("idle")
            elif kind == "progress":
                title, detail = str(data.get("title", "준비 중")), str(data.get("detail", ""))
                log_entry = title + (" · " + detail if detail else "")
                # Byte updates still refresh the UI, but repeated file names
                # must not push connection failures out of the diagnostic log.
                if not self.logs or self.logs[-1] != log_entry:
                    self.logs.append(log_entry)
                if not self.setup_window or not self.setup_window.winfo_exists():
                    return
                if self.setup_cancel.is_set():
                    return
                phase = str(data.get("phase", ""))
                model_progress = _ui_local_model_progress(data)
                if model_progress:
                    friendly = model_progress["title"]
                    activity = model_progress["activity"]
                    self.setup_headline.configure(text=model_progress["headline"])
                    self.setup_note.configure(text="이미지 재생성에 필요한 가중치를 컴퓨터에 준비합니다.")
                elif phase == "retry":
                    friendly = "다운로드 연결을 다시 시도하고 있습니다."
                    activity = "확인된 파일은 보관하고 필요한 설치를 이어갑니다."
                elif "설치" in title + detail or "다운로드" in title + detail:
                    friendly = "필요한 실행 구성요소를 다운로드하고 있습니다."
                    activity = "다운로드 후 설치한 구성요소를 확인합니다."
                else:
                    friendly = "실행 구성요소를 확인하고 있습니다."
                    activity = "이미지를 열고 처리할 준비가 되었는지 확인합니다."
                if not model_progress:
                    self.setup_headline.configure(text="실행 구성요소 준비")
                    self.setup_note.configure(text="구성요소를 확인한 뒤 필요한 모델 파일을 준비합니다.")
                self.setup_title.configure(text=friendly)
                self.setup_activity.configure(text=activity, text_color=C["sub"])
                total, done = data.get("total_bytes"), data.get("downloaded_bytes")
                ratio = data.get("progress")
                if total and done is not None:
                    ratio = float(done) / float(total)
                if isinstance(ratio, (int, float)) and math.isfinite(ratio):
                    self.setup_bar.stop()
                    self.setup_bar.configure(mode="determinate")
                    self.setup_bar.set(max(0, min(1, ratio)))
                else:
                    self.setup_bar.configure(mode="indeterminate")
                    self.setup_bar.start()
                pieces = []
                if done is not None and total:
                    pieces.append(size_text(done) + " / " + size_text(total))
                speed = data.get("bytes_per_second")
                if phase not in ("retry", "connecting") and speed and float(speed) > 0:
                    pieces.append(size_text(speed) + "/s")
                eta = duration_text(data.get("eta_seconds")) if phase not in ("retry", "connecting") else ""
                if eta:
                    pieces.append(eta)
                if phase == "retry":
                    attempt, maximum = data.get("retry_attempt"), data.get("max_retries")
                    if isinstance(attempt, int) and isinstance(maximum, int) and maximum > 0:
                        pieces.append(f"자동 재시도 {attempt}/{maximum}")
                elif phase == "connecting":
                    pieces.append("서버 연결 중")
                fallback = ("받은 파일을 확인하고 있습니다. 아직 준비 중입니다." if phase in ("verify", "ready")
                            else "받은 파일을 보관한 채 자동으로 다시 연결합니다." if phase == "retry"
                            else "다운로드 서버의 응답을 기다리고 있습니다." if phase == "connecting"
                            else "전송 속도를 확인하고 있습니다…" if phase == "download"
                            else "구성요소 확인이 끝나면 이미지 작업 화면이 열립니다.")
                self.setup_metrics.configure(text=" · ".join(pieces) if pieces else fallback)
                self._fit_setup_window()
            elif kind == "restart":
                if self.setup_cancel.is_set():
                    self._setup_event("cancelled", None)
                    return
                self.logs.append("설치한 구성 요소를 적용하기 위해 앱을 다시 시작합니다.")
                if self.setup_window and self.setup_window.winfo_exists():
                    self.setup_bar.stop()
                    self.setup_activity.configure(text="설치한 구성요소를 적용한 뒤 모델 준비를 이어갑니다.", text_color=C["sub"])
                    self.setup_start.configure(state="disabled", text="다시 시작 중")
                    self.setup_later.configure(state="disabled")
                try:
                    # This handler runs on Tk's main thread. The launcher carries
                    # the existing setup consent to the new interpreter.
                    app_restart_after_setup()
                except Exception as exc:
                    self._setup_event("error", "앱을 다시 시작하지 못했습니다: " + str(exc))
                    return
                self._close()
            elif kind == "ready":
                # A final result may already be queued when the user cancels.
                # Progress phase='ready' and 100% downloaded never unlock UI.
                if self.setup_cancel.is_set() or not self.setup_busy:
                    self._setup_event("cancelled", None)
                    return
                if not self._init_drag_and_drop():
                    self._setup_event("error", self.dnd_error)
                    return
                self.setup_busy = False
                self.setup_stage = "ready"
                self.setup_ready = True
                self.setup_checked = True
                self.setup_error = None
                self._destroy_setup()
                self.resume_after_setup = False
                self._set_state("idle")
            elif kind == "cancelled":
                self.setup_busy = False
                self.setup_stage = "interrupted"
                self.setup_ready = False
                self.setup_error = None
                self.resume_after_setup = False
                self.logs.append("사용 준비 중단" + (": " + str(data) if data else ""))
                self._open_setup()
                self._setup_consent_controls()
                self._sync_setup_gate()
            elif kind == "error":
                if self.setup_cancel.is_set():
                    self._setup_event("cancelled", data)
                    return
                self.setup_busy = False
                self.setup_stage = "error"
                self.setup_ready = False
                self.setup_error = data
                raw_error = json.dumps(data, ensure_ascii=False, indent=2, default=str) if isinstance(data, dict) else str(data)
                self.logs.append("준비 오류: " + raw_error)
                self.resume_after_setup = False
                if not self.setup_window or not self.setup_window.winfo_exists():
                    self._open_setup()
                self._setup_consent_controls()
                self._sync_setup_gate()

        def _pick(self):
            if not self._require_setup_ready() or self.busy:
                return
            path = filedialog.askopenfilename(parent=self, title="이미지 선택",
                filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp"), ("모든 파일", "*")])
            if path:
                self._load(path)

        def _drop(self, event):
            if self._drop_enter(event) != COPY:
                return REFUSE_DROP
            try:
                path = _normalise_dropped_image_path(self.tk, getattr(event, "data", ""))
                # Confirm the actual image header before replacing an existing
                # result. A text/PDF file renamed to .png must not clear it.
                with Image.open(path) as candidate:
                    if candidate.format not in ("PNG", "JPEG", "WEBP"):
                        raise ValueError("PNG, JPG 또는 WEBP 이미지 파일을 선택해 주세요.")
            except (OSError, ValueError, tk.TclError, Image.DecompressionBombError) as exc:
                self._drop_message(str(exc) if isinstance(exc, ValueError) else
                    "이미지를 열지 못했습니다. 파일 형식과 읽기 권한을 확인해 주세요.")
                return REFUSE_DROP
            return COPY if self._load(path) else REFUSE_DROP

        def _clear_result(self):
            self.result = None
            self.result_path = None
            self.output_scan = None
            self.saved_path = None
            if self.result_dir is not None:
                self.result_dir.cleanup()
                self.result_dir = None
            self.thumbnails.pop("처리 후", None)
            self.tabs.configure(values=["원본"])
            self.tabs.set("원본")

        def _load(self, path):
            if not self._require_setup_ready() or self.busy:
                return False
            self.last_error = None
            self.work_stage = ""
            self._clear_result()
            self.job += 1
            job = self.job
            self.cancel_event = threading.Event()
            cancel = self.cancel_event
            self.input_path = None
            self.input_scan = None
            self.thumbnails.clear()
            self._set_preview_content(text="이미지를 불러오고 있습니다…")
            self.file_summary.configure(text="")
            self.caption.configure(text="원본 파일은 그대로 보관됩니다.")
            self._set_state("loading")
            def worker():
                try:
                    source = str(Path(path).expanduser().resolve(strict=True))
                    scan = app_scan_file(source)
                    image = thumb(source)
                    if cancel.is_set():
                        self._emit(job, "cancelled")
                    else:
                        self._emit(job, "loaded", (source, scan, image))
                except Exception as exc:
                    self._emit(job, "cancelled" if cancel.is_set() else "error",
                               app_error_payload(exc, "이미지 불러오기"))
            threading.Thread(target=worker, daemon=True, name="unmark-load").start()
            return True

        def _primary_action(self):
            if not self._require_setup_ready() or self.busy:
                return
            if self.result_path and self.phase in ("done", "saved"):
                self._save()
            elif self.input_path:
                self._process()
            else:
                self._pick()

        def _process(self):
            if not self._require_setup_ready() or self.busy or not self.input_path:
                return
            self.last_error = None
            self.work_stage = "이미지 처리 준비"
            self._clear_result()
            self.job += 1
            job = self.job
            source = self.input_path
            self.cancel_event = threading.Event()
            cancel = self.cancel_event
            options = {"remove_visible": True, "pixel_process": False,
                "regenerate": True, "preserve_color_profile": True}
            options.update(self.processing_options)
            self._select_tab("원본")
            self._set_state("processing")
            stage = [self.work_stage]
            def progress(*args, **kwargs):
                payload = progress_payload(*args, **kwargs)
                stage[0] = str(payload.get("title") or stage[0])
                self._emit(job, "progress", payload)
            def worker():
                directory = None
                try:
                    directory = tempfile.TemporaryDirectory(prefix="unmark-preview-")
                    output = str(Path(directory.name) / "processed.png")
                    result = app_process_isolated(source, output, options, progress, cancel)
                    if cancel.is_set():
                        directory.cleanup()
                        self._emit(job, "cancelled")
                        return
                    stage[0] = "결과 미리보기"
                    image = thumb(output)
                    scan = result.get("output_scan") or app_scan_file(output)
                    self._emit(job, "done", (directory, result, scan, image, output))
                except Exception as exc:
                    if directory is not None:
                        directory.cleanup()
                    self._emit(job, "cancelled" if cancel.is_set() else "error",
                               app_error_payload(exc, stage[0]))
            threading.Thread(target=worker, daemon=True, name="unmark-process").start()

        def _save(self):
            if not self._require_setup_ready() or self.busy or not self.result_path:
                return
            source = Path(self.input_path)
            candidate = source.with_name(source.stem + "_cleaned.png")
            number = 2
            while candidate.exists():
                candidate = source.with_name(source.stem + "_cleaned_%d.png" % number)
                number += 1
            output = filedialog.asksaveasfilename(parent=self, title="이미지 저장", defaultextension=".png",
                initialdir=str(source.parent), initialfile=candidate.name, filetypes=[("PNG", "*.png")])
            if not output:
                return
            if Path(output).exists():
                self._set_small_status("기존 파일을 보존할 수 있도록 다른 이름으로 저장해 주세요.")
                return
            job = self.job
            processed = self.result_path
            self._set_state("saving")
            def worker():
                try:
                    app_export_result(processed, output)
                    self._emit(job, "saved", output)
                except Exception as exc:
                    self._emit(job, "save_error", app_error_payload(exc, "처리 결과 저장"))
            threading.Thread(target=worker, daemon=True, name="unmark-export").start()

        def _reset_and_pick(self):
            if not self._require_setup_ready() or self.busy:
                return
            self.last_error = None
            self.work_stage = ""
            self._clear_result()
            self.input_path = None
            self.input_scan = None
            self.thumbnails.clear()
            self._set_preview_content(image=self.upload_icon_image, text=self.empty_preview_text)
            self.file_summary.configure(text="")
            self.caption.configure(text="원본 파일은 그대로 보관됩니다.")
            self._set_state("idle")
            self._pick()

        def _poll(self):
            if self.close_event.is_set():
                return
            for _ in range(100):
                try:
                    job, kind, data = self.events.get_nowait()
                except queue.Empty:
                    break
                if job == "setup":
                    self._setup_event(kind, data)
                    if self.close_event.is_set():
                        return
                    continue
                if job != self.job:
                    if kind == "done":
                        data[0].cleanup()
                    continue
                if kind == "loaded":
                    self.last_error = None
                    self.input_path, self.input_scan, image = data
                    self.thumbnails["원본"] = image
                    self._set_state("ready")
                    self._select_tab("원본")
                elif kind == "progress":
                    title, detail = str(data.get("title", "")), str(data.get("detail", ""))
                    self.work_stage = title or self.work_stage
                    self.logs.append(title + (" · " + detail if detail else ""))
                    phase = str(data.get("phase", ""))
                    friendly = "이미지 처리 중"
                    message = "처리한 파일의 주파수 패턴과 메타데이터를 다시 확인합니다."
                    if phase == "encode":
                        friendly = "이미지 변환 중"
                        message = "처리에 필요한 이미지 정보를 읽고 있습니다."
                    elif phase == "decode":
                        friendly = "결과 이미지 생성 중"
                        message = "처리한 데이터를 이미지로 변환합니다."
                    elif phase == "detail_restore":
                        friendly = "원본 디테일 복원 중"
                        message = "처리된 이미지에 원본의 세부 표현을 되살립니다."
                    elif phase in ("regenerate", "inference", "model", "model_load", "load") or "모델" in title:
                        friendly = "이미지 처리 중" if phase in ("regenerate", "inference") else "로컬 모델 준비 중"
                        message = "이미지는 이 컴퓨터에서 처리합니다."
                    elif "파일 확인" in title:
                        friendly = "이미지 확인 중"
                    elif phase == "metadata" or "메타" in title:
                        friendly = "메타데이터 삭제 중"
                    elif phase in ("pixel", "pixel_process") or "픽셀" in title:
                        friendly = "이미지 픽셀 처리 중"
                    elif "가시" in title or "워터마크" in title:
                        friendly = "가시 워터마크 처리 중"
                    elif "검증" in title:
                        friendly = "처리 결과 확인 중"
                    self.state_title.configure(text=friendly)
                    self._show_work_progress(data)
                    self.state_detail.configure(text=message)
                    self._refresh_evidence()
                    self._set_small_status("취소할 수 있습니다.")
                elif kind == "done":
                    self.last_error = None
                    self.result_dir, self.result, self.output_scan, image, self.result_path = data
                    self.input_scan = self.result.get("input_scan") or self.input_scan
                    self.thumbnails["처리 후"] = image
                    self.tabs.configure(values=["원본", "처리 후"])
                    self.tabs.set("처리 후")
                    self._set_state("done")
                    self._select_tab("처리 후")
                elif kind == "saved":
                    self.last_error = None
                    self.saved_path = data
                    self._set_state("saved")
                elif kind == "save_error":
                    detail = self._remember_error(data, "처리 결과 저장")
                    self._set_state("done", "저장 실패: " + detail)
                elif kind == "cancelled":
                    self._set_state("ready" if self.input_path else "idle")
                    self._set_small_status("작업을 중단했습니다.")
                elif kind == "error":
                    detail = self._remember_error(data)
                    self._set_state("error", detail)
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]
            self.poll_timer = self.after(60, self._poll)

        def _select_tab(self, _value):
            if not self._require_setup_ready():
                return
            selected = self.tabs.get()
            scan = self.output_scan if selected == "처리 후" else self.input_scan
            path = self.result_path if selected == "처리 후" else self.input_path
            if scan:
                meta = scan.get("metadata", {})
                name = (Path(self.input_path).stem + " · 처리 후") if selected == "처리 후" else Path(self.input_path).name
                self.file_summary.configure(text=name if len(name) <= 64 else name[:61] + "…")
                self.caption.configure(text="%s · %s · %s × %s px" % (
                    selected, meta.get("format", ""), meta.get("width", "?"), meta.get("height", "?")))
            self._refresh_evidence()
            self._reset_evidence_scroll()
            self._render_preview()

        def _schedule_preview(self, _event):
            if self.preview_timer:
                self.after_cancel(self.preview_timer)
            self.preview_timer = self.after(90, self._render_preview)

        def _render_preview(self):
            self.preview_timer = None
            if self.close_event.is_set() or not self.setup_ready:
                return
            image = self.thumbnails.get(self.tabs.get())
            if image is None:
                return
            width, height = max(80, self.preview_area.winfo_width() - 8), max(80, self.preview_area.winfo_height() - 8)
            scale = min(width / image.width, height / image.height, 1)
            preview_image = ctk.CTkImage(light_image=image, dark_image=image,
                size=(max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            self._set_preview_content(image=preview_image)

        def _cancel(self):
            if self.busy:
                self.cancel_event.set()
                self.secondary.configure(state="disabled", text="중단 요청됨")
                self._set_small_status("작업을 중단하고 있습니다…")

        def _text_window(self, title, content):
            parent = self.setup_window if self.setup_window and self.setup_window.winfo_exists() else self
            dialog = ctk.CTkToplevel(parent)
            dialog.title(title)
            self._center_dialog(dialog, 640, 490)
            dialog.transient(parent)
            dialog.grid_columnconfigure(0, weight=1)
            dialog.grid_rowconfigure(1, weight=1)
            self.label(dialog, title, 20, True).grid(row=0, column=0, sticky="w", padx=24, pady=(22, 14))
            textbox = ctk.CTkTextbox(dialog, font=(font_name, 12), wrap="word",
                takefocus=True, exportselection=False, undo=False)
            textbox.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=24, pady=(0, 8))
            textbox.insert("1.0", content)
            textbox.configure(state="disabled")
            copy_status = self.label(dialog, "", 11, color=C["sub"], height=18, anchor="w")
            copy_status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(0, 14))

            def copy_text(selection_only=False):
                try:
                    if selection_only:
                        if not textbox.tag_ranges("sel"):
                            copy_status.configure(text="복사할 텍스트를 선택해 주세요.", text_color=C["sub"])
                            return "break"
                        text = textbox.get("sel.first", "sel.last")
                    else:
                        # Keep the complete report, including intentional final
                        # newlines; Tk's implicit terminal newline is not data.
                        text = content
                    # The app owns the clipboard so closing this popup does not
                    # discard the copied report while the app remains open.
                    self.clipboard_clear()
                    self.clipboard_append(text)
                except (tk.TclError, RuntimeError):
                    copy_status.configure(text="복사하지 못했습니다. 다시 시도해 주세요.", text_color=C["red"])
                else:
                    copy_status.configure(text="선택한 내용을 복사했습니다." if selection_only else "전체 내용을 복사했습니다.",
                        text_color=C["green"])
                return "break"

            def select_all(_event=None):
                textbox.focus_set()
                textbox.tag_remove("sel", "1.0", "end")
                textbox.tag_add("sel", "1.0", "end-1c")
                return "break"

            self.button(dialog, "전체 복사", copy_text, width=96, height=30).grid(
                row=0, column=1, sticky="e", padx=(8, 24), pady=(22, 14))
            menu = tk.Menu(dialog, tearoff=False)
            menu.add_command(label="선택 복사", command=lambda: copy_text(True))
            menu.add_command(label="전체 복사", command=copy_text)
            menu.add_separator()
            menu.add_command(label="전체 선택", command=select_all)

            def show_menu(event):
                textbox.focus_set()
                menu.entryconfigure(0, state="normal" if textbox.tag_ranges("sel") else "disabled")
                try:
                    menu.tk_popup(event.x_root, event.y_root)
                finally:
                    menu.grab_release()
                return "break"

            # Disabled Text widgets can still select/copy. Explicit focus and
            # bindings make that work consistently without enabling edits or
            # reaching into CustomTkinter's private native widget.
            textbox.bind("<Button-1>", lambda _event: textbox.focus_set(), add="+")
            copy_events = ["<<Copy>>", "<Control-c>", "<Control-C>"]
            select_events = ["<<SelectAll>>", "<Control-a>", "<Control-A>"]
            menu_events = ["<Button-3>"]
            if dialog.tk.call("tk", "windowingsystem") == "aqua":
                copy_events += ["<Command-c>", "<Command-C>"]
                select_events += ["<Command-a>", "<Command-A>"]
                menu_events += ["<Button-2>", "<Control-Button-1>"]
            for sequence in copy_events:
                textbox.bind(sequence, lambda _event: copy_text(True), add="+")
            for sequence in select_events:
                textbox.bind(sequence, select_all, add="+")
            for sequence in menu_events:
                textbox.bind(sequence, show_menu, add="+")
            for sequence in ("<<Cut>>", "<<Paste>>", "<<PasteSelection>>", "<<Clear>>", "<<Undo>>", "<<Redo>>"):
                textbox.bind(sequence, lambda _event: "break", add="+")
            textbox.focus_set()
            def close_info():
                dialog.destroy()
                if self.setup_window and self.setup_window.winfo_exists():
                    self._grab_setup(self.setup_window)
            dialog.protocol("WM_DELETE_WINDOW", close_info)
            if parent is not self:
                self.after(60, lambda: self._grab_setup(dialog))

        def _show_logs(self):
            identity = "실행 정보\n" + json.dumps(app_identity(), ensure_ascii=False, indent=2, default=str)
            self._text_window("상세 로그", identity + "\n\n" + ("\n".join(self.logs[-200:]) or "기록된 내용이 없습니다."))

        def _show_info(self):
            identity = "실행 정보\n" + json.dumps(app_identity(), ensure_ascii=False, indent=2, default=str)
            if self.last_error:
                error = self.last_error
                text = ("실패 단계: " + str(error.get("stage") or "이미지 처리") +
                        "\n원인: " + str(error.get("error") or error.get("detail") or "원인 미확인"))
                if error.get("log_path"):
                    text += "\n로그 파일: " + str(error["log_path"])
                if error.get("traceback"):
                    text += "\n\n상세 오류 기록\n" + str(error["traceback"])
                self._text_window("처리 오류 정보", text + "\n\n" + identity)
                return
            selected = self.output_scan if self.tabs.get() == "처리 후" else self.input_scan
            if not selected:
                self._show_logs()
                return
            text = ("주파수 패턴은 이미지에서 측정한 통계이며 AI 생성 확률이나 SynthID 판정값이 아닙니다.\n"
                    "로컬 모델의 이미지 재생성 내역과 파일의 메타데이터·등록된 가시 표식을 확인합니다.\n")
            text += "\n" + identity
            if self.result:
                text += "\n\n이미지 재생성 실행 내역\n" + json.dumps(self.result.get("regeneration") or {},
                    ensure_ascii=False, indent=2, default=str)
                pixel = self.result.get("pixel_processing") or {}
                if pixel.get("processed") is True:
                    text += "\n\n추가 픽셀 처리 실행 내역\n" + json.dumps(pixel, ensure_ascii=False, indent=2, default=str)
                text += "\n\n처리 전후 검사\n" + json.dumps({
                    "before": (self.input_scan or {}).get("frequency"),
                    "after": (self.output_scan or {}).get("frequency"),
                    "metadata_removed": self.result.get("metadata", {}).get("removed_metadata", []),
                    "visible": self.result.get("visible"),
                }, ensure_ascii=False, indent=2, default=str)
            text += "\n\n선택한 이미지 검사 정보\n" + json.dumps(selected, ensure_ascii=False, indent=2, default=str)
            self._text_window("처리 정보" if self.result else "원본 검사 정보", text)

        def _close(self):
            self.close_event.set()
            self.cancel_event.set()
            self.setup_cancel.set()
            for timer in (self.poll_timer, self.preview_timer):
                if timer:
                    try:
                        self.after_cancel(timer)
                    except tk.TclError:
                        pass
            self._destroy_setup()
            while not self.events.empty():
                try:
                    _job, kind, data = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "done" and data:
                    data[0].cleanup()
            if self.result_dir is not None and self.phase != "saving":
                self.result_dir.cleanup()
                self.result_dir = None
            self.destroy()

    ctk.set_appearance_mode("Light")
    try:
        app = AIUnmarkerApp()
    except tk.TclError as exc:
        raise RuntimeError("데스크톱 화면을 열 수 없습니다. GUI를 지원하는 환경에서 실행해 주세요.") from exc
    app.mainloop()



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        app_cli_main()
    except KeyboardInterrupt:
        print("작업을 중단했습니다.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        crash_log = app_write_crash(exc)
        message = f"{type(exc).__name__}: {exc}\n\n오류 기록: {crash_log}"
        print(message, file=sys.stderr)
        if len(sys.argv) == 1:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("AI Unmarker Pro 실행 오류", message, parent=root)
                root.destroy()
            except Exception:
                pass
        raise SystemExit(1)

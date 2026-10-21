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

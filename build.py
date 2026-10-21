"""Build the self-contained main.py as a native PyInstaller desktop application.

    python3 build.py                      # prepare private build environment and build
    python build.py --console             # diagnostic console build
    python build.py --icon icons/icon.png --font fonts/Pretendard-Regular.otf
    python build.py --dry-run             # no imports of application/model code
    python build.py --check               # dependency checks; no build

The generated spec implements --onedir and --windowed (unless --console is set).
Build on the destination OS and CPU architecture. This is not a cross-compiler.
Only installed Python package code, required shared libraries, package metadata,
visible-mark templates and supplied fonts/icons are collected. The local diffusion
runtime is included; model weights and caches are excluded and audited afterwards.
The bundled pip worker can repair required Python packages in the app's private
user directory. A finished application must pass its offline runtime probe.

Official references:
https://pyinstaller.org/en/stable/usage.html
https://pyinstaller.org/en/stable/hooks.html
https://pyinstaller.org/en/stable/spec-files.html
https://github.com/Eliav2/tkinterdnd2#pyinstaller
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import plistlib
import queue
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import textwrap
import threading
import time
import urllib.parse
import urllib.request
import uuid


APP_NAME = "AI Unmarker Pro"
APP_VERSION = "1.3.0"
ENGINE_ID = "local-qwen-q4-detail-v3"
RELEASE_ID = "2026-10-21.7"
BUILD_REVISION = RELEASE_ID
BUILD_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BUILD_SCRIPT_DIR
# Runtime requirements come from the selected main.py without importing it.
# Only build tools belong here, so updating main.py cannot leave a stale second
# runtime version table in the builder. requirements.txt is not needed to build.
BUILD_REQUIREMENTS = (
    "pyinstaller>=6.14,<7", "pip>=25.2,<27",
)
BOOTSTRAP_PIP_VERSION = "25.3"
PACKAGING_LAYOUT_VERSION = "original-assets-compact-ui-20261020-6"


def requirement_constraints(requirements) -> dict[str, str]:
    """Parse version-only requirements before pip or packaging is available."""
    parsed, names = {}, set()
    for requirement in requirements:
        match = re.fullmatch(
            r"([a-zA-Z0-9][a-zA-Z0-9_.-]*)(?:\[[a-zA-Z0-9_,.-]+\])?([<>=!~][a-zA-Z0-9.,<>=!~*+_-]+)",
            requirement if isinstance(requirement, str) else "",
        )
        if not match:
            raise RuntimeError(f"구성요소의 버전 조건이 올바르지 않습니다: {requirement!r}")
        name, constraint = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in names:
            raise RuntimeError(f"구성요소가 중복 지정되었습니다: {name}")
        names.add(normalized)
        parsed[name] = constraint
    if not parsed:
        raise RuntimeError("실행에 필요한 구성요소 목록이 비어 있습니다.")
    return parsed


def source_build_requirements(source: Path) -> tuple[str, ...]:
    """Use the same literal requirement table as main.py's runtime probe."""
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    declarations = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "_RS_REQUIREMENTS" for target in targets):
            declarations.append(node.value)
    if len(declarations) != 1:
        raise RuntimeError(f"실행 코드의 구성요소 목록(_RS_REQUIREMENTS)을 확인할 수 없습니다: {source}")
    try:
        manifest = ast.literal_eval(declarations[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise RuntimeError("실행 코드의 구성요소 목록은 고정된 사전 형식이어야 합니다.") from exc
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("실행 코드의 구성요소 목록이 비어 있거나 올바르지 않습니다.")
    requirements = []
    for name, entry in manifest.items():
        if not isinstance(name, str) or not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise RuntimeError(f"실행 코드의 구성요소 항목을 확인해 주세요: {name!r}")
        install, constraint = entry
        parsed_name, parsed_constraint = next(iter(requirement_constraints([install]).items()))
        if re.sub(r"[-_.]+", "-", name).lower() != re.sub(r"[-_.]+", "-", parsed_name).lower() or constraint != parsed_constraint:
            raise RuntimeError(f"설치와 실행 검사의 버전 조건이 다릅니다: {name}")
        requirements.append(install)
    runtime_names = {re.sub(r"[-_.]+", "-", name).lower() for name in requirement_constraints(requirements)}
    for requirement in BUILD_REQUIREMENTS:
        name = next(iter(requirement_constraints([requirement]))).lower()
        if name in {"pyinstaller", "pip"} and name not in runtime_names:
            requirements.append(requirement)
    requirement_constraints(requirements)
    return tuple(requirements)


# Native torchvision filenames vary by wheel. Preserve the installed names
# (_C_stable.so/image_stable.so included) rather than assuming obsolete modules.
TORCHVISION_HOOK = '''\
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries = collect_dynamic_libs(
    "torchvision", search_patterns=["*.so", "*.so.*", "*.dylib", "*.dll", "*.pyd"]
)
module_collection_mode = "pyz+py"
if not binaries:
    raise RuntimeError("torchvision의 네이티브 라이브러리를 찾지 못했습니다. 빌드를 중단합니다.")
'''
TORCHVISION_IMAGE_HOOK = '''\
# The actual native image codec files are collected by hook-torchvision.py.
hiddenimports = []
'''

# Explicit lazy pipeline imports and source files keep the local Qwen, GGUF,
# and Lightning APIs available without collecting any pretrained weights.
HIDDEN_IMPORTS = [
    "tkinter", "tkinter.filedialog", "tkinter.messagebox", "_tkinter",
    "customtkinter", "tkinterdnd2", "tkinterdnd2.TkinterDnD",
    "PIL.Image", "PIL.ImageOps", "PIL.ImageTk", "PIL.PngImagePlugin",
    "PIL.JpegImagePlugin", "PIL.WebPImagePlugin", "PIL.ImageCms",
    "truststore", "psutil", "numpy", "pywt", "pywt._extensions._dwt", "pywt._extensions._pywt", "cv2",
    "torch", "torchvision", "torchvision.transforms",
    "diffusers", "diffusers.configuration_utils", "diffusers.image_processor",
    "diffusers.pipelines.pipeline_utils",
    "diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img",
    "diffusers.models.transformers.transformer_qwenimage",
    "diffusers.models.autoencoders.autoencoder_kl_qwenimage",
    "diffusers.models.attention_processor", "diffusers.models.embeddings",
    "diffusers.schedulers.scheduling_flow_match_euler_discrete",
    "diffusers.loaders.single_file_model", "diffusers.loaders.single_file_utils",
    "diffusers.loaders.lora_pipeline", "diffusers.loaders.peft",
    "diffusers.quantizers.gguf.gguf_quantizer", "diffusers.quantizers.gguf.utils",
    "diffusers.quantizers.quantization_config", "diffusers.hooks.group_offloading",
    "transformers", "transformers.modeling_utils", "transformers.image_utils",
    "transformers.image_processing_utils", "transformers.feature_extraction_utils",
    "transformers.models.qwen2.configuration_qwen2",
    "transformers.models.qwen2.tokenization_qwen2",
    "transformers.models.qwen2.tokenization_qwen2_fast",
    "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "peft", "peft.mapping", "peft.tuners.lora.layer", "peft.tuners.lora.model",
    "peft.utils.save_and_load", "gguf", "gguf.gguf_reader", "gguf.gguf_writer",
    "gguf.constants", "gguf.quants",
    "tokenizers", "safetensors", "safetensors.torch", "accelerate",
    "huggingface_hub", "huggingface_hub.file_download",
    "huggingface_hub._snapshot_download", "huggingface_hub.utils",
    "regex", "filelock", "yaml", "tqdm", "tqdm.auto", "packaging", "packaging.version",
    "importlib.metadata", "importlib.resources", "requests", "certifi",
    "remove_ai_watermarks", "remove_ai_watermarks.image_io",
    "remove_ai_watermarks.watermark_registry", "c2pa", "c2pa.c2pa", "c2pa.lib",
    "pip", "pip.__main__", "pip._internal.cli.main", "pip._vendor",
    "runpy", "sysconfig", "ssl", "ctypes",
]
REQUIRED_MODEL_PACKAGES = {"torch", "torchvision", "diffusers", "transformers",
                           "huggingface-hub", "accelerate", "safetensors", "peft", "gguf"}

# Keep native image/GUI runtime libraries. These extensions are model payloads.
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".ggml", ".h5", ".hdf5"}
CACHE_PARTS = {".cache", ".huggingface", "snapshots", "blobs", "checkpoints", "__pycache__"}
# These are build/test payloads, never runtime libraries. In particular, an
# interrupted native-library rewrite may leave large hidden temporary copies
# which the upstream Torch hook's *.so.* pattern would otherwise collect.
NON_RUNTIME_ASSETS = (
    ("torch/lib/.lib*.so.*", "temporary_native_library_copy"),
    ("torch/bin/test_*", "torch_test_executable"),
    ("torch/bin/*test", "torch_test_executable"),
    ("torch/bin/*test.exe", "torch_test_executable"),
    ("torch/test/*", "torch_native_test_payload"),
    ("c2pa/libs/*.rlib", "c2pa_static_rust_archive"),
    ("c2pa/libs/*.d", "c2pa_build_dependency_manifest"),
)


def bundle_exclusion_reason(name):
    """Identify model/cache payloads and precisely scoped non-runtime assets."""
    path = Path(str(name).replace("\\", "/"))
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "model_weight"
    if any(part.lower() in CACHE_PARTS or part.lower().startswith("models--") for part in path.parts):
        return "model_or_python_cache"
    normalized = "/" + path.as_posix().lower().lstrip("/")
    for pattern, reason in NON_RUNTIME_ASSETS:
        if fnmatch.fnmatchcase(normalized, "*/" + pattern):
            return reason
    return None


def source_build_identity(source: Path) -> dict[str, str]:
    """Reject mixed releases or missing local model code before installation."""
    data = source.read_bytes()
    tree = ast.parse(data.decode("utf-8-sig"), filename=str(source))
    expected = {"APP_VERSION": APP_VERSION, "ENGINE_ID": ENGINE_ID, "RELEASE_ID": RELEASE_ID}
    actual = {}
    for name, value in expected.items():
        declarations = []
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                declarations.append(node.value)
        if len(declarations) != 1:
            raise RuntimeError(f"실행 코드와 빌드 코드가 같은 배포본이 아닙니다: {name} 확인 실패. v{APP_VERSION}의 main.py와 build.py를 함께 사용해 주세요.")
        try:
            found = ast.literal_eval(declarations[0])
        except (ValueError, TypeError, SyntaxError) as exc:
            raise RuntimeError(f"실행 코드의 {name}은 고정된 문자열이어야 합니다.") from exc
        if found != value:
            raise RuntimeError(f"실행 코드와 빌드 코드의 {name}이 다릅니다: 실행={found!r}, 빌드={value!r}. 같은 배포본의 main.py와 build.py를 함께 사용해 주세요.")
        actual[name] = found

    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for name in ("app_process", "regenerate_image", "model_engine_preflight"):
        if functions.count(name) != 1:
            raise RuntimeError(f"로컬 모델 실행 코드가 온전히 반영되지 않았습니다: {name} 확인 실패.")
    if "check_openai_provenance" in functions:
        raise RuntimeError("이 배포본에는 외부 AI 검사 API를 포함할 수 없습니다.")

    requirements = source_build_requirements(source)
    dependencies = {re.sub(r"[-_.]+", "-", name).lower() for name in requirement_constraints(requirements)}
    missing = REQUIRED_MODEL_PACKAGES - dependencies
    if missing:
        raise RuntimeError("로컬 모델 실행 구성요소가 빠져 있습니다: " + ", ".join(sorted(missing)))

    policies = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "MODEL_EXECUTION_POLICY" for target in targets):
            policies.append(node.value)
    try:
        policy = ast.literal_eval(policies[0]) if len(policies) == 1 else None
    except (ValueError, TypeError, SyntaxError):
        policy = None
    if not isinstance(policy, dict):
        raise RuntimeError("로컬 모델의 메모리 제한 설정(MODEL_EXECUTION_POLICY)을 확인할 수 없습니다.")
    for key, ceiling in (("max_working_side", 1024), ("batch_size", 1), ("threads", 2)):
        value = policy.get(key)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise RuntimeError(f"로컬 모델의 {key} 제한이 올바르지 않습니다: {value!r}; 최대 {ceiling}.")
    for key in ("cuda_memory_policy", "mps_memory_policy"):
        value = policy.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"로컬 모델의 메모리 관리 설정이 빠져 있습니다: {key}.")

    imported, imported_paths, imported_objects = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
            imported_paths.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported_paths.add(node.module)
            imported_objects.update(node.module + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            dynamic_import = (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            if dynamic_import:
                imported.add(node.args[0].value.split(".")[0])
                imported_paths.add(node.args[0].value)
    blocked_imports = sorted(name for name in imported if name in {"openai", "anthropic", "openai_provenance", "provenance_api"})
    if blocked_imports:
        raise RuntimeError("실행 코드에 외부 AI API 모듈이 포함되어 있습니다: " + ", ".join(blocked_imports))
    legacy_modules = ("diffusers.pipelines.stable_diffusion", "diffusers.pipelines.controlnet",
                      "diffusers.models.controlnets", "diffusers.models.unets.unet_2d_condition",
                      "diffusers.loaders.ip_adapter", "transformers.models.clip")
    legacy_objects = {"diffusers.ControlNetModel", "diffusers.StableDiffusionPipeline",
                      "diffusers.StableDiffusionImg2ImgPipeline", "diffusers.StableDiffusionControlNetImg2ImgPipeline",
                      "transformers.CLIPVisionModelWithProjection", "transformers.CLIPTextModel",
                      "transformers.CLIPImageProcessor", "transformers.CLIPTokenizer"}
    legacy = sorted((imported_objects & legacy_objects) | {
        name for name in imported_paths
        if any(name == prefix or name.startswith(prefix + ".") for prefix in legacy_modules)})
    if legacy:
        raise RuntimeError("이전 이미지 엔진이 실행 코드에 남아 있습니다. Qwen 배포본 전체를 사용해 주세요: " + ", ".join(legacy))
    return {"app_version": actual["APP_VERSION"], "engine_id": actual["ENGINE_ID"],
            "release_id": actual["RELEASE_ID"], "source_sha256": hashlib.sha256(data).hexdigest()}


def build_identity_hook(identity: dict[str, str]) -> str:
    """Embed the selected source digest into the executable, before main runs."""
    return "import sys\nsys._ai_unmarker_build = " + repr(dict(identity)) + "\n"


def verify_engine_result(result, expected: dict[str, str]) -> None:
    if not isinstance(result, dict):
        raise RuntimeError("이미지 엔진 실행 확인 결과가 올바른 JSON 객체가 아닙니다.")
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"이미지 엔진 실행 파일이 선택한 소스와 다릅니다: {key}={result.get(key)!r}, 예상={value!r}")
    preflight = result.get("model_preflight")
    if result.get("ready") is not True or not isinstance(preflight, dict) or preflight.get("ready") is not True:
        raise RuntimeError("로컬 모델 엔진이 CPU 실행 검사를 통과하지 못했습니다.")
    checked = preflight.get("checked")
    if (not isinstance(checked, list) or any(not isinstance(item, str) for item in checked)
            or not {"qwen", "gguf", "lightning", "vae", "embeddings", "detail_restore"}.issubset(checked)):
        raise RuntimeError("로컬 엔진 검사에서 Qwen, GGUF, Lightning, VAE, 임베딩, 디테일 복원 중 일부가 빠져 있습니다.")
    if preflight.get("device") != "cpu" or preflight.get("weights_loaded") is not False:
        raise RuntimeError("빌드 검사는 CPU 소형 연산만 사용해야 하며 다운로드 모델 가중치를 불러오면 안 됩니다.")
    if preflight.get("bounded_memory") is not True:
        raise RuntimeError("로컬 모델의 메모리 제한 설정이 실행 검사에서 확인되지 않았습니다.")


def build_environment_dir() -> Path:
    identity = f"{sys.implementation.cache_tag}-{platform.python_version()}-{sysconfig.get_platform()}-{platform.machine()}"
    return PROJECT_DIR / "build" / "venv" / re.sub(r"[^a-zA-Z0-9_.-]", "_", identity)


def build_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _private_venv_config(environment: Path) -> bool:
    try:
        settings = dict(line.split("=", 1) for line in (environment / "pyvenv.cfg").read_text(encoding="utf-8").splitlines() if "=" in line)
        return {key.strip().lower(): value.strip().lower() for key, value in settings.items()}.get("include-system-site-packages") == "false"
    except (OSError, ValueError):
        return False


def is_build_environment(environment: Path) -> bool:
    # A venv executable commonly symlinks to the system executable. Its resolved
    # filename alone cannot establish isolation; the actual prefix and cfg must agree.
    return (Path(sys.prefix).resolve() == environment.resolve()
            and Path(sys.base_prefix).resolve() != Path(sys.prefix).resolve()
            and _private_venv_config(environment))


def clean_build_environment(environment: Path) -> dict[str, str]:
    child = os.environ.copy()
    for key in list(child):
        if key.startswith(("PYTHON", "PIP_")) or key in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "__PYVENV_LAUNCHER__", "_PYTHON_SYSCONFIGDATA_NAME", "_PYTHON_HOST_PLATFORM"}:
            child.pop(key, None)
    # Do not rewrite HOME or the user's shell environment. All mutable caches
    # and configuration paths used by these child processes are project-private.
    private = PROJECT_DIR / "build" / "environment"
    child.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                  "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
                  "VIRTUAL_ENV": str(environment), "XDG_CACHE_HOME": str(private / "cache"),
                  "XDG_CONFIG_HOME": str(private / "config"), "HF_HOME": str(private / "huggingface"),
                  "TORCH_HOME": str(private / "torch"), "PYINSTALLER_CONFIG_DIR": str(private / "pyinstaller")})
    child["PATH"] = str(build_python(environment).parent) + os.pathsep + child.get("PATH", "")
    return child


def _environment_probe(environment: Path) -> bool:
    if not build_python(environment).is_file() or not _private_venv_config(environment):
        return False
    code = "import json,site,sys; print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,'user_site':site.ENABLE_USER_SITE}))"
    try:
        result = subprocess.run([str(build_python(environment)), "-I", "-B", "-c", code], env=clean_build_environment(environment),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, check=True)
        value = json.loads(result.stdout)
        return (Path(value["prefix"]).resolve() == environment.resolve()
                and Path(value["base"]).resolve() != environment.resolve() and value["user_site"] is False)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


@contextlib.contextmanager
def build_lock():
    directory = PROJECT_DIR / "build"
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "build.lock", "a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("이 프로젝트에서 다른 빌드가 실행 중입니다. 해당 빌드가 끝난 뒤 다시 실행해 주세요.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def new_build_log(prefix="build") -> Path:
    directory = PROJECT_DIR / "build" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.log"


def run_logged(command, log_path: Path, environment, *, label="명령", timeout=5400, check=True) -> int:
    """Stream child output to the terminal and a file, with a bounded lifetime."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines, recent = queue.Queue(), []
    options = {"start_new_session": True} if os.name != "nt" else {}
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        log.write("Command: " + subprocess.list2cmdline(list(map(str, command))) + "\n")
        process = subprocess.Popen(list(map(str, command)), cwd=PROJECT_DIR, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, **options)
        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        started, finished = time.monotonic(), False
        try:
            while not finished or process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise RuntimeError(f"{label} 시간이 {timeout}초를 초과했습니다. 기록: {log_path}")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    finished = True
                    continue
                print(line, end="", flush=True)
                log.write(line)
                recent = (recent + [line.strip()])[-60:]
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    with contextlib.suppress(OSError):
                        os.killpg(process.pid, signal.SIGTERM)
                try:
                    # A re-entry worker must be allowed to clean up its own
                    # PyInstaller process group before the parent force-kills it.
                    process.wait(timeout=15 if "--_build-venv" in command else 5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        with contextlib.suppress(OSError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
        finally:
            process.stdout.close()
        log.write(f"Exit code: {code}\n")
    if code and check:
        errors = [line for line in recent if re.search(r"(?i)(error|exception|failed|실패|미설치|필요)", line)]
        cause = " / ".join((errors or [line for line in recent if line])[-3:])[-900:]
        raise RuntimeError(f"{label} 실패 (종료 코드 {code}). {cause}\n기록: {log_path}")
    return code


def _bootstrap_pip_wheel(log_path: Path) -> Path:
    # Prefer Python's own offline wheel; no host pip import or installation.
    spec = importlib.util.find_spec("ensurepip")
    if spec and spec.submodule_search_locations:
        wheels = sorted((Path(next(iter(spec.submodule_search_locations))) / "_bundled").glob("pip-*.whl"), reverse=True)
        if wheels:
            return wheels[0]
    # Some distributors omit ensurepip. Download only the pinned pure-Python pip
    # wheel and verify PyPI's SHA-256 before executing it in the private venv.
    tooling = PROJECT_DIR / "build" / "tooling"
    tooling.mkdir(parents=True, exist_ok=True)
    url = f"https://pypi.org/pypi/pip/{BOOTSTRAP_PIP_VERSION}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.loads(response.read(4 * 1024 * 1024))
    info = next(item for item in metadata["urls"] if item["filename"].endswith("-py3-none-any.whl"))
    parsed = urllib.parse.urlsplit(info["url"])
    digest = info["digests"]["sha256"]
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org" or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise RuntimeError("pip 준비 파일의 출처 또는 검증 정보가 올바르지 않습니다.")
    target = tooling / Path(info["filename"]).name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        return target
    print("설치 도구를 준비합니다.", flush=True)
    with urllib.request.urlopen(info["url"], timeout=30) as response:
        payload = response.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError("pip 준비 파일의 무결성 검사에 실패했습니다.")
    temporary = target.with_suffix(".part")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def verify_prepared_build_environment(environment: Path, requirements, log_path: Path) -> None:
    """Check newly installed versions in the exact interpreter used for builds."""
    report = PROJECT_DIR / "build" / "unmarker-generated" / "build-packages.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    probe = '''\
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
try:
    from packaging.specifiers import SpecifierSet
except ImportError:
    from pip._vendor.packaging.specifiers import SpecifierSet
expected, constraints, report_path = sys.argv[1:]
checks = {}
for name, constraint in json.loads(constraints).items():
    try:
        package = metadata.distribution(name)
        version = package.version
        checks[name] = {
            "ok": version in SpecifierSet(constraint), "required": constraint,
            "version": version, "location": str(package.locate_file("")),
            "metadata_path": str(getattr(package, "_path", package.locate_file(""))),
        }
    except Exception as exc:
        checks[name] = {"ok": False, "required": constraint, "error": f"{type(exc).__name__}: {exc}"}
isolated = bool(sys.flags.isolated) and sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == Path(expected).resolve()
result = {"ready": isolated and all(item["ok"] for item in checks.values()),
          "python": sys.executable, "prefix": sys.prefix, "isolated": isolated, "checks": checks}
Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"설치 확인 Python: {sys.executable}", flush=True)
for name, item in checks.items():
    if name.lower() == "tkinterdnd2" or not item["ok"]:
        state = "확인" if item["ok"] else "불일치"
        print(f"{state}: {name} {item.get('version', '미설치')} (요구: {item['required']})", flush=True)
print(f"구성요소 설치 확인: {sum(item['ok'] for item in checks.values())}/{len(checks)}", flush=True)
sys.exit(0 if result["ready"] else 1)
'''
    command = [str(build_python(environment)), "-I", "-B", "-c", probe, str(environment),
               json.dumps(requirement_constraints(requirements)), str(report)]
    code = run_logged(command, log_path, clean_build_environment(environment), label="설치 버전 확인", timeout=60, check=False)
    try:
        result = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("설치 확인 결과가 사전 형식이 아닙니다.")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"설치한 구성요소의 확인 결과를 읽지 못했습니다. 기록: {log_path}") from exc
    if code or result.get("ready") is not True:
        failed = [f"{name}{item['required']} 필요 (설치됨: {item.get('version', '미설치')})"
                  for name, item in result.get("checks", {}).items() if not item.get("ok")]
        reason = " / ".join(failed[:4]) or "빌드용 Python의 격리 상태를 확인하지 못했습니다."
        raise RuntimeError(f"구성요소 설치 후 확인 실패: {reason}\n상세 정보: {report}\n기록: {log_path}")


def prepare_build_environment(environment: Path, log_path: Path, requirements=None) -> None:
    requirements = tuple(requirements) if requirements is not None else source_build_requirements(PROJECT_DIR / "main.py")
    requirement_constraints(requirements)
    if not environment.resolve().is_relative_to((PROJECT_DIR / "build").resolve()) or environment.is_symlink():
        raise RuntimeError("빌드 환경은 프로젝트 build 폴더 안의 독립된 디렉터리여야 합니다.")
    child = clean_build_environment(environment)
    if not _environment_probe(environment):
        print(f"프로젝트 전용 Python 환경을 준비합니다: {environment}", flush=True)
        run_logged([sys.executable, "-I", "-m", "venv", "--without-pip", str(environment)], log_path, child, label="Python 환경 준비", timeout=120)
    if not _environment_probe(environment):
        raise RuntimeError("프로젝트 전용 가상환경의 격리 상태를 확인하지 못했습니다.")
    python = str(build_python(environment))
    run_logged([python, "-I", "-B", "-c", "import tkinter,_tkinter; print('Tcl/Tk import: OK')"],
               log_path, child, label="Python Tcl/Tk 확인", timeout=30)
    # Execute the bootstrap wheel with the verified venv interpreter. This also
    # repairs an existing venv whose pip metadata exists but pip cannot import.
    wheel = _bootstrap_pip_wheel(log_path)
    bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
    common = ["--isolated", "--disable-pip-version-check", "--no-input", "install", "--only-binary=:all:",
              "--retries", "2", "--timeout", "30", "--index-url", "https://pypi.org/simple",
              "--cache-dir", str(PROJECT_DIR / "build" / "wheel-cache"), "--progress-bar", "off"]
    print("빌드에 필요한 구성요소를 설치합니다. 처음에는 다운로드에 시간이 걸릴 수 있습니다.", flush=True)
    run_logged([python, "-I", "-c", bootstrap, str(wheel), *common, "--upgrade", "--force-reinstall", f"pip=={BOOTSTRAP_PIP_VERSION}"],
               log_path, child, label="pip 준비", timeout=600)
    requirements_file = PROJECT_DIR / "build" / "build-requirements.txt"
    requirements_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    run_logged([python, "-I", "-m", "pip", *common, "--requirement", str(requirements_file)], log_path, child, label="빌드 구성요소 설치")
    run_logged([python, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"], log_path, child, label="의존성 확인", timeout=120)
    verify_prepared_build_environment(environment, requirements, log_path)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Unmarker Pro native desktop build")
    parser.add_argument("--name", default=APP_NAME)
    parser.add_argument("--main", type=Path, default=BUILD_SCRIPT_DIR / "main.py", help="Python entry point; its folder owns fonts/, icons/, build/ and dist/")
    parser.add_argument("--console", action="store_true", help="Keep the console for diagnosing startup errors")
    parser.add_argument("--icon", type=Path, help="Optional .png/.ico/.icns application icon")
    parser.add_argument("--font", type=Path, action="append", default=[], help="Optional .ttf/.otf font; repeatable")
    parser.add_argument("--clean", action="store_true", help="Clear PyInstaller's own build cache")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing or building")
    parser.add_argument("--check", action="store_true", help="Check dependencies without building")
    parser.add_argument("--_build-venv", action="store_true", help=argparse.SUPPRESS)
    return parser


def check_dependencies(requirements=None) -> None:
    required = requirement_constraints(requirements if requirements is not None else source_build_requirements(PROJECT_DIR / "main.py"))
    importlib.invalidate_caches()
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Python 3.11~3.14가 필요합니다. 호환성을 위해 3.11 또는 3.12를 권장합니다.")
    if not importlib.util.find_spec("tkinter") or not importlib.util.find_spec("_tkinter"):
        raise RuntimeError("현재 Python에 Tcl/Tk가 없습니다. Tkinter를 포함한 Python 환경에서 빌드해 주세요.")
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:
        raise RuntimeError("빌드 구성요소가 준비되지 않았습니다. python3 build.py를 실행하면 자동으로 준비합니다.") from exc
    problems = []
    for distribution, constraint in required.items():
        try:
            version = importlib.metadata.version(distribution)
            if version not in SpecifierSet(constraint):
                problems.append(f"{distribution}{constraint} 필요 (설치됨: {version})")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{distribution}{constraint} 미설치")
    # The visible extra owns the OpenCV version choice. Avoid conflicting wheels.
    opencv = []
    for distribution in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            importlib.metadata.version(distribution)
            opencv.append(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    if not opencv or not importlib.util.find_spec("cv2"):
        problems.append("cv2 미설치: remove-ai-watermarks[visible]==0.46.0 설치를 확인해 주세요.")
    elif len(opencv) > 1:
        problems.append("OpenCV 배포판이 여러 개 설치되어 있습니다: " + ", ".join(opencv) + ". 별도 가상환경에 requirements.txt만 설치해 주세요.")
    if problems:
        raise RuntimeError("빌드 의존성을 확인해 주세요:\n- " + "\n- ".join(problems))
    # This checks wheel dependency compatibility (including the torch/vision pair)
    # without importing the application or any model library.
    subprocess.run([sys.executable, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"],
                   env=clean_build_environment(build_environment_dir()), check=True)


def check_native_files(requirements, generated: Path) -> dict:
    """Check wheel-recorded native file sizes without loading shared libraries.

    A successful pip/metadata check does not detect a truncated .so/.dylib.
    Reading each file's stat is cheap, even for Torch; do not hash gigabytes of
    unchanged libraries on every build. The actual API probe still follows.
    """
    checked, broken = 0, []
    for name in requirement_constraints(requirements):
        package = importlib.metadata.distribution(name)
        problems = []
        for entry in package.files or ():
            filename = str(entry).lower()
            if not (filename.endswith((".so", ".dylib", ".dll", ".pyd")) or ".so." in filename):
                continue
            if entry.size is None:
                continue
            checked += 1
            path = Path(package.locate_file(entry))
            try:
                actual = path.stat().st_size
            except OSError as exc:
                problems.append({"path": str(path), "expected_bytes": entry.size,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            if actual != entry.size:
                problems.append({"path": str(path), "expected_bytes": entry.size, "actual_bytes": actual})
        if problems:
            broken.append({"package": name, "version": package.version, "files": problems})
    result = {"ready": not broken, "checked_files": checked, "broken": broken}
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "native-files.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def repair_native_files(requirements, generated: Path, *, allow_repair: bool) -> None:
    """Repair only demonstrably damaged wheels, once, before native execution."""
    report = check_native_files(requirements, generated)
    if report["ready"]:
        print(f"네이티브 라이브러리 파일 확인: {report['checked_files']}개", flush=True)
        return
    damaged = ", ".join(item["package"] for item in report["broken"])
    if not allow_repair or not is_build_environment(build_environment_dir()):
        raise RuntimeError(f"설치된 라이브러리 파일이 손상되었습니다: {damaged}\n상세 기록: {generated / 'native-files.json'}")
    (generated / "native-files-before-repair.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"손상된 라이브러리를 다시 설치합니다: {damaged}", flush=True)
    log_path = new_build_log("repair-native")
    child = clean_build_environment(build_environment_dir())
    for item in report["broken"]:
        # Retain the selected version and CPU/CUDA wheel family. A forced
        # dependency re-resolution could silently replace a CPU Torch build.
        name, version = item["package"], item["version"]
        index = "https://pypi.org/simple"
        variant = re.search(r"\+(cpu|cu[0-9]+)$", version)
        if re.sub(r"[-_.]+", "-", name).lower() in {"torch", "torchvision"} and variant:
            index = "https://download.pytorch.org/whl/" + variant.group(1)
        command = [sys.executable, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
                   "install", "--only-binary=:all:", "--force-reinstall", "--no-deps", "--no-cache-dir",
                   "--retries", "2", "--timeout", "30", "--index-url", index, "--progress-bar", "off", f"{name}=={version}"]
        run_logged(command, log_path, child, label=f"{name} 파일 복구")
    repaired = check_native_files(requirements, generated)
    if not repaired["ready"]:
        raise RuntimeError(f"라이브러리 파일 복구 후에도 크기가 일치하지 않습니다. 반복 설치를 중단합니다.\n상세 기록: {generated / 'native-files.json'}\n설치 기록: {log_path}")
    subprocess.run([sys.executable, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"],
                   env=child, check=True)
    print("라이브러리 파일을 복구했습니다. 실제 실행을 확인합니다.", flush=True)


def app_icon_candidates(resource_dir: Path) -> list[Path]:
    preferred = ("icon.icns", "icon.png", "icon.ico") if sys.platform == "darwin" else ("icon.ico", "icon.png", "icon.icns")
    # User-supplied project icons always take precedence over the bundled
    # Phosphor fallback, regardless of the fallback's preferred file format.
    directories = (resource_dir / "icons", resource_dir, resource_dir / "assets" / "icons")
    return [directory / name for directory in directories for name in preferred]


def asset_plan(args) -> tuple[Path | None, list[tuple[str, str]]]:
    # The entry point's resource loader reads beside main.py. Keep that layout
    # when --main selects another project instead of borrowing this builder's assets.
    resource_dir = args.main.expanduser().resolve().parent
    icon = args.icon.expanduser().resolve() if args.icon else None
    if icon is None:
        icon = next((path for path in app_icon_candidates(resource_dir) if path.is_file()), None)
    if icon and (not icon.is_file() or icon.suffix.lower() not in {".png", ".ico", ".icns"}):
        raise RuntimeError("아이콘 파일은 실제 존재하는 .png, .ico 또는 .icns 파일이어야 합니다.")
    # A project update may retain the older assets/ folder. Do not package an
    # obsolete fallback icon or duplicate font beside the user's originals.
    user_icon_present = any(path.is_file() for path in app_icon_candidates(resource_dir)
                            if path.parent in (resource_dir, resource_dir / "icons"))
    user_font_names = {path.name.casefold() for path in (resource_dir / "fonts").rglob("*")
                       if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}}
    user_font_names.update(Path(path).name.casefold() for path in args.font)
    assets = []
    for folder, extensions in (("fonts", {".ttf", ".otf"}),
                               ("assets/fonts", {".ttf", ".otf", ".txt", ".md", ".json"}),
                               ("icons", {".png", ".ico", ".icns"}),
                               ("assets/icons", {".png", ".ico", ".icns"})):
        if folder == "assets/icons" and user_icon_present:
            continue
        root = resource_dir / folder
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    if (folder == "assets/fonts" and path.suffix.lower() in {".ttf", ".otf"}
                            and path.name.casefold() in user_font_names):
                        continue
                    destination = str(Path(folder) / path.parent.relative_to(root))
                    assets.append((str(path.resolve()), destination))
    # Older project copies also keep icon.* beside main.py. Preserve those
    # files in the canonical runtime icons/ directory without inventing art.
    for path in app_icon_candidates(resource_dir):
        if path.parent == resource_dir and path.is_file():
            destination = (str(path.resolve()), "icons")
            if not any(Path(src).name == path.name and dest == "icons" for src, dest in assets):
                assets.append(destination)
    for path in args.font:
        path = path.expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".ttf", ".otf"}:
            raise RuntimeError(f"폰트 파일을 확인해 주세요: {path}")
        assets.append((str(path), "fonts"))
    return icon, assets


def tk_runtime_binaries() -> list[tuple[str, str]]:
    """Resolve Tcl/Tk shipped beside a relocatable Linux Python.

    The standard PyInstaller hook handles macOS frameworks and Windows DLLs.
    On Linux, an interpreter can resolve its own lib/ through RUNPATH while
    dependency analysis does not search that directory. Include only the
    Tcl/Tk versions actually used by this Python; the frozen app must run
    without an LD_LIBRARY_PATH supplied by its build machine.
    """
    if not sys.platform.startswith("linux"):
        return []
    import _tkinter
    import tkinter as tk
    library = Path(tk.Tcl().eval("info library"))
    directories = [library.parent, Path(sys.base_prefix) / "lib"]
    tcl, tk_version = str(_tkinter.TCL_VERSION), str(_tkinter.TK_VERSION)
    patterns = [f"libtcl{tcl}.so*", f"libtk{tk_version}.so*",
                f"libtcl{tcl.split('.')[0]}tk{tk_version}.so*"]
    found = {}
    for directory in directories:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.is_file():
                    found.setdefault(path.name, (str(path.absolute()), "."))
    return list(found.values())


def build_worker_arguments(args) -> list[str]:
    """Retain the caller's chosen files when the child changes working folder."""
    arguments = ["--main", str(args.main), "--name", args.name]
    for attribute, flag in (("console", "--console"), ("clean", "--clean"), ("check", "--check")):
        if getattr(args, attribute):
            arguments.append(flag)
    if args.icon:
        arguments.extend(("--icon", str(args.icon)))
    for font in args.font:
        arguments.extend(("--font", str(font)))
    return arguments


SPEC_BODY = r'''
import fnmatch
import json
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

MODEL_SUFFIXES = set(CONFIG["weight_suffixes"])
CACHE_PARTS = set(CONFIG["cache_parts"])
NON_RUNTIME_ASSETS = CONFIG["non_runtime_assets"]
def exclusion_reason(name):
    path = Path(str(name).replace("\\", "/"))
    if path.suffix.lower() in MODEL_SUFFIXES:
        return "model_weight"
    if any(part.lower() in CACHE_PARTS or part.lower().startswith("models--") for part in path.parts):
        return "model_or_python_cache"
    normalized = "/" + path.as_posix().lower().lstrip("/")
    for pattern, reason in NON_RUNTIME_ASSETS:
        if fnmatch.fnmatchcase(normalized, "*/" + pattern):
            return reason
    return None

excludes = ["**/__pycache__", "**/tests", "**/.cache", "**/snapshots", "**/blobs", "**/checkpoints"]
for extension in MODEL_SUFFIXES:
    excludes += ["*" + extension, "**/*" + extension]
datas = list(CONFIG["assets"])
binaries = list(CONFIG.get("extra_binaries", []))
hiddenimports = list(CONFIG["hiddenimports"])
# Pip's vendored dependencies and command modules are imported dynamically by
# the dedicated installer worker; collect all of them, not only pip.__init__.
hiddenimports += collect_submodules("pip", on_error="warn")

# Preserve lazy import source discovery and package data/templates without
# enumerating/importing every unrelated model architecture via collect_all().
for package in ("customtkinter", "tkinterdnd2", "diffusers", "transformers", "peft", "gguf",
                "huggingface_hub", "accelerate", "safetensors", "tokenizers",
                "remove_ai_watermarks", "c2pa", "pip", "certifi"):
    datas += collect_data_files(package, include_py_files=True, excludes=excludes)

# Standard torch/numpy/PIL/cv2 hooks remain active. The local torchvision hook
# preserves the installed wheel's actual native-library names.
# Native tkdnd shared libraries must retain their package-relative directories.
binaries += collect_dynamic_libs("tkinterdnd2", search_patterns=["*.dll", "*.dylib", "*.so", "*.so.*"])
# c2pa.lib builds its ctypes library path dynamically; static import analysis
# cannot reliably discover libc2pa_c.so/.dylib or c2pa_c.dll. Preserve c2pa/libs
# and its platform subdirectories, which the SDK resolves relative to __file__.
binaries += collect_dynamic_libs("c2pa", search_patterns=["*.dll", "*.dylib", "*.so", "*.so.*"])

# Package versions are consulted by the pinned detector and runtime checks.
for package in CONFIG["metadata"]:
    datas += copy_metadata(package, recursive=True)
for package in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
    try:
        distribution(package)
    except PackageNotFoundError:
        continue
    datas += copy_metadata(package, recursive=True)

a = Analysis(
    [CONFIG["main"]], pathex=[CONFIG["project"]],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[CONFIG["hooks"]], hooksconfig={}, runtime_hooks=CONFIG["runtime_hooks"],
    excludes=["tensorflow", "tensorflow_probability", "jax", "jaxlib", "flax", "keras",
              "paddle", "onnxruntime", "bitsandbytes", "xformers",
              "pytest", "IPython", "notebook", "jupyterlab"],
    noarchive=False,
)

# Defense in depth: apply the same filter after third-party hooks have run.
# Shared runtime libraries (.so/.dylib/.dll) and CUDA .ptx/.fatbin are not filtered.
excluded_entries = []
def retain_runtime(entries):
    retained = []
    for entry in entries:
        reason = exclusion_reason(entry[0])
        if reason is None:
            retained.append(entry)
            continue
        source = Path(entry[1])
        try:
            byte_count = source.stat().st_size if source.is_file() else 0
        except OSError:
            byte_count = 0
        excluded_entries.append({"destination": entry[0], "source": str(source.absolute()),
                                 "kind": entry[2], "bytes": byte_count, "reason": reason})
    return retained
a.datas = retain_runtime(a.datas)
a.binaries = retain_runtime(a.binaries)
unique_sources = {entry["source"]: entry["bytes"] for entry in excluded_entries}
exclusion_report = {"entries": excluded_entries, "entry_count": len(excluded_entries),
                    "unique_source_count": len(unique_sources),
                    "unique_source_bytes": sum(unique_sources.values()),
                    "note": "Collected source bytes excluded before packaging; not a measured bundle-size difference."}
Path(CONFIG["exclusion_report"]).write_text(json.dumps(exclusion_report, ensure_ascii=False, indent=2), encoding="utf-8")
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name=CONFIG["name"], debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=CONFIG["console"],
    disable_windowed_traceback=False, contents_directory="_internal",
    icon=CONFIG["icon"],
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=CONFIG["name"])
if CONFIG["mac_app"]:
    app = BUNDLE(coll, name=CONFIG["name"] + ".app", icon=CONFIG["icon"],
                 bundle_identifier="com.local.unmark",
                 info_plist={"CFBundleDisplayName": CONFIG["name"],
                             "CFBundleName": CONFIG["name"],
                             "CFBundleShortVersionString": CONFIG["version"],
                             "CFBundleVersion": CONFIG["version"],
                             "NSHighResolutionCapable": True})
'''


def audit_bundle(target: Path) -> None:
    findings = []
    for file in target.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(target)
        reason = bundle_exclusion_reason(relative)
        if reason:
            findings.append(str(relative) + " (" + reason + ")")
    if findings:
        raise RuntimeError("배포 파일에 모델·캐시 또는 실행에 쓰지 않는 빌드 파일이 있습니다. 배포하지 마세요:\n" + "\n".join(findings[:30]))


def verify_mac_bundle_identity(target: Path, name: str, expected_icon: Path | None, expected_version: str = APP_VERSION) -> None:
    """Confirm the app name and supplied icon survived macOS bundling."""
    info_path = target / "Contents" / "Info.plist"
    try:
        with info_path.open("rb") as stream:
            info = plistlib.load(stream)
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        raise RuntimeError(f"앱의 표시 정보를 읽지 못했습니다: {info_path}") from exc
    if not isinstance(info, dict):
        raise RuntimeError(f"앱의 표시 정보가 사전 형식이 아닙니다: {info_path}")
    for key in ("CFBundleName", "CFBundleDisplayName"):
        if info.get(key) != name:
            raise RuntimeError(f"앱 이름이 빌드 설정과 다릅니다: {key}={info.get(key)!r}, 예상={name!r}")
    for key in ("CFBundleShortVersionString", "CFBundleVersion"):
        if info.get(key) != expected_version:
            raise RuntimeError(f"앱 버전이 빌드 설정과 다릅니다: {key}={info.get(key)!r}, 예상={expected_version!r}")
    if expected_icon:
        filename = info.get("CFBundleIconFile")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise RuntimeError(f"앱 아이콘의 번들 참조가 올바르지 않습니다: {info_path}")
        if not Path(filename).suffix:
            filename += ".icns"
        resource = target / "Contents" / "Resources" / filename
        if not resource.is_file() or resource.stat().st_size == 0:
            raise RuntimeError(f"앱에 원본 아이콘이 포함되지 않았습니다: {resource}\n선택한 원본: {expected_icon}")


def _built_probe_file_tail(path: Path, limit: int = 65536) -> str:
    """Keep complete logs on disk, but bound the copy printed to the build log."""
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - limit))
            text = stream.read(limit).decode("utf-8", errors="replace")
        prefix = f"[전체 {size:,}바이트 중 마지막 {limit:,}바이트; 전체 기록: {path}]\n" if size > limit else ""
        return prefix + (text or "(기록이 비어 있습니다.)")
    except OSError as exc:
        return f"(기록을 읽지 못했습니다: {type(exc).__name__}: {exc})"


def _built_probe_failure(reason: str, generated: Path, result=None, *, source: bool = False) -> RuntimeError:
    """Expose file-IPC failures to the terminal and the parent prepare log."""
    prefix = "source-probe" if source else "runtime-probe"
    result_file = generated / (prefix + ".json")
    runtime_log = generated / (prefix + ".log")
    process_log = generated / (prefix + "-process.log")
    diagnostic_log = generated / (prefix + "-diagnostics.log")
    failures = []
    if isinstance(result, dict):
        checks = result.get("checks", {})
        if isinstance(checks, dict):
            for package, check in checks.items():
                if isinstance(check, dict) and check.get("ok") is not True:
                    failures.append(f"{package}: {check.get('error') or '실행 검사 실패 (상세 오류 없음)'}")
        if result.get("error"):
            failures.append(str(result["error"]))
        if result.get("missing"):
            failures.append("실패한 구성요소: " + str(result["missing"]))
    heading = ("빌드 전 실행 확인 실패: " if source else "완성본 실행 확인 실패: ") + reason
    details = [heading]
    if failures:
        details.append("실제 검사 오류:\n" + "\n".join(failures))
    details.extend([
        f"검사 결과 JSON ({result_file}):\n{_built_probe_file_tail(result_file)}",
        f"런타임 상세 기록 ({runtime_log}):\n{_built_probe_file_tail(runtime_log)}",
        f"프로세스 stdout/stderr 기록 ({process_log}):\n{_built_probe_file_tail(process_log)}",
    ])
    diagnostic = "\n\n".join(details) + "\n"
    # The normal build parent captures this output into prepare-*.log. Keep a
    # separate copy as well for direct --_build-venv runs and terminal failures.
    try:
        diagnostic_log.write_text(diagnostic, encoding="utf-8")
    except OSError as exc:
        diagnostic += f"진단 기록 저장 실패: {type(exc).__name__}: {exc}\n"
    print(diagnostic, end="", flush=True)
    summary = " / ".join(failures)
    if len(summary) > 2000:
        summary = summary[:2000] + "… (위 상세 기록 참조)"
    return RuntimeError(heading + ("\n" + summary if summary else "")
                        + f"\n진단 기록: {diagnostic_log}\n전체 런타임 기록: {runtime_log}\n프로세스 기록: {process_log}")


def verify_built_runtime(target: Path, name: str, generated: Path, mac_app: bool, *, timeout: float = 300) -> None:
    """Validate file IPC and preserve both runtime and bootloader diagnostics."""
    if mac_app:
        executable = target / "Contents" / "MacOS" / name
    else:
        executable = target / (name + ".exe" if sys.platform == "win32" else name)
    result_file = generated / "runtime-probe.json"
    log_file = generated / "runtime-probe.log"
    process_log = generated / "runtime-probe-process.log"
    generated.mkdir(parents=True, exist_ok=True)
    for path in (result_file, log_file, process_log, generated / "runtime-probe-diagnostics.log"):
        path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS": "1"})
    command = [str(executable), "--unmark-runtime-probe", "--ignore-overlay",
               "--result-file", str(result_file), "--log-file", str(log_file)]
    try:
        # Windowed workers report Python exceptions through --log-file. Native
        # loader/early bootstrap failures may happen before that file is opened;
        # capture their stdout/stderr separately instead of discarding them.
        code = run_logged(command, process_log, environment, label="완성본 실행 확인",
                          timeout=timeout, check=False)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise _built_probe_failure(f"프로세스 실행 오류: {type(exc).__name__}: {exc}", generated) from exc
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise _built_probe_failure(f"종료 코드 {code}; 결과 JSON을 읽지 못했습니다: {type(exc).__name__}: {exc}", generated) from exc
    if not isinstance(result, dict) or type(result.get("ready")) is not bool:
        raise _built_probe_failure(f"종료 코드 {code}; 잘못된 결과 JSON 형식 (객체와 boolean ready 필요)", generated, result)
    checks, missing = result.get("checks", {}), result.get("missing", [])
    if (not isinstance(checks, dict)
            or any(not isinstance(check, dict) or type(check.get("ok")) is not bool for check in checks.values())
            or not isinstance(missing, list) or any(not isinstance(name, str) for name in missing)):
        raise _built_probe_failure(f"종료 코드 {code}; 잘못된 checks/missing 결과 형식", generated, result)
    if code != 0 or result["ready"] is not True or missing or not checks or any(check["ok"] is not True for check in checks.values()):
        raise _built_probe_failure(f"종료 코드 {code}; ready={result['ready']}; 검사 항목 {len(checks)}개", generated, result)


def verify_source_runtime(source: Path, generated: Path) -> None:
    """Exercise main.py's real import/API checks before expensive packaging."""
    generated.mkdir(parents=True, exist_ok=True)
    result_file, log_file = generated / "source-probe.json", generated / "source-probe.log"
    for path in (result_file, log_file, generated / "source-probe-process.log", generated / "source-probe-diagnostics.log"):
        path.unlink(missing_ok=True)
    environment = clean_build_environment(build_environment_dir())
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS": "1"})
    # Native imports may terminate the interpreter before it can write JSON.
    # Faulthandler retains the failing module in the captured process log.
    command = [sys.executable, "-I", "-B", "-X", "faulthandler", str(source), "--unmark-runtime-probe", "--ignore-overlay",
               "--result-file", str(result_file), "--log-file", str(log_file)]
    print("설치한 구성요소의 실제 실행을 확인합니다.", flush=True)
    try:
        code = run_logged(command, generated / "source-probe-process.log", environment, label="빌드 전 실행 확인", timeout=300, check=False)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise _built_probe_failure(f"프로세스 실행 오류: {type(exc).__name__}: {exc}", generated, source=True) from exc
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise _built_probe_failure(f"종료 코드 {code}; 결과 JSON을 읽지 못했습니다: {type(exc).__name__}: {exc}", generated, source=True) from exc
    if not isinstance(result, dict) or type(result.get("ready")) is not bool:
        raise _built_probe_failure(f"종료 코드 {code}; 잘못된 결과 JSON 형식 (객체와 boolean ready 필요)", generated, result, source=True)
    checks, missing = result.get("checks", {}), result.get("missing", [])
    if (not isinstance(checks, dict)
            or any(not isinstance(check, dict) or type(check.get("ok")) is not bool for check in checks.values())
            or not isinstance(missing, list) or any(not isinstance(name, str) for name in missing)):
        raise _built_probe_failure(f"종료 코드 {code}; 잘못된 checks/missing 결과 형식", generated, result, source=True)
    if code != 0 or result["ready"] is not True or missing or not checks or any(check["ok"] is not True for check in checks.values()):
        raise _built_probe_failure(f"종료 코드 {code}; ready={result['ready']}; 검사 항목 {len(checks)}개", generated, result, source=True)


def verify_engine_runtime(command_prefix: list[str], generated: Path, identity: dict[str, str], *, frozen: bool) -> dict:
    """Run CPU model-API checks and compare the embedded release/source identity."""
    label = "완성본 이미지 엔진 확인" if frozen else "빌드 전 이미지 엔진 확인"
    stem = "frozen-engine-probe" if frozen else "source-engine-probe"
    generated.mkdir(parents=True, exist_ok=True)
    result_file = generated / (stem + ".json")
    process_log = generated / (stem + ".log")
    for path in (result_file, process_log):
        path.unlink(missing_ok=True)
    environment = clean_build_environment(build_environment_dir())
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    command = [*command_prefix, "--engine-probe", "--ignore-overlay", "--result-file", str(result_file)]
    try:
        code = run_logged(command, process_log, environment, label=label, timeout=120, check=False)
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if code:
            detail = result.get("error", "") if isinstance(result, dict) else ""
            raise RuntimeError(f"종료 코드 {code}. {detail}")
        verify_engine_result(result, identity)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        tail = _built_probe_file_tail(process_log, limit=12000)
        raise RuntimeError(f"{label} 실패: {exc}\n결과: {result_file}\n기록: {process_log}\n{tail}") from exc
    return result


def _execute_build(args) -> int:
    try:
        if not args.name.strip() or any(c in args.name for c in '/\\:\0') or args.name in {".", ".."}:
            raise RuntimeError("앱 이름에 경로 구분자나 빈 이름을 사용할 수 없습니다.")
        source = args.main.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".py":
            raise RuntimeError(f"실행할 Python 파일을 찾을 수 없습니다: {source}")
        # Recheck after installation/reentry, before any expensive packaging.
        identity = source_build_identity(source)
        requirements = source_build_requirements(source)
        required = requirement_constraints(requirements)
        icon, assets = asset_plan(args)
        generated = PROJECT_DIR / "build" / "unmarker-generated"
        hooks = generated / "hooks"
        config = {
            "name": args.name, "version": identity["app_version"],
            "main": str(source), "project": str(source.parent),
            "icon": str(icon) if icon else None, "assets": assets,
            "console": bool(args.console), "mac_app": sys.platform == "darwin" and not args.console,
            "hooks": str(hooks), "hiddenimports": HIDDEN_IMPORTS,
            "weight_suffixes": sorted(WEIGHT_SUFFIXES), "cache_parts": sorted(CACHE_PARTS),
            "non_runtime_assets": NON_RUNTIME_ASSETS,
            "exclusion_report": str(generated / "packaging-exclusions.json"),
            "metadata": list(dict.fromkeys([name for name in required if name.lower() != "pyinstaller"] + ["tokenizers"])),
            "runtime_hooks": [str(hooks / "runtime-build-identity.py")],
            "extra_binaries": [],
        }
        target = PROJECT_DIR / "dist" / (args.name + ".app" if config["mac_app"] else args.name)
        spec_path = generated / "unmarker.spec"
        layout_marker = generated / "completed-build-layout.txt"
        builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        layout_identity = "\n".join((PACKAGING_LAYOUT_VERSION, str(build_environment_dir()),
                                      identity["source_sha256"], builder_sha256))
        try:
            refresh_packaging_cache = layout_marker.read_text(encoding="utf-8") != layout_identity
        except OSError:
            refresh_packaging_cache = True
        command = [sys.executable, "-I", "-B", "-m", "PyInstaller", "--noconfirm", "--distpath", str(PROJECT_DIR / "dist"), "--workpath", str(PROJECT_DIR / "build" / "pyinstaller")]
        if args.clean or refresh_packaging_cache:
            command.append("--clean")
        command.append(str(spec_path))
        print(f"대상: {platform.system()} / {platform.machine()} / Python {platform.python_version()}")
        print(f"구성: onedir / {'console' if args.console else 'windowed'} / {args.name}")
        print(f"빌드 도구: {BUILD_REVISION} / {Path(__file__).resolve()}")
        print(f"실행 코드: {source}")
        print(f"앱 버전: {identity['app_version']} · 실행 엔진: {identity['engine_id']}")
        print(f"소스 SHA-256: {identity['source_sha256']}")
        print(f"빌드 Python: {sys.executable}")
        print(f"프로젝트 폴더: {PROJECT_DIR}")
        print(f"완성본 위치: {target}")
        if icon:
            print(f"앱 아이콘: {icon}")
        else:
            print("앱 아이콘 파일이 없습니다. 원래 icon.icns/icon.png/icon.ico를 icons/ 또는 main.py 옆에 두면 자동으로 포함합니다.")
        font_count = sum(Path(path).suffix.lower() in {".ttf", ".otf"} for path, _dest in assets)
        print(f"포함할 폰트 파일: {font_count}개 · {source.parent / 'fonts'}")
        print("기본 처리: Qwen GGUF · Lightning · 디테일 복원을 로컬에서 실행합니다.")
        print("실행 라이브러리는 포함하며 모델 가중치와 캐시는 포함하지 않습니다.")
        if args.dry_run:
            # Also validate the generated spec without importing PyInstaller.
            compile("CONFIG = " + repr(config) + "\n" + SPEC_BODY, str(spec_path), "exec")
            print("실행 예정:", subprocess.list2cmdline(command))
            print(f"전용 Python 환경: {build_environment_dir()}")
            print(f"기본 실행 시 자동 준비할 구성요소: {len(requirements)}개 (main.py 기준, requirements.txt 불필요).")
            print(f"선택 에셋: {len(assets)}개. 의존성 검사·파일 생성·빌드는 실행하지 않았습니다.")
            return 0
        try:
            check_dependencies(requirements)
        except (RuntimeError, subprocess.SubprocessError):
            # A direct worker invocation or a source update between preparation
            # and reentry must also repair stale packages once, before probing.
            # --check and --dry-run remain strictly read-only.
            if args.check or not args._build_venv:
                raise
            print("실행 코드에 맞춰 빌드 구성요소를 업데이트합니다.", flush=True)
            log = new_build_log("repair")
            print(f"구성요소 업데이트 기록: {log}", flush=True)
            prepare_build_environment(build_environment_dir(), log, requirements)
            check_dependencies(requirements)
        if args.check:
            print("의존성과 main.py 문법을 확인했습니다. 빌드는 실행하지 않았습니다.")
            return 0
        repair_native_files(requirements, generated, allow_repair=args._build_venv)
        verify_source_runtime(source, generated)
        source_engine = verify_engine_runtime(
            [sys.executable, "-I", "-B", "-X", "faulthandler", str(source)], generated, identity, frozen=False)
        if source_build_identity(source) != identity:
            raise RuntimeError("실행 검사 중 main.py가 변경되었습니다. 변경을 마친 뒤 다시 빌드해 주세요.")
        config["extra_binaries"] = tk_runtime_binaries()
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "runtime-build-identity.py").write_text(build_identity_hook(identity), encoding="utf-8")
        # Main's loader expects resource_root/icons/icon.{png,ico}; normalize
        # an explicitly supplied icon filename into that optional resource path.
        if args.icon and icon:
            staged = generated / "icons" / ("icon" + icon.suffix.lower())
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(icon, staged)
            canonical = staged.name.lower()
            config["assets"] = [(path, dest) for path, dest in assets if not (dest == "icons" and Path(path).name.lower() == canonical)]
            config["assets"].append((str(staged), "icons"))
        (generated / "build-assets.json").write_text(json.dumps({
            "app_name": args.name, "project": str(PROJECT_DIR), "output": str(target),
            "build_identity": identity,
            "icon": str(icon) if icon else None,
            "icon_search_paths": [str(path) for path in app_icon_candidates(source.parent)],
            "font_directory": str(source.parent / "fonts"),
            "assets": [{"source": path, "destination": destination} for path, destination in config["assets"]],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (hooks / "hook-tkinterdnd2.py").write_text(textwrap.dedent('''\
            from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
            datas = collect_data_files("tkinterdnd2")
            binaries = collect_dynamic_libs("tkinterdnd2", search_patterns=["*.dll", "*.dylib", "*.so", "*.so.*"])
            hiddenimports = ["tkinterdnd2.TkinterDnD"]
            '''), encoding="utf-8")
        # JIT/lazy framework utilities need original source alongside bytecode.
        # Native dependency rewriting and signing remain handled by the hooks.
        (hooks / "hook-diffusers.py").write_text(
            "module_collection_mode = {name: 'pyz+py' for name in ('diffusers', 'transformers', 'torch', 'torchvision', 'peft', 'gguf')}\n",
            encoding="utf-8",
        )
        (hooks / "hook-torchvision.py").write_text(TORCHVISION_HOOK, encoding="utf-8")
        (hooks / "hook-torchvision.io.image.py").write_text(TORCHVISION_IMAGE_HOOK, encoding="utf-8")
        (hooks / "hook-pip.py").write_text("module_collection_mode = 'pyz+py'\n", encoding="utf-8")
        spec_path.write_text("CONFIG = " + repr(config) + "\n" + SPEC_BODY, encoding="utf-8")
        environment = clean_build_environment(build_environment_dir())
        # Restrict only the build child. These flags are NOT a runtime hook and
        # are not persisted in the packaged app; normal first-run setup remains.
        environment.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(generated / "empty-hf-cache"),
            "TORCH_HOME": str(generated / "empty-torch-cache"),
        })
        if refresh_packaging_cache:
            print("변경된 패키징 규칙을 적용합니다. 기존 설치 환경은 재사용하고 빌드 분석 캐시를 갱신합니다.", flush=True)
        # Packaging replaces files in dist/. A prior successful receipt must
        # not remain authoritative if this replacement fails or is cancelled.
        # Only the fully verified build below writes these markers again.
        (generated / "build-receipt.json").unlink(missing_ok=True)
        layout_marker.unlink(missing_ok=True)
        run_logged(command, new_build_log("package"), environment, label="앱 패키징")
        if not target.exists():
            raise RuntimeError(f"빌드 결과를 찾을 수 없습니다: {target}")
        if config["mac_app"]:
            verify_mac_bundle_identity(target, args.name, icon, identity["app_version"])
        audit_bundle(target)
        # Check the bundled distribution itself; an external overlay must not
        # conceal a missing wheel, native library, template or package metadata.
        verify_built_runtime(target, args.name, generated, config["mac_app"])
        executable = (target / "Contents" / "MacOS" / args.name if config["mac_app"]
                      else target / (args.name + ".exe" if sys.platform == "win32" else args.name))
        frozen_engine = verify_engine_runtime([str(executable)], generated, identity, frozen=True)
        if source_build_identity(source) != identity:
            raise RuntimeError("패키징 중 main.py가 변경되었습니다. 결과를 배포하지 말고 다시 빌드해 주세요.")
        (generated / "build-receipt.json").write_text(json.dumps({
            **identity, "build_revision": BUILD_REVISION, "builder_sha256": builder_sha256,
            "source": str(source), "output": str(target), "platform": platform.system(),
            "architecture": platform.machine(), "app_name": args.name,
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "source_engine_verified": source_engine["ready"],
            "frozen_engine_verified": frozen_engine["ready"],
            "model_preflight": frozen_engine["model_preflight"],
            "packaging_exclusions": str(generated / "packaging-exclusions.json"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        layout_marker.write_text(layout_identity, encoding="utf-8")
        print(f"빌드 완료: {target}")
        print("배포 시 이 폴더 또는 .app 전체를 전달해 주세요. 실행 파일만 따로 옮기면 동작하지 않습니다.")
        return 0
    except (OSError, RuntimeError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"빌드 실패: {exc}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    global PROJECT_DIR
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = argument_parser().parse_args(arguments)
    try:
        if not args.name.strip() or any(c in args.name for c in '/\\:\0') or args.name in {".", ".."}:
            raise RuntimeError("앱 이름에 경로 구분자나 빈 이름을 사용할 수 없습니다.")
        # Validate cheap inputs before preparing any packages. The build repeats
        # this check after installation so changed source files are also caught.
        source = args.main.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".py":
            raise RuntimeError(f"실행할 Python 파일을 찾을 수 없습니다: {source}")
        source_build_identity(source)
        # Interpret relative paths once, in the original caller's cwd. Every
        # subsequent installer, probe and package worker uses absolute paths
        # and the selected application's project directory.
        args.main = source
        args.icon = args.icon.expanduser().resolve() if args.icon else None
        args.font = [font.expanduser().resolve() for font in args.font]
        PROJECT_DIR = source.parent
        environment = build_environment_dir()
        requirements = source_build_requirements(source)
        asset_plan(args)
        if args.dry_run:
            return _execute_build(args)
        if args._build_venv:
            if not is_build_environment(environment) or not sys.flags.isolated:
                raise RuntimeError("빌드 재진입은 프로젝트 전용 가상환경의 격리된 Python에서만 허용됩니다.")
            if args.check:
                return _execute_build(args)
            def cancel_worker(_signum, _frame):
                raise KeyboardInterrupt()
            previous_term = signal.signal(signal.SIGTERM, cancel_worker) if os.name != "nt" else None
            try:
                with build_lock():
                    return _execute_build(args)
            finally:
                if previous_term is not None:
                    signal.signal(signal.SIGTERM, previous_term)
        if sys.implementation.name != "cpython" or not (3, 11) <= sys.version_info[:2] < (3, 15) or sys.maxsize <= 2 ** 32:
            raise RuntimeError("64비트 CPython 3.11~3.14가 필요합니다.")
        reentry = [str(build_python(environment)), "-I", "-B", str(Path(__file__).resolve()),
                   *build_worker_arguments(args), "--_build-venv"]
        if args.check:
            if not _environment_probe(environment):
                raise RuntimeError("준비된 프로젝트 빌드 환경이 없습니다. python3 build.py를 실행하면 자동으로 준비합니다. --check는 설치하지 않습니다.")
            return subprocess.run(reentry, cwd=PROJECT_DIR, env=clean_build_environment(environment), check=False).returncode
        with build_lock():
            log = new_build_log("prepare")
            print(f"빌드 도구: {BUILD_REVISION} / {Path(__file__).resolve()}", flush=True)
            print(f"구성요소 기준: {source}", flush=True)
            print(f"빌드 준비 기록: {log}", flush=True)
            prepare_build_environment(environment, log, requirements)
        # Release the preparation lock before the child acquires it for the
        # complete build. No two installers/builds can mutate this project at once.
        return run_logged(reentry, log, clean_build_environment(environment), label="앱 빌드", timeout=10800)
    except KeyboardInterrupt:
        print("빌드를 취소했습니다.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"빌드 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Local Qwen GGUF image processing with bounded weight residency.

Independent integration of the Qwen img2img + Gaussian detail-restoration
recipe documented by https://github.com/0xROOTPLS/DeSynth at commit
0c2b14fa6b92fff3721877838fcb9de78fdec0ff. That repository did not include a
license file at that revision; its source is referenced, not vendored here.
The model/library licenses and pinned download identities are recorded in the
application's model manifest and notices. Diffusers/PEFT use Apache-2.0 APIs:
https://github.com/huggingface/diffusers/tree/v0.36.0/src/diffusers
https://github.com/huggingface/peft/tree/v0.18.0

This module executes image processing. It contains no SynthID detector and
does not turn successful inference into a watermark-removal verification.
"""

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
    "restore_sigma": 1.95,
}
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


def _qwen_restore_detail(clean, original, sigma, cancel_event=None):
    """Gaussian(clean) + original - Gaussian(original), in bounded row strips."""
    import cv2
    import numpy as np
    from PIL import Image
    source = np.asarray(original.convert("RGB"))
    cleaned = np.asarray(clean.convert("RGB"))
    if clean.size != original.size:
        cleaned = cv2.resize(cleaned, original.size, interpolation=cv2.INTER_LANCZOS4)
    result = np.empty_like(source)
    halo = math.ceil(sigma * 4) + 2
    for top in range(0, original.height, 128):
        _check_cancel(cancel_event)
        bottom = min(original.height, top + 128)
        start, stop = max(0, top - halo), min(original.height, bottom + halo)
        first = source[start:stop].astype(np.float32)
        second = cleaned[start:stop].astype(np.float32)
        low_source = cv2.GaussianBlur(first, (0, 0), sigmaX=sigma, sigmaY=sigma)
        low_clean = cv2.GaussianBlur(second, (0, 0), sigmaX=sigma, sigmaY=sigma)
        band = np.clip(low_clean + (first - low_source), 0, 255)
        if not np.isfinite(band).all():
            raise ModelRuntimeError("이미지 복원 결과에 유효하지 않은 수치가 있습니다.")
        # Match DeSynth's final float -> uint8 conversion. A different rounding
        # rule changes most restored pixels even with identical model output.
        result[top:bottom] = band[top - start:bottom - start].astype(np.uint8)
    return Image.fromarray(result)


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
                  "restore_sigma": float(supplied.get("restore_sigma", 1.95)),
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
        restored = _qwen_restore_detail(cleaned, rgb, settings["restore_sigma"], cancel_event)
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
            "detail_restore": {"performed": settings["restore"], "method": "gaussian_low_clean_high_original" if settings["restore"] else "none",
                               "sigma": settings["restore_sigma"]},
            "mmap": mmap_report, "weights_loaded": True, "pixel_comparison": comparison,
            "preservation_check": _qwen_preservation_check(comparison),
            "output_path": str(target), "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_process_rss_bytes": _qwen_peak_rss_bytes(),
            "synthid": {"status": "not_tested", "removal_verified": False},
            "detail": "이미지 처리 및 원본 디테일 복원 완료" if changed else "Qwen 실행 완료 · 출력 픽셀 변경 없음"}


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

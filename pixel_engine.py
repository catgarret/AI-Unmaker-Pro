"""Fast bounded pixel filtering and descriptive frequency measurements.

The frequency statistics are an uncalibrated texture heuristic, not a SynthID
classifier or a probability of AI origin. This module never imports a neural
network runtime and never turns an executed filter into a verified removal.
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time

from file_publish import publish_new_file


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

"""Decode the public DWT-DCT payloads supported by the pinned visible package.

This is a real payload comparison, separate from descriptive FFT statistics.
Its scope is the three public embedding formats below, not Google's SynthID.
All optional imports are lazy so the first-run bootstrap remains available.
"""
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

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

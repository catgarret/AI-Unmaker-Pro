"""로컬 디테일 복원: 기본 복원, 윤곽별 혼합, 선택적 선명화.

DeSynth의 공개 수식을 독립적으로 구현합니다. 큰 이미지도 RGB 실수 배열을
전체 크기로 여러 개 만들지 않도록 겹치는 행 단위로 처리합니다.
이 모듈은 화질을 조정하며 워터마크 검출·제거 성공을 판정하지 않습니다.
"""
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

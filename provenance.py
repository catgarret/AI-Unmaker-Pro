"""공식 검증 서비스의 명세와 명시적으로 요청한 OpenAI 출처 검사를 담당합니다.

로컬 처리와 독립적인 기능입니다. 검사 결과로 자동 재처리하거나 서비스를
반복 호출하지 않으며, C2PA와 SynthID 결과도 서로 합치지 않습니다.
"""
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

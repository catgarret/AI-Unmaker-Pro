"""파일 내용에 묶인 외부 검사 기록. 네트워크 호출이나 픽셀 판정은 하지 않습니다.

웹에서 확인한 결과는 관찰 기록으로, 앱에서 받은 API 응답은 API 기록으로
표시합니다. JSON은 전자서명 증명이 아니므로 출처와 검사일을 항상 함께 씁니다.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

VERIFICATION_RECORD_LIMIT = 128 * 1024
VERIFICATION_PROVIDERS = {"google": "Google", "openai": "OpenAI"}
VERIFICATION_STATUSES = ("detected", "not_detected", "not_tested", "error")


def verification_store_root():
    override = os.environ.get("UNMARK_VERIFICATION_STORE")
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise ValueError("검사 기록 폴더는 절대 경로여야 합니다.")
        return root
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "AI Unmarker Pro" / "verification-records"


def verification_file_hash(path):
    """이름이 같아도 한 바이트라도 바뀌면 다른 검사 대상으로 취급합니다."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalise_verification_record(record):
    if not isinstance(record, dict):
        raise ValueError("검사 기록 형식이 올바르지 않습니다.")
    digest = record.get("sha256")
    provider, method = record.get("provider"), record.get("method")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("검사한 파일의 SHA-256이 필요합니다.")
    if not isinstance(provider, str) or provider not in VERIFICATION_PROVIDERS or method not in ("official_api", "official_web_observation"):
        raise ValueError("지원하는 공식 검사 기록이 아닙니다.")
    if provider == "google" and method == "official_api":
        raise ValueError("Google 웹 검사 기록만 지원합니다.")
    try:
        checked = datetime.fromisoformat(record["checked_at"])
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        raise ValueError("검사 날짜가 올바르지 않습니다.") from None
    signals = record.get("signals")
    if not isinstance(signals, dict) or not signals or set(signals) - {"synthid", "c2pa"}:
        raise ValueError("SynthID와 C2PA의 개별 검사 결과가 필요합니다.")
    clean = {}
    for name, signal in signals.items():
        status = signal.get("status") if isinstance(signal, dict) else None
        if status not in VERIFICATION_STATUSES:
            raise ValueError("검사 상태를 확인하지 못했습니다.")
        clean[name] = {"status": status, "detected": status == "detected" if status in
                       ("detected", "not_detected") else None}
    return {"schema": "unmarker.verification.v1", "sha256": digest, "provider": provider,
            "method": method, "checked_at": checked.astimezone(timezone.utc).isoformat(),
            "signals": clean, "imported": record.get("imported") is True}


def read_verification_records(path):
    """선택한 JSON만 읽습니다. 기록 안의 URL·경로를 열거나 실행하지 않습니다."""
    with Path(path).open("rb") as stream:
        raw = stream.read(VERIFICATION_RECORD_LIMIT + 1)
    if len(raw) > VERIFICATION_RECORD_LIMIT:
        raise ValueError("검사 기록은 128 KiB 이하여야 합니다.")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("검사 기록 JSON을 읽지 못했습니다.") from None
    rows = payload.get("records", [payload]) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError("검사 기록은 1~64개여야 합니다.")
    return [normalise_verification_record(row) for row in rows]


def save_verification_record(record, *, root=None):
    """제공사별 최신 기록을 원자적으로 저장해 중간 파일을 읽지 않게 합니다."""
    record = normalise_verification_record(record)
    folder = Path(root) if root is not None else verification_store_root()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = folder / (record["sha256"] + "." + record["provider"] + ".json")
    if destination.exists():
        try:
            previous = read_verification_records(destination)[0]
            if previous["checked_at"] > record["checked_at"]:
                return previous
        except (OSError, ValueError):
            pass
    fd, temporary = tempfile.mkstemp(prefix=".verification-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return record


def file_verification_records(path, *, root=None):
    digest = verification_file_hash(path)
    folder = Path(root) if root is not None else verification_store_root()
    records, errors = [], []
    for provider in VERIFICATION_PROVIDERS:
        try:
            rows = read_verification_records(folder / (digest + "." + provider + ".json"))
            if len(rows) != 1 or rows[0]["sha256"] != digest or rows[0]["provider"] != provider:
                raise ValueError("file binding mismatch")
            records.extend(rows)
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            errors.append(provider)
    return {"sha256": digest, "records": records, "record_errors": errors}


def import_file_verification(path, report_path, *, root=None):
    """가져오기는 현재 파일과 해시가 일치하는 기록에만 적용됩니다."""
    digest = verification_file_hash(path)
    rows = [row for row in read_verification_records(report_path) if row["sha256"] == digest]
    if not rows:
        raise ValueError("선택한 이미지와 일치하는 검사 기록이 없습니다.")
    for row in rows:
        save_verification_record(dict(row, imported=True), root=root)
    return file_verification_records(path, root=root)

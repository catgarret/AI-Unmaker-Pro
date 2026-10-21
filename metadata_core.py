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

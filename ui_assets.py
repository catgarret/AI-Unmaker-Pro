import ctypes
import ctypes.util
import os
from pathlib import Path
import struct
import sys

_UA_STATE = None
_UA_LIBRARIES = {}
_UA_ICONS = {}
_UA_GLYPHS = {
    "upload-simple": 0xE4C0,
    "image": 0xE2CA,
    "download-simple": 0xE20C,
    "arrow-counter-clockwise": 0xE038,
    "x": 0xE4F6,
    "check": 0xE182,
    "info": 0xE2CE,
    "magnifying-glass": 0xE30C,
    "eye": 0xE220,
    "check-circle": 0xE184,
    "warning-circle": 0xE4E2,
    "circle-notch": 0xEB44,
}


def ui_resource_roots():
    roots = []
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS))
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        roots.append(executable_dir)
        if sys.platform == "darwin" and executable_dir.name == "MacOS":
            roots.append(executable_dir.parent / "Resources")
    roots.append(Path(__file__).resolve().parent)
    return list(dict.fromkeys(path.resolve() for path in roots))


def ui_app_icon_candidates():
    """Prefer every project icon before the bundled Phosphor fallback."""
    names = ("icon.ico", "icon.png", "icon.icns") if sys.platform == "win32" else ("icon.png", "icon.icns", "icon.ico")
    candidates = []
    for root in ui_resource_roots():
        for directory in (root / "icons", root):
            for name in names:
                candidate = directory / name
                if candidate not in candidates:
                    candidates.append(candidate)
    for root in ui_resource_roots():
        for name in names:
            candidate = root / "assets" / "icons" / name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def ui_apply_app_identity(window, title="AI Unmarker Pro"):
    """Apply the original app name/icon through Tk's public window APIs."""
    import tkinter as tk
    window.title(title)
    window.tk.call("tk", "appname", "AI Unmarker Pro")
    candidates = ui_app_icon_candidates()
    result = {"name": "AI Unmarker Pro", "title": title, "icon_applied": False,
              "icon_path": None, "searched": [str(path) for path in candidates], "errors": []}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if sys.platform == "win32" and path.suffix.lower() == ".ico":
                window.iconbitmap(str(path))
            else:
                if path.suffix.lower() == ".png":
                    photo = tk.PhotoImage(master=window, file=str(path))
                else:
                    # Pillow is already present in the main desktop UI. A
                    # minimal first-run bootstrap can continue to a PNG when
                    # Pillow has not been installed yet.
                    from PIL import Image, ImageTk
                    with Image.open(path) as source:
                        photo = ImageTk.PhotoImage(source.convert("RGBA"), master=window)
                window.iconphoto(True, photo)
                window._ua_app_icon_photo = photo  # keep the Tk image alive
                if sys.platform == "win32":
                    # CTk installs its default Windows icon after 200 ms when
                    # no .ico was used. Reapply the user's PNG through public
                    # Tk methods after that scheduled initialization.
                    def retain_project_icon():
                        try:
                            if window.winfo_exists():
                                window.iconphoto(True, photo)
                        except tk.TclError:
                            pass
                    window.after(250, retain_project_icon)
            result.update(icon_applied=True, icon_path=str(path))
            return result
        except (ImportError, OSError, ValueError, tk.TclError) as exc:
            result["errors"].append(f"{path}: {type(exc).__name__}: {exc}")
    return result


def _ua_font_table(path, tag):
    """Read a bounded SFNT table without depending on Pillow during first setup."""
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        header = stream.read(12)
        if len(header) != 12 or header[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
            return b""
        count = struct.unpack_from(">H", header, 4)[0]
        if count > 256:
            return b""
        records = stream.read(count * 16)
        if len(records) != count * 16:
            return b""
        for position in range(0, len(records), 16):
            table, _checksum, offset, length = struct.unpack_from(">4sIII", records, position)
            if table == tag and length <= 8 * 1024 * 1024 and offset + length <= size:
                stream.seek(offset)
                return stream.read(length)
    return b""


def _ua_font_family(path):
    data = _ua_font_table(path, b"name")
    if len(data) < 6:
        return ""
    _format, count, storage = struct.unpack_from(">HHH", data)
    names = []
    for index in range(min(count, (len(data) - 6) // 12)):
        platform_id, _encoding, language, name_id, length, offset = struct.unpack_from(">HHHHHH", data, 6 + index * 12)
        if name_id not in (1, 16) or storage + offset + length > len(data):
            continue
        encoding = "utf-16-be" if platform_id in (0, 3) else "mac_roman"
        try:
            value = data[storage + offset:storage + offset + length].decode(encoding).strip("\x00 ")
        except UnicodeError:
            continue
        if value:
            priority = (name_id == 16, language in (0, 0x409), platform_id in (0, 3))
            names.append((priority, value))
    return max(names)[1] if names else ""


def _ua_has_glyph(path, codepoint):
    data = _ua_font_table(path, b"cmap")
    if len(data) < 4:
        return False
    count = struct.unpack_from(">H", data, 2)[0]
    for index in range(min(count, (len(data) - 4) // 8)):
        platform_id, encoding, offset = struct.unpack_from(">HHI", data, 4 + index * 8)
        if platform_id not in (0, 3) or (platform_id == 3 and encoding not in (1, 10)) or offset + 4 > len(data):
            continue
        form = struct.unpack_from(">H", data, offset)[0]
        if form == 12 and offset + 16 <= len(data):
            length, _language, groups = struct.unpack_from(">III", data, offset + 4)
            if offset + length > len(data) or 16 + groups * 12 > length:
                continue
            for group in range(groups):
                start, end, glyph = struct.unpack_from(">III", data, offset + 16 + group * 12)
                if start <= codepoint <= end:
                    return glyph + codepoint - start != 0
        elif form == 4 and offset + 16 <= len(data) and codepoint <= 0xFFFF:
            length = struct.unpack_from(">H", data, offset + 2)[0]
            segments = struct.unpack_from(">H", data, offset + 6)[0] // 2
            if offset + length > len(data) or 16 + segments * 8 > length:
                continue
            end_base = offset + 14
            start_base = end_base + segments * 2 + 2
            delta_base = start_base + segments * 2
            range_base = delta_base + segments * 2
            for segment in range(segments):
                start = struct.unpack_from(">H", data, start_base + segment * 2)[0]
                end = struct.unpack_from(">H", data, end_base + segment * 2)[0]
                if not start <= codepoint <= end:
                    continue
                delta = struct.unpack_from(">h", data, delta_base + segment * 2)[0]
                range_position = range_base + segment * 2
                distance = struct.unpack_from(">H", data, range_position)[0]
                if not distance:
                    return (codepoint + delta) & 0xFFFF != 0
                glyph_position = range_position + distance + 2 * (codepoint - start)
                if glyph_position + 2 > offset + length:
                    return False
                glyph = struct.unpack_from(">H", data, glyph_position)[0]
                return bool(glyph and (glyph + delta) & 0xFFFF)
    return False


def _ua_register_font(path):
    """Register fonts for this process; never require a system-wide install."""
    if sys.platform == "darwin":
        if "coretext" not in _UA_LIBRARIES:
            core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            text = ctypes.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
            core.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
            core.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
            core.CFRelease.argtypes = [ctypes.c_void_p]
            core.CFRelease.restype = None
            text.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            text.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
            _UA_LIBRARIES["coretext"] = (core, text)
        core, text = _UA_LIBRARIES["coretext"]
        encoded = os.fsencode(path)
        url = core.CFURLCreateFromFileSystemRepresentation(None, encoded, len(encoded), False)
        if not url:
            return False
        error = ctypes.c_void_p()
        try:
            return bool(text.CTFontManagerRegisterFontsForURL(url, 1, ctypes.byref(error)))
        finally:
            if error.value:
                core.CFRelease(error)
            core.CFRelease(url)
    if sys.platform == "win32":
        gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi.AddFontResourceExW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
        gdi.AddFontResourceExW.restype = ctypes.c_int
        return bool(gdi.AddFontResourceExW(str(path), 0x10, None))
    if sys.platform.startswith("linux"):
        if "fontconfig" not in _UA_LIBRARIES:
            library = ctypes.util.find_library("fontconfig")
            if not library:
                return False
            fontconfig = ctypes.CDLL(library)
            fontconfig.FcConfigGetCurrent.restype = ctypes.c_void_p
            fontconfig.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            fontconfig.FcConfigAppFontAddFile.restype = ctypes.c_int
            _UA_LIBRARIES["fontconfig"] = fontconfig
        fontconfig = _UA_LIBRARIES["fontconfig"]
        return bool(fontconfig.FcConfigAppFontAddFile(fontconfig.FcConfigGetCurrent(), os.fsencode(path)))
    return False


def ui_load_fonts():
    global _UA_STATE
    if _UA_STATE is not None:
        return _UA_STATE
    records, errors = [], []
    seen = set()
    roots = ui_resource_roots()
    # Preserve the original fonts/ folder. The shipped fallback is used only
    # when the corresponding font family was not supplied by the project.
    directories = [(root / "fonts", 0) for root in roots]
    directories += [(root / "assets" / "fonts", 1) for root in roots]
    project_families = set()
    for directory, priority in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf") or not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                family = _ua_font_family(path)
                category = next((name for name in ("pretendard", "phosphor") if name in family.lower()), None)
                if not category or (priority and category in project_families):
                    continue
                registered = _ua_register_font(path)
                records.append({"path": str(path), "family": family, "registered": registered,
                                "priority": priority})
                if priority == 0:
                    project_families.add(category)
            except (OSError, ValueError, struct.error) as exc:
                errors.append(f"{path.name}: {exc}")
    text_fonts = [item for item in records if "pretendard" in item["family"].lower()]
    text_fonts.sort(key=lambda item: (item["priority"], item["family"].lower() != "pretendard",
                                      "regular" not in Path(item["path"]).stem.lower()))
    icon_fonts = [item for item in records if "phosphor" in item["family"].lower()]
    icon_fonts.sort(key=lambda item: (item["priority"], item["family"].lower() != "phosphor",
                                      "regular" not in item["path"].lower(), item["path"]))
    _UA_STATE = {"family": text_fonts[0]["family"] if text_fonts else "Pretendard",
                 "fonts": records, "icon_fonts": icon_fonts, "errors": errors}
    return _UA_STATE


def ui_font_family():
    return ui_load_fonts()["family"]


def ui_icon_image(name, size=24, color="#687180"):
    """Render the real Phosphor glyph from the local font, including U+E4C0."""
    codepoint = _UA_GLYPHS.get(name)
    if codepoint is None:
        return None
    size = max(8, int(size))
    key = (name, size, color)
    if key in _UA_ICONS:
        return _UA_ICONS[key].copy()
    from PIL import Image, ImageDraw, ImageFont
    for record in ui_load_fonts()["icon_fonts"]:
        try:
            if not _ua_has_glyph(record["path"], codepoint):
                continue
            scale = 4
            edge = size * scale
            font = ImageFont.truetype(record["path"], edge)
            glyph = chr(codepoint)
            left, top, right, bottom = font.getbbox(glyph)
            if right <= left or bottom <= top:
                continue
            result = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
            ImageDraw.Draw(result).text(((edge - right - left) / 2, (edge - bottom - top) / 2), glyph, font=font, fill=color)
            result = result.resize((size, size), Image.Resampling.LANCZOS)
            _UA_ICONS[key] = result
            return result.copy()
        except (OSError, ValueError, struct.error):
            continue
    return None

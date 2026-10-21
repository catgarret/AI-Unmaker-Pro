import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile


_PUBLISH_BLOCK_SIZE = 2 * 1024 * 1024
_LINK_UNSUPPORTED = {errno.EXDEV, errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP}


def _publication_identity(info):
    return info.st_dev, info.st_ino


def _publication_signature(info):
    return (*_publication_identity(info), info.st_size, info.st_mtime_ns)


def _rename_no_replace(source, target):
    """Atomically rename or fail; never fall back to a replacing POSIX rename."""
    if sys.platform == "win32":
        # Windows os.rename always rejects an existing destination.
        os.rename(source, target)
        return
    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if b"\0" in source_bytes or b"\0" in target_bytes:
        raise ValueError("저장 경로에 null 문자를 사용할 수 없습니다.")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, target_bytes, 0x00000004)
    else:
        rename = None
    if rename is None:
        raise OSError(errno.ENOTSUP, "이 환경은 기존 파일을 보호하는 새 파일 저장을 지원하지 않습니다.", str(target))
    if result != 0:
        code = ctypes.get_errno() or errno.EIO
        if code in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}:
            raise OSError(code, "이 파일 시스템은 기존 파일을 보호하는 새 파일 저장을 지원하지 않습니다.", str(target))
        raise OSError(code, os.strerror(code), str(target))


def publish_new_file(staged_path, target_path):
    """Return the new target Path; never overwrite it or remove the source.

    Existing files, directories and dangling symlinks fail with FileExistsError.
    Only unsupported-hard-link errors enable the copy fallback. Copying happens
    in an O_EXCL-created private file; a no-replace rename publishes it after
    verification. No code path unlinks or replaces the public destination.
    The caller guarantees that the source is a private, immutable staged file.
    """
    source = Path(staged_path).expanduser().resolve(strict=True)
    target = Path(target_path).expanduser().absolute()
    with source.open("rb") as src:
        original = os.fstat(src.fileno())
        if not stat.S_ISREG(original.st_mode):
            raise ValueError("저장할 임시 파일이 일반 파일이 아닙니다.")
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in _LINK_UNSUPPORTED:
                raise
        else:
            linked = target.lstat()
            if (not stat.S_ISREG(linked.st_mode)
                    or _publication_identity(linked) != _publication_identity(original)
                    or _publication_signature(os.fstat(src.fileno())) != _publication_signature(original)):
                # Do not unlink the public name: a concurrent writer could
                # replace it between an identity check and unlink.
                raise RuntimeError("저장하는 동안 파일 또는 저장 경로가 변경되었습니다.")
            return target

        with tempfile.TemporaryDirectory(prefix=".unmarker-publish-", dir=target.parent) as directory:
            private = Path(directory) / "complete"
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(private, flags, 0o666)
            try:
                dst = os.fdopen(descriptor, "w+b")
            except BaseException:
                os.close(descriptor)
                raise
            with dst:
                expected = hashlib.sha256()
                total = 0
                for block in iter(lambda: src.read(_PUBLISH_BLOCK_SIZE), b""):
                    if dst.write(block) != len(block):
                        raise OSError(errno.EIO, "저장 파일을 끝까지 쓰지 못했습니다.")
                    expected.update(block)
                    total += len(block)
                dst.flush()
                os.fsync(dst.fileno())
                if (total != original.st_size
                        or _publication_signature(os.fstat(src.fileno())) != _publication_signature(original)
                        or _publication_signature(source.stat()) != _publication_signature(original)):
                    raise RuntimeError("저장하는 동안 임시 파일이 변경되었습니다.")
                dst.seek(0)
                actual = hashlib.sha256()
                for block in iter(lambda: dst.read(_PUBLISH_BLOCK_SIZE), b""):
                    actual.update(block)
                if (os.fstat(dst.fileno()).st_size != total or actual.digest() != expected.digest()):
                    raise RuntimeError("저장 파일의 내용 검증에 실패했습니다.")
                copied = os.fstat(dst.fileno())
            _rename_no_replace(private, target)
            published = target.lstat()
            if (not stat.S_ISREG(published.st_mode)
                    or _publication_identity(published) != _publication_identity(copied)):
                raise RuntimeError("저장하는 동안 저장 경로가 변경되었습니다.")
    return target

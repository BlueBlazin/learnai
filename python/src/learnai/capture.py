"""File-descriptor-level output capture for the harness worker.

Redirecting ``sys.stdout`` is not enough: C extensions (and anything else that writes
to file descriptor 1 or 2 directly) would bypass it and corrupt the protocol stream.
So the worker points descriptors 1 and 2 themselves at a capture file and keeps a
private duplicate of the original stdout for protocol lines.
"""

from __future__ import annotations

import ctypes
import io
import os
import sys

from learnai.protocol import STDOUT_LIMIT_BYTES


def _flush_c_stdio() -> None:
    """Flush C-level stdio buffers (``printf`` from extensions buffers when fd 1 is a file)."""
    # On POSIX, None names the running process and whatever C library it loaded. Windows
    # has no such handle: the C runtime that Python and its extensions share is named.
    try:
        ctypes.CDLL("ucrtbase" if os.name == "nt" else None).fflush(None)
    except OSError, AttributeError:
        pass


class FdCapture:
    """Send everything written to fds 1 and 2 into ``path``; hand it out per case."""

    def __init__(self, path: str) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_c_stdio()
        self.protocol_fd = os.dup(1)  # the real stdout: protocol lines only
        self.diagnostic_fd = os.dup(2)  # the real stderr: harness diagnostics
        # O_APPEND makes every write land at the current end of file, so truncating
        # the file between cases is all the bookkeeping there is. Both descriptors
        # share one file so stdout and stderr stay interleaved in order.
        # O_BINARY (Windows only): without it the C runtime rewrites "\n" as "\r\n" in
        # everything written through these descriptors.
        flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND | getattr(os, "O_BINARY", 0)
        self._file_fd = os.open(path, flags, 0o600)
        os.dup2(self._file_fd, 1)
        os.dup2(self._file_fd, 2)
        # Unbuffered text streams: if the watchdog has to kill this process, whatever
        # the solution printed so far is already in the file for the supervisor to read.
        sys.stdout = self._text_stream(1)
        sys.stderr = self._text_stream(2)

    @staticmethod
    def _text_stream(fd: int) -> io.TextIOWrapper:
        raw = os.fdopen(fd, "wb", buffering=0, closefd=False)
        # newline="\n": the captured text is what the solution wrote, on every platform.
        return io.TextIOWrapper(
            raw, encoding="utf-8", errors="backslashreplace", write_through=True, newline="\n"
        )

    def _flush(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except OSError, ValueError:
                pass
        _flush_c_stdio()

    def begin(self) -> None:
        """Discard anything captured so far and start a fresh capture."""
        self._flush()
        os.ftruncate(self._file_fd, 0)

    def take(self) -> tuple[bytes, int]:
        """Return (captured bytes, total size) since the last ``begin`` and reset.

        At most a little over 64 KB is read back; the total size lets the caller say
        how much was dropped.
        """
        self._flush()
        size = os.fstat(self._file_fd).st_size
        want = min(size, STDOUT_LIMIT_BYTES + 1)
        chunks: list[bytes] = []
        offset = 0
        while offset < want:
            # Seek and read instead of os.pread, which Windows does not have. Writers
            # are unaffected: O_APPEND puts every write at the end of the file.
            os.lseek(self._file_fd, offset, os.SEEK_SET)
            chunk = os.read(self._file_fd, want - offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        os.ftruncate(self._file_fd, 0)
        return b"".join(chunks), size

    def write_protocol(self, line: str) -> None:
        data = (line + "\n").encode("utf-8")
        while data:
            written = os.write(self.protocol_fd, data)
            data = data[written:]

    def write_diagnostic(self, text: str) -> None:
        try:
            os.write(self.diagnostic_fd, text.encode("utf-8", errors="replace"))
        except OSError:
            pass

"""
Flight recorder: every byte that crosses the USB link, in both directions,
from the moment the port opens until it closes.

WHY RAW BYTES AND NOT PARSED PACKETS

A recording exists for the run where something went wrong, and that is
exactly the run where parsing may be the thing that failed: a version
mismatch, a CRC storm, a board that rebooted mid-run and came back talking
boot text. A recorder that stored only what parsed would have nothing to show
for any of those. This one stores what arrived, stamped with when it arrived,
and leaves the interpretation to whoever reads it later - with whatever parser
is right by then.

It is also cheap enough to leave on. The board sends 426 bytes a millisecond,
so an hour is ~1.5 GB, written in large buffered chunks from the reader thread
that already holds the bytes. Nothing here can block the link: a full disk or
a failed write turns the recorder off and says so, and the robot carries on.

FILE FORMAT (little-endian)

    header   b"ZEUSFLT1"  u32 n  n bytes of UTF-8 JSON  (version, start time, port)
    record   u8 dir  u64 t_ns  u32 len  len bytes
             dir 0 = board -> Pi (RX), as read off the port, unframed
             dir 1 = Pi -> board (TX), one whole packet per record
             t_ns  = time.monotonic_ns() on the Pi when the bytes were handed
                     over - host time, so link latency and jitter can be
                     measured against the board's own tick in the packets

Read it back with read_records(), or zeus_rerun's flight_to_rrd.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from typing import BinaryIO, Iterator, Optional, Tuple

MAGIC = b"ZEUSFLT1"
RX = 0
TX = 1
_REC = struct.Struct("<BQI")

# A large write buffer: the reader thread hands over a few hundred bytes per
# read, and a syscall per read at 1 kHz is wasted work on a Pi.
_BUFFER = 1 << 20


class FlightRecorder:
    """Append-only, thread-safe. The link's reader thread writes RX and the
    sender writes TX; one lock serialises them so records never interleave."""

    def __init__(self, path: str, port: str = "", proto_version: int = 0) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._f: Optional[BinaryIO] = open(self.path, "wb", buffering=_BUFFER)
        self._lock = threading.Lock()
        self.bytes_written = 0
        self.error: Optional[str] = None

        header = json.dumps({
            "format": 1,
            "proto_version": proto_version,
            "port": port,
            "host": socket.gethostname(),
            "start_wall": time.time(),
            "start_mono_ns": time.monotonic_ns(),
        }).encode()
        self._f.write(MAGIC + struct.pack("<I", len(header)) + header)

    def write(self, direction: int, data: bytes) -> None:
        if not data:
            return
        t = time.monotonic_ns()
        with self._lock:
            if self._f is None:
                return
            try:
                self._f.write(_REC.pack(direction, t, len(data)))
                self._f.write(data)
                self.bytes_written += _REC.size + len(data)
            except OSError as exc:
                # Disk full, card pulled: stop recording, never stop the robot.
                self.error = str(exc)
                self._close_locked()

    def flush(self) -> None:
        with self._lock:
            if self._f is not None:
                try:
                    self._f.flush()
                except OSError as exc:
                    self.error = str(exc)
                    self._close_locked()

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None

    @property
    def active(self) -> bool:
        return self._f is not None


def read_header(f: BinaryIO) -> dict:
    if f.read(len(MAGIC)) != MAGIC:
        raise ValueError("not a Zeus flight log (bad magic)")
    (n,) = struct.unpack("<I", f.read(4))
    return json.loads(f.read(n))


def read_records(path: str) -> Tuple[dict, Iterator[Tuple[int, int, bytes]]]:
    """(header, iterator of (direction, t_ns, data)). A record cut short by a
    crash or a power cut ends the iteration quietly - that tail is the one
    part of a recording that is allowed to be missing."""
    f = open(path, "rb")
    header = read_header(f)

    def gen():
        with f:
            while True:
                h = f.read(_REC.size)
                if len(h) < _REC.size:
                    return
                d, t, n = _REC.unpack(h)
                data = f.read(n)
                if len(data) < n:
                    return
                yield d, t, data

    return header, gen()


# Files make_room() may delete, oldest first. Recordings, their Rerun
# conversions and the policy's timing logs - nothing else in the directory.
_PRUNABLE = ("zeus_", "policy_timing_")
_PRUNABLE_EXT = (".flt", ".rrd", ".csv")
KEEP_FILE = "KEEP"


def make_room(directory: str = "~/zeus_logs", min_free_bytes: int = 4 << 30,
              free_bytes=None) -> list:
    """
    Delete the oldest recordings until the disk has min_free_bytes free.

    2026-11-28: the Pi's 29 GB card filled with 137 recordings (17 GB) and ROS
    could not even create its log directory - the robot would not launch. A
    run is 100-600 MB, so this runs before every new recording.

    Never deleted: anything named in <directory>/KEEP (one file name per line)
    - the walks worth keeping are also the OLDEST, exactly what oldest-first
    takes first - and anything that is not a recording, Rerun file or timing
    log. Returns the paths it removed. Best effort: a file it cannot remove is
    skipped, and it never raises.
    """
    import shutil
    d = os.path.abspath(os.path.expanduser(directory))
    free = free_bytes or (lambda: shutil.disk_usage(d).free)
    removed = []
    try:
        if free() >= min_free_bytes:
            return removed
        keep = set()
        try:
            with open(os.path.join(d, KEEP_FILE)) as f:
                keep = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        except OSError:
            pass
        names = [n for n in os.listdir(d)
                 if n.startswith(_PRUNABLE) and n.endswith(_PRUNABLE_EXT) and n not in keep]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(d, n)))
        for n in names:
            if free() >= min_free_bytes:
                break
            try:
                os.remove(os.path.join(d, n))
                removed.append(os.path.join(d, n))
            except OSError:
                continue
    except OSError:
        pass
    return removed


def default_path(directory: str = "~/zeus_logs") -> str:
    """~/zeus_logs/zeus_YYYYmmdd_HHMMSS.flt - one file per run, sortable."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.expanduser(directory), f"zeus_{stamp}.flt")

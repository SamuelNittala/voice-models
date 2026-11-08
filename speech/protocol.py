"""What the speech front and its two workers share. Standard library only: the front never imports torch.

Frames over pipes: one kind byte, a 4-byte big-endian length, the payload.
  Front to worker:  Q (JSON request), then B (raw audio) for speech to text.
  Worker to front:  R (ready, JSON), A (MP3 bytes), E (done, JSON), X (failed, JSON {"status", "error"}).
"""
import json
import os
import re
import select
import struct
import sys
import threading
import time

HEADER = struct.Struct(">cI")
MAX_FRAME = 4 * 1024 * 1024
MAX_TTS_CHARS = 300
VOCABULARY_TOKENS = 223  # Whisper keeps only the last 223 tokens of a prompt (n_text_ctx // 2 - 1).

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9]")


class FrameError(Exception):
    pass


def write_frame(handle, kind, payload=b"", lock=None):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload).encode()
    data = HEADER.pack(kind, len(payload)) + payload
    if lock is None:
        handle.write(data)
        handle.flush()
        return
    with lock:
        handle.write(data)
        handle.flush()


class FrameReader:
    """Reads frames from a pipe's file descriptor, each read within a deadline (time.monotonic)."""
    def __init__(self, fd):
        self.fd = fd
        self.buffer = bytearray()

    def _fill(self, size, deadline):
        while len(self.buffer) < size:
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise TimeoutError("no frame in time")
            ready, _, _ = select.select([self.fd], [], [], left)
            if not ready:
                raise TimeoutError("no frame in time")
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise EOFError("the other side closed the pipe")
            self.buffer += chunk

    def read(self, deadline=None):
        self._fill(HEADER.size, deadline)
        kind, length = HEADER.unpack(bytes(self.buffer[:HEADER.size]))
        if kind not in (b"Q", b"B", b"R", b"A", b"E", b"X") or length > MAX_FRAME:
            raise FrameError("not a frame")
        self._fill(HEADER.size + length, deadline)
        payload = bytes(self.buffer[HEADER.size:HEADER.size + length])
        del self.buffer[:HEADER.size + length]
        return kind, payload


def json_payload(payload):
    try:
        value = json.loads(payload)
    except ValueError:
        raise FrameError("not JSON") from None
    if not isinstance(value, dict):
        raise FrameError("not a JSON object")
    return value


# ---------------------------------------------------------------- worker side

def worker_pipes():
    """The worker's frame output and input. The real stdout is kept for frames only, and fd 1 then points at stderr,
    so a print from any library goes to the terminal instead of breaking a frame."""
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out, FrameReader(0)


def watch_parent():
    """Exits the worker if the front goes away, whatever it is doing."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os._exit(0)
    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


# ---------------------------------------------------------------- pure helpers

def clean_text(text):
    """Control characters out, whitespace collapsed."""
    kept = "".join(" " if ch in "\t\r\n" else ch for ch in text if ch in "\t\r\n" or ch.isprintable())
    return " ".join(kept.split())


def sentences(text):
    """Split after `.`, `!` or `?` followed by a space. A piece with fewer than two letters or digits joins the one
    before it, so nothing is ever spoken alone that is only punctuation or a stray number."""
    pieces = []
    for piece in _SENTENCE_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        if pieces and len(_WORD.findall(piece)) < 2:
            pieces[-1] = f"{pieces[-1]} {piece}"
        else:
            pieces.append(piece)
    return pieces


def fit_vocabulary(text, count_tokens, limit=VOCABULARY_TOKENS):
    """The vocabulary as Whisper's prompt, with whole comma-separated items dropped from the end until it fits."""
    items = [item for item in text.split(", ") if item.strip()]
    while items:
        prompt = ", ".join(items).strip()
        if count_tokens(" " + prompt) <= limit:
            return prompt
        items.pop()
    return None


AUDIO_TYPES = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/x-m4a": "mp4",
               "audio/m4a": "mp4", "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav"}


def audio_family(data):
    """The container the bytes really are, from their magic, or None."""
    if data[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[4:8] == b"ftyp":
        return "mp4"
    return None

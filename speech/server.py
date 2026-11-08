#!/usr/bin/env python3
"""Speech service for the voice gateway on the Mac: Chatterbox text to speech and Whisper speech to text.

A small HTTP front (standard library only, no torch) and two worker processes it starts, warms up and watches: one
for Chatterbox (tts_worker.py), one for faster-whisper (stt_worker.py). They are separate processes because Whisper
slows from 0.22 s to 0.35 s a clip when chatterbox is imported in the same process.

  GET  /health  each worker's state: starting, ready, busy, restarting or down.
  POST /tts     JSON {"text"}, 1-300 characters once control characters are out. Answers audio/mpeg, chunked: the
                first sentence is made and sent first, the rest follow in order through one MP3 encoder, so the parts
                are one stream. The 200 goes out only once the first MP3 bytes exist; a failure before is JSON.
  POST /stt     the recording as the body (at most 2 MiB; webm, ogg, mp4/m4a or wav, checked against the bytes
                themselves), an optional X-Vocabulary header used as Whisper's prompt, English. Answers {"text"}.

Settings, both required; the service refuses to start if either is unset or invalid:
  SPEECH_BIND_IP          the tailnet address to listen on: 127.0.0.1 or an address in 100.64.0.0/10.
  SPEECH_ALLOWED_CLIENTS  comma separated IPv4 addresses allowed to connect; 127.0.0.1 is always allowed.

Security: binds SPEECH_BIND_IP or 127.0.0.1 only, never 0.0.0.0. Clients other than SPEECH_ALLOWED_CLIENTS and
127.0.0.1 are dropped before a byte is read. `Authorization: Bearer <token>` on every request, compared in constant
time; the token is made on first start in ./token, 0600. Strict sizes and timeouts, one request at a time per worker
with at most two waiting, and no path or file name is ever taken from a request. The log has method, path and status
only, never text or audio.

Run in the foreground with ./start.sh, which reads both settings from ./local.env. Ctrl+C stops the front and both
workers.
"""
import argparse
import hmac
import ipaddress
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as p  # noqa: E402

LOOPBACK = "127.0.0.1"  # This machine or an SSH tunnel: always a bind choice and always an allowed client.
TAILNET = ipaddress.ip_network("100.64.0.0/10")
BIND_ENV = "SPEECH_BIND_IP"
CLIENTS_ENV = "SPEECH_ALLOWED_CLIENTS"
DEFAULT_PORT = 8790
TOKEN_FILE = HERE / "token"
MAX_TTS_BODY = 4096
MAX_STT_BODY = 2 * 1024 * 1024
MAX_VOCABULARY_CHARS = 1000
SOCKET_TIMEOUT = 10  # For reading a request's headers and body.
MAX_CONNECTIONS = 8
MAX_WAITING = 2  # Besides the one running, per worker.
LIMITS = {  # Seconds: waiting in line, the first answer frame, the whole answer, warming up.
    "tts": {"queue": 10, "first": 15, "total": 40, "ready": 240},
    "stt": {"queue": 15, "first": 30, "total": 30, "ready": 240},
}
RESTART_DELAYS = (2, 5, 15, 30, 60)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,512}")


class Busy(Exception):
    """Too many waiting, or no turn in time."""


class Unavailable(Exception):
    def __init__(self, state):
        super().__init__(state)
        self.state = state


class WorkerFailed(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def log_line(text):
    print(f"speech: {text}", file=sys.stderr, flush=True)


class Worker:
    """One model process: started, warmed up and watched. One request at a time, at most MAX_WAITING more in line.
    A worker that dies, sends garbage or overruns a limit is killed and started again after a short delay."""
    def __init__(self, kind, argv, limits, log=log_line, env=None, restart_delays=RESTART_DELAYS):
        self.kind, self.argv, self.limits, self.log, self.env = kind, argv, limits, log, env
        self.restart_delays = restart_delays
        self.state = "down"
        self.proc = None
        self.reader = None
        self.turn_lock = threading.Lock()
        self.guard = threading.Lock()
        self.in_line = 0
        self.stopping = False
        self.failures = 0
        self.timer = None

    # Life
    def start(self):
        threading.Thread(target=self._start, name=f"{self.kind}-start", daemon=True).start()

    def _start(self):
        with self.guard:
            if self.stopping:
                return
            self.state = "starting" if self.failures == 0 else "restarting"
        proc = None
        try:
            proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(HERE),
                                    env=self.env)
            reader = p.FrameReader(proc.stdout.fileno())
            kind, _payload = reader.read(time.monotonic() + self.limits["ready"])
            if kind != b"R":
                raise p.FrameError("no ready frame")
        except (OSError, TimeoutError, EOFError, p.FrameError) as error:
            self.log(f"{self.kind} worker did not start ({type(error).__name__})")
            self._kill(proc)
            self._again()
            return
        with self.guard:
            if self.stopping:
                self._kill(proc)
                return
            self.proc, self.reader, self.state, self.failures = proc, reader, "ready", 0
        self.log(f"{self.kind} worker ready (pid {proc.pid})")

    def _again(self):
        with self.guard:
            if self.stopping:
                self.state = "down"
                return
            delay = self.restart_delays[min(self.failures, len(self.restart_delays) - 1)]
            self.failures += 1
            self.state = "restarting"
            self.timer = threading.Timer(delay, self.start)
            self.timer.daemon = True
            self.timer.start()

    @staticmethod
    def _kill(proc):
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)

    def broken(self, why):
        """Called while holding the turn: the worker cannot be trusted any more."""
        self.log(f"{self.kind} worker restarted ({why})")
        with self.guard:
            proc, self.proc, self.reader = self.proc, None, None
            self.state = "restarting"
        self._kill(proc)
        self._again()

    def stop(self):
        with self.guard:
            self.stopping = True
            proc, self.proc = self.proc, None
            if self.timer is not None:
                self.timer.cancel()
            self.state = "down"
        self._kill(proc)

    # Turns
    @contextmanager
    def turn(self):
        with self.guard:
            if self.in_line >= 1 + MAX_WAITING:
                raise Busy()
            self.in_line += 1
        try:
            if not self.turn_lock.acquire(timeout=self.limits["queue"]):
                raise Busy()
            try:
                with self.guard:
                    if self.state != "ready":
                        raise Unavailable(self.state)
                    self.state = "busy"
                yield self
            finally:
                with self.guard:
                    if self.state == "busy":
                        self.state = "ready"
                self.turn_lock.release()
        finally:
            with self.guard:
                self.in_line -= 1

    def send(self, *frames):
        try:
            for kind, payload in frames:
                p.write_frame(self.proc.stdin, kind, payload)
        except (OSError, ValueError, AttributeError):
            self.broken("could not write")
            raise WorkerFailed(502) from None

    def read(self, deadline):
        try:
            return self.reader.read(deadline)
        except TimeoutError:
            self.broken("too slow")
            raise WorkerFailed(504) from None
        except (EOFError, p.FrameError, OSError, ValueError, AttributeError) as error:
            self.broken(type(error).__name__)
            raise WorkerFailed(502) from None


def load_or_make_token(path):
    """The token, made on first start. Refuses a link, a non-file, a file others can read or someone else's file."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise SystemExit(f"speech: {path} is not a regular file that can be read") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise SystemExit(f"speech: {path} is not a regular file of this user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise SystemExit(f"speech: {path} can be read by other users; chmod 600 it")
        raw = os.read(fd, 1024)
    finally:
        os.close(fd)
    token = raw.decode("ascii", "replace").strip()
    if _TOKEN_RE.fullmatch(token) is None:
        raise SystemExit(f"speech: {path} does not hold a usable token; remove it to make a new one")
    return token


def _ipv4(text):
    try:
        return ipaddress.IPv4Address(text)
    except ValueError:
        return None


def bind_setting(value):
    """SPEECH_BIND_IP: 127.0.0.1 or one tailnet address (100.64.0.0/10). Anything else stops the start."""
    address = _ipv4((value or "").strip())
    if address is None or not (str(address) == LOOPBACK or address in TAILNET):
        raise SystemExit(f"speech: set {BIND_ENV} to 127.0.0.1 or a tailnet address in {TAILNET}")
    return str(address)


def clients_setting(value):
    """SPEECH_ALLOWED_CLIENTS: comma separated IPv4 addresses. 127.0.0.1 is always added."""
    items = [item.strip() for item in (value or "").split(",")]
    addresses = [_ipv4(item) for item in items]
    if not value or any(address is None or address.is_unspecified for address in addresses):
        raise SystemExit(f"speech: set {CLIENTS_ENV} to comma separated IPv4 addresses, e.g. 100.64.0.2")
    return frozenset({LOOPBACK, *(str(address) for address in addresses)})


def settings(env):
    """Both settings from the environment, checked before anything else starts."""
    return bind_setting(env.get(BIND_ENV)), clients_setting(env.get(CLIENTS_ENV))


def valid_bind(host, bind_ip):
    return host in (bind_ip, LOOPBACK)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "speech"
    sys_version = ""
    timeout = SOCKET_TIMEOUT

    def log_request(self, code="-", size="-"):
        path = getattr(self, "path", "").split("?", 1)[0][:40]
        safe = path if path.isascii() and path.isprintable() else "(unprintable)"
        self.server.log(f"{getattr(self, 'command', None) or '-'} {safe} {getattr(code, 'value', code)}")

    def log_message(self, format, *args):
        pass  # The default would write the request line and more; log_request says what is needed.

    def send_error(self, code, message=None, explain=None):
        self._json(int(code), {"error": "bad request" if int(code) < 500 else "server error"})

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if status >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        given = self.headers.get_all("Authorization") or []
        if len(given) != 1:
            return False
        return hmac.compare_digest(given[0].encode("latin-1", "replace"), f"Bearer {self.server.token}".encode())

    def _body(self, limit):
        if self.headers.get("Transfer-Encoding"):
            self._json(411, {"error": "content length required"})
            return None
        raw = self.headers.get_all("Content-Length") or []
        if len(raw) != 1 or not raw[0].isascii() or not raw[0].isdigit():
            self._json(400, {"error": "bad content length"})
            return None
        length = int(raw[0])
        if length > limit:
            self._json(413, {"error": "too large"})
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._json(400, {"error": "body cut short"})
            return None
        return body

    def do_GET(self):
        if not self._authorised():
            self._json(401, {"error": "token"})
        elif self.path == "/health":
            self._json(200, {kind: worker.state for kind, worker in self.server.workers.items()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            self._json(401, {"error": "token"})
        elif self.path == "/tts":
            self._tts()
        elif self.path == "/stt":
            self._stt()
        else:
            self._json(404, {"error": "not found"})

    # Text to speech
    def _tts(self):
        body = self._body(MAX_TTS_BODY)
        if body is None:
            return
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            self._json(400, {"error": "text missing"})
            return
        text = p.clean_text(text)
        if not text:
            self._json(400, {"error": "text missing"})
            return
        if len(text) > p.MAX_TTS_CHARS:
            self._json(413, {"error": "text too long"})
            return
        self._with_worker("tts", lambda worker: self._speak(worker, text))

    def _speak(self, worker, text):
        started = time.monotonic()
        limits = worker.limits
        worker.send((b"Q", {"op": "tts", "text": text}))
        kind, payload = worker.read(started + limits["first"])
        if kind == b"X":
            self._worker_error(payload)
            return
        if kind != b"A":
            worker.broken("unexpected frame")
            raise WorkerFailed(502)
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.streaming = True
        listening = self._chunk(payload)
        while True:  # Read to the end even if the client went away, so the next request starts on a clean pipe.
            kind, payload = worker.read(started + limits["total"])
            if kind == b"A":
                listening = listening and self._chunk(payload)
            elif kind == b"E":
                if listening:
                    self._end_chunks()
                return
            elif kind == b"X":
                self.close_connection = True  # No closing chunk: the client sees a failed load.
                return
            else:
                worker.broken("unexpected frame")
                self.close_connection = True
                return

    def _chunk(self, data):
        try:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
            return True
        except OSError:
            self.close_connection = True
            return False

    def _end_chunks(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            self.close_connection = True

    # Speech to text
    def _stt(self):
        kinds = self.headers.get_all("Content-Type") or []
        family = p.AUDIO_TYPES.get(kinds[0].split(";", 1)[0].strip().lower()) if len(kinds) == 1 else None
        if family is None:
            self._json(415, {"error": "unsupported audio type"})
            return
        vocabulary = self.headers.get_all("X-Vocabulary") or [""]
        if len(vocabulary) != 1 or len(vocabulary[0]) > MAX_VOCABULARY_CHARS or not (
                vocabulary[0].isascii() and vocabulary[0].isprintable()):
            self._json(400, {"error": "bad vocabulary"})
            return
        audio = self._body(MAX_STT_BODY)
        if audio is None:
            return
        if not audio:
            self._json(400, {"error": "empty recording"})
            return
        if p.audio_family(audio) != family:
            self._json(415, {"error": "the recording is not what its type says"})
            return
        self._with_worker("stt", lambda worker: self._listen(worker, vocabulary[0], family, audio))

    def _listen(self, worker, vocabulary, family, audio):
        started = time.monotonic()
        worker.send((b"Q", {"op": "stt", "vocabulary": vocabulary, "family": family}), (b"B", audio))
        kind, payload = worker.read(started + worker.limits["first"])
        if kind == b"X":
            self._worker_error(payload)
            return
        if kind != b"E":
            worker.broken("unexpected frame")
            raise WorkerFailed(502)
        try:
            text = p.json_payload(payload).get("text")
        except p.FrameError:
            text = None
        if not isinstance(text, str):
            worker.broken("unreadable answer")
            raise WorkerFailed(502)
        self._json(200, {"text": text})

    # Both
    def _with_worker(self, kind, work):
        worker = self.server.workers[kind]
        self.streaming = False
        try:
            with worker.turn():
                work(worker)
        except Busy:
            self._json(503, {"error": f"{kind} is busy"})
        except Unavailable as error:
            self._json(503, {"error": f"{kind} is {error.state}"})
        except WorkerFailed as error:
            if self.streaming:
                self.close_connection = True
            else:
                self._json(error.status, {"error": f"{kind} failed"})

    def _worker_error(self, payload):
        try:
            info = p.json_payload(payload)
        except p.FrameError:
            info = {}
        status = info.get("status") if isinstance(info.get("status"), int) and 400 <= info["status"] < 600 else 500
        reason = info.get("error") if isinstance(info.get("error"), str) else "failed"
        self._json(status, {"error": reason[:200]})


class SpeechServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, token, workers, clients, log=log_line):
        self.token, self.workers, self.clients, self.log = token, workers, frozenset(clients), log
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, Handler)

    def verify_request(self, request, client_address):
        """Anyone but the allowed clients or this machine is dropped before a byte is read."""
        try:
            return str(ipaddress.ip_address(client_address[0])) in self.clients
        except ValueError:
            return False

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        kind = sys.exc_info()[0]
        self.log(f"request failed: {kind.__name__ if kind else 'unknown'}")


def worker_env():
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"})
    return env


def main(argv=None):
    bind_ip, clients = settings(os.environ)
    parser = argparse.ArgumentParser(description="Chatterbox and Whisper for the voice gateway.")
    parser.add_argument("--bind", default=bind_ip, help=f"{bind_ip} or {LOOPBACK} (default {bind_ip}, from {BIND_ENV})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if not valid_bind(args.bind, bind_ip):
        raise SystemExit(f"speech: refusing to bind {args.bind!r}: only {bind_ip} or {LOOPBACK}")
    if not 1024 <= args.port <= 65535:
        raise SystemExit("speech: the port must be 1024-65535")
    token = load_or_make_token(TOKEN_FILE)
    python = sys.executable
    workers = {kind: Worker(kind, [python, str(HERE / f"{kind}_worker.py")], LIMITS[kind], env=worker_env())
               for kind in ("tts", "stt")}
    try:
        server = SpeechServer((args.bind, args.port), token, workers, clients)
    except OSError as error:
        raise SystemExit(f"speech: cannot listen on {args.bind}:{args.port}: {error.strerror}") from None
    def on_term(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)  # The terminal or SSH session it runs in went away.
    for worker in workers.values():
        worker.start()
    log_line(f"listening on http://{args.bind}:{args.port}/ for {', '.join(sorted(clients))}; token in {TOKEN_FILE}")
    log_line("warming up both models; GET /health says when they are ready. Ctrl+C stops everything.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log_line("stopping")
        server.server_close()
        for worker in workers.values():
            worker.stop()
        log_line("stopped")


if __name__ == "__main__":
    main()

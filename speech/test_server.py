"""Tests for the speech front with fake workers (fake_worker.py): no models, no GPU. Run:
~/voice-models/run.sh python -m unittest discover -s speech -p 'test_*.py'
or with any Python 3.11+ (the front is standard library only): python3 -m unittest discover -s speech -p 'test_*.py'
"""
import http.client
import io
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as p  # noqa: E402
import server as s  # noqa: E402

TOKEN = "T" * 43
FAST = {"queue": 2, "first": 2, "total": 4, "ready": 10}
WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 100
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 40
SETTINGS = {"SPEECH_BIND_IP": "100.64.0.1", "SPEECH_ALLOWED_CLIENTS": "100.64.0.2"}  # Documentation addresses.


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)

    def text(self):
        return "\n".join(self.lines)


def wait_for(condition, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class FrontCase(unittest.TestCase):
    MODES = {"tts": "ok", "stt": "ok"}
    LIMITS = FAST
    CLIENTS = frozenset({"127.0.0.1"})
    ENV = {}

    def setUp(self):
        self.log = Lines()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workers = {}
        for kind in ("tts", "stt"):
            env = {**os.environ, "FAKE_MODE": self.MODES[kind], **self.ENV}
            self.workers[kind] = s.Worker(kind, [sys.executable, str(HERE / "fake_worker.py"), kind], dict(self.LIMITS),
                                          log=self.log, env=env, restart_delays=(0.2,))
            self.workers[kind].start()
        self.server = s.SpeechServer(("127.0.0.1", 0), TOKEN, self.workers, clients=self.CLIENTS, log=self.log)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        for kind, worker in self.workers.items():
            if self.MODES[kind] not in ("never-ready", "slow-start"):
                self.assertTrue(wait_for(lambda w=worker: w.state == "ready"), kind)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        for worker in self.workers.values():
            worker.stop()

    def request(self, method, path, body=None, token=TOKEN, headers=None, timeout=10):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            sent = dict(headers or {})
            if token is not None:
                sent["Authorization"] = f"Bearer {token}"
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def tts(self, text, **kwargs):
        return self.request("POST", "/tts", json.dumps({"text": text}).encode(),
                            headers={"Content-Type": "application/json"}, **kwargs)

    def stt(self, audio=WEBM, kind="audio/webm;codecs=opus", vocabulary=None, **kwargs):
        headers = {"Content-Type": kind}
        if vocabulary is not None:
            headers["X-Vocabulary"] = vocabulary
        return self.request("POST", "/stt", audio, headers=headers, **kwargs)


class PureTests(unittest.TestCase):
    def test_sentences(self):
        self.assertEqual(p.sentences("Done. The tests pass! Ready? Yes"), ["Done.", "The tests pass!", "Ready?", "Yes"])
        self.assertEqual(p.sentences("From production 4.6.2. Then dev."), ["From production 4.6.2.", "Then dev."])
        self.assertEqual(p.sentences("It is 4. 6."), ["It is 4. 6."])  # "6." alone has one digit: it joins.
        self.assertEqual(p.sentences("   "), [])

    def test_clean_text(self):
        self.assertEqual(p.clean_text("a\x00b\x1b[31m c\n\td"), "ab[31m c d")

    def test_fit_vocabulary_drops_whole_items_from_the_end(self):
        words = lambda text: len(text.split())  # noqa: E731
        text = "Branches: dev, main. Repositories: a, b, c."
        self.assertEqual(p.fit_vocabulary(text, words, limit=100), text)
        self.assertEqual(p.fit_vocabulary(text, words, limit=6), "Branches: dev, main. Repositories: a, b")
        self.assertIsNone(p.fit_vocabulary(text, words, limit=0))
        self.assertIsNone(p.fit_vocabulary("", words))

    def test_audio_family(self):
        self.assertEqual(p.audio_family(WEBM), "webm")
        self.assertEqual(p.audio_family(b"OggS" + b"\x00" * 10), "ogg")
        self.assertEqual(p.audio_family(WAV), "wav")
        self.assertEqual(p.audio_family(b"\x00\x00\x00\x20ftypM4A "), "mp4")
        self.assertIsNone(p.audio_family(b"ID3\x04" + b"\x00" * 10))

    def test_frames_round_trip(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(write_fd, "wb", buffering=0) as out:
            p.write_frame(out, b"A", b"\x00\x01")
            p.write_frame(out, b"E", {"text": "hi"})
        reader = p.FrameReader(read_fd)
        self.assertEqual(reader.read(time.monotonic() + 1), (b"A", b"\x00\x01"))
        kind, payload = reader.read(time.monotonic() + 1)
        self.assertEqual((kind, p.json_payload(payload)), (b"E", {"text": "hi"}))
        with self.assertRaises(EOFError):
            reader.read(time.monotonic() + 1)
        os.close(read_fd)

    def test_binds(self):
        self.assertTrue(s.valid_bind("100.64.0.1", "100.64.0.1"))
        self.assertTrue(s.valid_bind("127.0.0.1", "100.64.0.1"))
        for bad in ("0.0.0.0", "", "::", "192.168.1.10", "localhost", "100.64.0.2"):
            self.assertFalse(s.valid_bind(bad, "100.64.0.1"), bad)
        with unittest.mock.patch.dict(os.environ, SETTINGS), self.assertRaises(SystemExit):
            s.main(["--bind", "0.0.0.0"])


class SettingsTests(unittest.TestCase):
    """SPEECH_BIND_IP and SPEECH_ALLOWED_CLIENTS: both required, checked before anything else starts."""
    def assertRefused(self, call, name):
        with self.assertRaises(SystemExit) as caught:
            call()
        message = str(caught.exception.code)
        self.assertTrue(message.startswith("speech: "), message)
        self.assertNotIn("\n", message)
        self.assertIn(name, message)

    def test_bind_setting(self):
        for good in ("100.64.0.1", "100.127.255.254", "127.0.0.1", " 100.64.0.1 "):
            self.assertEqual(s.bind_setting(good), good.strip())
        for bad in (None, "", " ", "0.0.0.0", "192.168.1.10", "100.63.255.255", "100.128.0.1", "::1", "localhost",
                    "100.64.0.256", "100.64.0.1,127.0.0.1"):
            self.assertRefused(lambda b=bad: s.bind_setting(b), "SPEECH_BIND_IP")

    def test_clients_setting_always_allows_this_machine(self):
        self.assertEqual(s.clients_setting("100.64.0.2"), frozenset({"100.64.0.2", "127.0.0.1"}))
        self.assertEqual(s.clients_setting("100.64.0.2, 192.168.1.20"),
                         frozenset({"100.64.0.2", "192.168.1.20", "127.0.0.1"}))
        self.assertEqual(s.clients_setting("127.0.0.1"), frozenset({"127.0.0.1"}))
        for bad in (None, "", " , ", "100.64.0.2,,", "nope", "0.0.0.0", "::1", "100.64.0.256", "100.64.0.0/10"):
            self.assertRefused(lambda b=bad: s.clients_setting(b), "SPEECH_ALLOWED_CLIENTS")

    def test_main_refuses_to_start_without_them(self):
        for missing in ("SPEECH_BIND_IP", "SPEECH_ALLOWED_CLIENTS"):
            env = {key: value for key, value in SETTINGS.items() if key != missing}
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                self.assertRefused(lambda: s.main([]), missing)
        with unittest.mock.patch.dict(os.environ, {**SETTINGS, "SPEECH_BIND_IP": "0.0.0.0"}, clear=True):
            self.assertRefused(lambda: s.main([]), "SPEECH_BIND_IP")

    def test_the_server_lets_in_only_the_configured_clients(self):
        server = s.SpeechServer(("127.0.0.1", 0), TOKEN, {}, clients=s.clients_setting("100.64.0.2"))
        self.addCleanup(server.server_close)
        self.assertTrue(server.verify_request(None, ("100.64.0.2", 50000)))
        self.assertTrue(server.verify_request(None, ("127.0.0.1", 50000)))
        for other in ("100.64.0.3", "192.168.1.10", "not an address"):
            self.assertFalse(server.verify_request(None, (other, 50000)), other)


class TokenTests(unittest.TestCase):
    def test_made_once_private_then_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            token = s.load_or_make_token(path)
            self.assertRegex(token, r"^[A-Za-z0-9_-]{43}$")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(s.load_or_make_token(path), token)

    def test_unsafe_files_stop_the_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            s.load_or_make_token(path)
            path.chmod(0o644)
            with self.assertRaises(SystemExit):
                s.load_or_make_token(path)
            path.unlink()
            (Path(tmp) / "other").write_text(TOKEN)
            path.symlink_to(Path(tmp) / "other")
            with self.assertRaises(SystemExit):
                s.load_or_make_token(path)
            path.unlink()
            path.write_text("short\n")
            path.chmod(0o600)
            with self.assertRaises(SystemExit):
                s.load_or_make_token(path)


class GuardTests(FrontCase):
    def test_token_on_every_route(self):
        for token in (None, "wrong", TOKEN + "x", ""):
            self.assertEqual(self.request("GET", "/health", token=token)[0], 401, token)
            self.assertEqual(self.tts("Hi.", token=token)[0], 401, token)
            self.assertEqual(self.stt(token=token)[0], 401, token)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest("GET", "/health")
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 401)
        connection.close()

    def test_health(self):
        status, _, body = self.request("GET", "/health")
        self.assertEqual((status, json.loads(body)), (200, {"tts": "ready", "stt": "ready"}))

    def test_unknown_paths_and_no_path_is_ever_a_file(self):
        for path in ("/", "/token", "/../token", "/health/../token", "/tts?file=/etc/passwd"):
            self.assertEqual(self.request("GET", path)[0], 404, path)
        self.assertEqual(self.request("POST", "/token", b"{}", headers={"Content-Type": "application/json"})[0], 404)

    def test_tts_body_rules(self):
        self.assertEqual(self.tts("")[0], 400)
        self.assertEqual(self.tts("\x00\x01 ")[0], 400)
        self.assertEqual(self.tts("x" * 301)[0], 413)
        self.assertEqual(self.request("POST", "/tts", b"not json")[0], 400)
        self.assertEqual(self.request("POST", "/tts", b'{"text": 5}')[0], 400)
        self.assertEqual(self.request("POST", "/tts", b"x" * (s.MAX_TTS_BODY + 1))[0], 413)
        self.assertEqual(self.tts("x" * 300)[0], 200)

    def test_stt_body_rules(self):
        self.assertEqual(self.stt(kind="application/json")[0], 415)
        self.assertEqual(self.stt(kind="audio/mpeg")[0], 415)
        self.assertEqual(self.stt(audio=WAV, kind="audio/webm")[0], 415)  # The bytes say wav.
        self.assertEqual(self.stt(audio=b"")[0], 400)
        self.assertEqual(self.stt(vocabulary="naïve")[0], 400)
        self.assertEqual(self.stt(vocabulary="x" * 1001)[0], 400)
        with unittest.mock.patch.object(s, "MAX_STT_BODY", 1000):  # Small, so the refused body fits a buffer.
            self.assertEqual(self.stt(audio=b"\x1aE\xdf\xa3" + b"\x00" * 1000)[0], 413)

    def test_chunked_upload_is_refused(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest("POST", "/stt")
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Content-Type", "audio/webm")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        connection.send(b"4\r\n\x1aE\xdf\xa3\r\n0\r\n\r\n")
        self.assertEqual(connection.getresponse().status, 411)
        connection.close()

    def test_a_slow_client_is_dropped(self):
        started = time.monotonic()
        with socket.create_connection(("127.0.0.1", self.port), timeout=s.SOCKET_TIMEOUT + 5) as sock:
            sock.sendall(b"POST /tts HTTP/1.1\r\nHost: x\r\n")
            data = sock.recv(100)
        self.assertEqual(data, b"")
        self.assertLess(time.monotonic() - started, s.SOCKET_TIMEOUT + 3)

    def test_the_log_has_no_text(self):
        self.tts("A very private sentence.")
        self.stt(vocabulary="secret-repo-name")
        self.assertNotIn("private", self.log.text())
        self.assertNotIn("secret", self.log.text())
        self.assertIn("POST /tts 200", self.log.text())


class OtherClientTests(FrontCase):
    CLIENTS = frozenset({"100.64.0.2"})  # Not 127.0.0.1, so this test's own connection is someone else.

    def test_dropped_before_anything_is_read(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            try:
                sock.sendall(f"GET /health HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n".encode())
                data = sock.recv(100)
            except ConnectionResetError:
                data = b""
        self.assertEqual(data, b"")


class SpeakTests(FrontCase):
    ENV = {"FAKE_GAP": "0.6"}

    def test_the_first_sentence_comes_first_then_the_rest_in_order(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(connection.close)
        started = time.monotonic()
        connection.request("POST", "/tts", json.dumps({"text": "One. Two. Three."}).encode(),
                           headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual((response.status, response.getheader("Content-Type"), response.getheader("Transfer-Encoding")),
                         (200, "audio/mpeg", "chunked"))
        first = response.read1(1000)
        self.assertEqual(first, b"[One.]")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(first + response.read(), b"[One.][Two.][Three.]")
        self.assertGreater(time.monotonic() - started, 1.1)

    def test_a_client_that_hangs_up_leaves_the_pipe_clean(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", "/tts", json.dumps({"text": "One. Two. Three."}).encode(),
                           headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
        response = connection.getresponse()
        response.read1(100)
        connection.close()
        status, _, body = self.tts("Again.")
        self.assertEqual((status, body), (200, b"[Again.]"))
        self.assertEqual(self.workers["tts"].state, "ready")

    def test_one_at_a_time_and_a_short_line(self):
        results = []

        def call():
            results.append(self.tts("One. Two. Three.")[0])
        threads = [threading.Thread(target=call) for _ in range(3 + s.MAX_WAITING)]
        for thread in threads:
            thread.start()
            time.sleep(0.05)
        for thread in threads:
            thread.join(20)
        self.assertIn(503, results)  # The fourth and fifth find the line full; later ones wait past `queue`.
        self.assertIn(200, results)

    def test_stt_passes_the_vocabulary_and_type(self):
        status, _, body = self.stt(vocabulary="Branches: dev, main.")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"text": "heard 104 bytes of webm; vocabulary: Branches: dev, main."})
        self.assertEqual(json.loads(self.stt(audio=WAV, kind="audio/wav")[2])["text"][:22], "heard 56 bytes of wav;")


class FailingWorkerTests(FrontCase):
    MODES = {"tts": "hang", "stt": "fail"}

    def test_a_hung_worker_is_timed_out_and_started_again(self):
        started = time.monotonic()
        self.assertEqual(self.tts("Hi.")[0], 504)
        self.assertLess(time.monotonic() - started, FAST["first"] + 2)
        self.assertTrue(wait_for(lambda: self.workers["tts"].state == "ready"))
        self.assertIn("tts worker restarted (too slow)", self.log.text())

    def test_a_worker_error_is_passed_on_as_a_status(self):
        status, _, body = self.stt()
        self.assertEqual((status, json.loads(body)), (500, {"error": "fake failure"}))
        self.assertEqual(self.workers["stt"].state, "ready")


class CrashTests(FrontCase):
    MODES = {"tts": "garbage", "stt": "ok"}

    def setUp(self):
        self.crash_file = tempfile.NamedTemporaryFile(delete=False)
        self.crash_file.close()
        self.ENV = {"FAKE_CRASH_FILE": self.crash_file.name}
        super().setUp()
        self.addCleanup(lambda: os.path.exists(self.crash_file.name) and os.unlink(self.crash_file.name))

    def test_a_crash_is_502_then_it_comes_back(self):
        self.assertEqual(self.stt()[0], 502)
        self.assertTrue(wait_for(lambda: self.workers["stt"].state == "ready"))
        self.assertEqual(self.stt()[0], 200)

    def test_garbage_is_502_and_a_restart(self):
        self.assertEqual(self.tts("Hi.")[0], 502)
        self.assertIn("tts worker restarted", self.log.text())


class StartingTests(FrontCase):
    MODES = {"tts": "never-ready", "stt": "slow-start"}
    ENV = {"FAKE_DELAY": "1.5"}

    def test_not_ready_yet_is_503_and_health_says_so(self):
        self.assertEqual(json.loads(self.request("GET", "/health")[2]), {"tts": "starting", "stt": "starting"})
        self.assertEqual(self.tts("Hi.")[0], 503)
        self.assertTrue(wait_for(lambda: self.workers["stt"].state == "ready"))
        self.assertEqual(self.stt()[0], 200)

    def test_stop_ends_every_worker_process(self):
        procs = []
        self.assertTrue(wait_for(lambda: self.workers["stt"].proc is not None))
        procs.append(self.workers["stt"].proc)
        for worker in self.workers.values():
            worker.stop()
        self.assertTrue(all(proc.poll() is not None for proc in procs))


if __name__ == "__main__":
    unittest.main()

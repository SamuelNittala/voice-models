"""Stands in for tts_worker.py or stt_worker.py in test_server.py: the same frames, no models.

Usage: fake_worker.py tts|stt. FAKE_MODE: ok, slow-start, never-ready, crash, hang, garbage, fail. FAKE_GAP seconds
between sentences. FAKE_CRASH_FILE: crash once on a request while that file exists (it is removed first).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as p  # noqa: E402


def main():
    kind = sys.argv[1]
    mode = os.environ.get("FAKE_MODE", "ok")
    out, reader = p.worker_pipes()
    p.watch_parent()
    print("a library printing to stdout, which must not break a frame")
    if mode == "never-ready":
        time.sleep(3600)
    if mode == "slow-start":
        time.sleep(float(os.environ.get("FAKE_DELAY", "1")))
    p.write_frame(out, b"R", {})
    while True:
        try:
            _kind, payload = reader.read()
        except EOFError:
            return
        request = p.json_payload(payload)
        audio = reader.read()[1] if kind == "stt" else b""
        crash_file = os.environ.get("FAKE_CRASH_FILE")
        if mode == "crash" or (crash_file and os.path.exists(crash_file)):
            if crash_file and os.path.exists(crash_file):
                os.unlink(crash_file)
            os._exit(3)
        if mode == "hang":
            time.sleep(3600)
        if mode == "garbage":
            out.write(b"\x00junk" * 10)
            continue
        if mode == "fail":
            p.write_frame(out, b"X", {"status": 500, "error": "fake failure"})
            continue
        if kind == "tts":
            for index, piece in enumerate(p.sentences(request["text"])):
                if index:
                    time.sleep(float(os.environ.get("FAKE_GAP", "0")))
                p.write_frame(out, b"A", f"[{piece}]".encode())
            p.write_frame(out, b"E", {})
        else:
            words = f"heard {len(audio)} bytes of {request['family']}; vocabulary: {request['vocabulary']}"
            p.write_frame(out, b"E", {"text": words})


if __name__ == "__main__":
    main()

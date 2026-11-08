"""Chatterbox Turbo text to speech for the speech front (server.py), one request at a time, as frames over pipes.

Chatterbox has no streaming, so the text is split into sentences: the first is made and encoded first, then the rest
in order. Every sentence goes through one ffmpeg MP3 encoder per request (24 kHz mono, constant bit rate, no Xing or
ID3 header), so the whole answer is one MP3 stream that plays without a join. A short silence goes between
sentences. Chatterbox puts its Perth watermark on every clip it makes.
"""
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as p  # noqa: E402

BIT_RATE = "64k"
GAP_SECONDS = 0.15
WARM_UP = "Warming up the voice now. This sentence is never sent anywhere."


def encoder(sample_rate):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    return subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
         "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", BIT_RATE, "-ar", str(sample_rate), "-ac", "1",
         "-write_xing", "0", "-id3v2_version", "0", "-flush_packets", "1", "-f", "mp3", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def speak(model, text, out, lock):
    import numpy as np

    pieces = p.sentences(p.clean_text(text)[:p.MAX_TTS_CHARS])
    if not pieces:
        p.write_frame(out, b"X", {"status": 400, "error": "no text"}, lock)
        return
    started = time.monotonic()
    mp3 = encoder(model.sr)
    sent = []

    def pump():  # MP3 bytes go out as soon as the encoder has them.
        while True:
            chunk = os.read(mp3.stdout.fileno(), 65536)
            if not chunk:
                return
            sent.append(len(chunk))
            p.write_frame(out, b"A", chunk, lock)
    pumping = threading.Thread(target=pump, name="mp3-pump", daemon=True)
    pumping.start()
    gap = np.zeros(int(model.sr * GAP_SECONDS), dtype=np.float32).tobytes()
    try:
        for index, piece in enumerate(pieces):
            wav = model.generate(piece)
            samples = np.clip(wav.squeeze().detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
            mp3.stdin.write((gap if index else b"") + samples.tobytes())
            mp3.stdin.flush()
        mp3.stdin.close()
        pumping.join(30)
        code = mp3.wait(30)
    except Exception as error:  # noqa: BLE001 - any failure is one short error frame, never the text.
        print(f"tts: generation failed ({type(error).__name__})", file=sys.stderr, flush=True)
        mp3.kill()
        mp3.wait(5)
        pumping.join(5)
        p.write_frame(out, b"X", {"status": 500, "error": "Chatterbox could not make the audio"}, lock)
        return
    if code != 0 or not sent:
        p.write_frame(out, b"X", {"status": 500, "error": "the MP3 encoder failed"}, lock)
        return
    print(f"tts: {len(pieces)} sentences, {sum(sent)} bytes, {time.monotonic() - started:.2f} s",
          file=sys.stderr, flush=True)
    p.write_frame(out, b"E", {}, lock)


def main():
    out, reader = p.worker_pipes()
    p.watch_parent()
    lock = threading.Lock()
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    started = time.monotonic()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    model.generate(WARM_UP)  # The first generate() takes about 8 s, so it is done before any request.
    torch.cuda.synchronize()
    print(f"tts: loaded and warmed up in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)
    p.write_frame(out, b"R", {"sample_rate": model.sr}, lock)
    while True:
        try:
            kind, payload = reader.read()
        except (EOFError, p.FrameError):
            return  # The front went away or the pipe is out of step: it starts a fresh worker.
        try:
            request = p.json_payload(payload) if kind == b"Q" else None
        except p.FrameError:
            request = None
        if not request or not isinstance(request.get("text"), str):
            p.write_frame(out, b"X", {"status": 400, "error": "bad request"}, lock)
            continue
        speak(model, request["text"], out, lock)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

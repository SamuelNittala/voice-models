"""faster-whisper large-v3-turbo speech to text for the speech front (server.py), one request at a time.

The recording arrives as bytes (webm/opus from Android Chrome, or ogg, mp4/m4a, wav), is decoded in memory by PyAV
and never written to disk. English only. The vocabulary from the gateway (branch and repo names) is Whisper's
initial prompt, cut by whole comma-separated items from the end to the 223 tokens Whisper keeps.
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as p  # noqa: E402

MODEL = "large-v3-turbo"
SAMPLE_RATE = 16000
MAX_SECONDS = 35


def main():
    out, reader = p.worker_pipes()
    p.watch_parent()
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    started = time.monotonic()
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", beam_size=5)[0])

    def tokens(text):
        return len(model.hf_tokenizer.encode(text, add_special_tokens=False).ids)
    print(f"stt: loaded and warmed up in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)
    p.write_frame(out, b"R", {})
    while True:
        try:
            kind, payload = reader.read()
            body_kind, audio = reader.read(time.monotonic() + 10)
        except (EOFError, TimeoutError, p.FrameError):
            return  # The front went away or the pipe is out of step: it starts a fresh worker.
        try:
            request = p.json_payload(payload) if kind == b"Q" and body_kind == b"B" else None
        except p.FrameError:
            request = None
        if not request:
            p.write_frame(out, b"X", {"status": 400, "error": "bad request"})
            continue
        began = time.monotonic()
        try:
            samples = decode_audio(io.BytesIO(audio), sampling_rate=SAMPLE_RATE)
        except Exception:  # noqa: BLE001 - PyAV raises many kinds; any of them means unreadable audio.
            p.write_frame(out, b"X", {"status": 415, "error": "the recording could not be decoded"})
            continue
        seconds = len(samples) / SAMPLE_RATE
        if seconds > MAX_SECONDS:
            p.write_frame(out, b"X", {"status": 413, "error": "the recording is too long"})
            continue
        vocabulary = request.get("vocabulary") if isinstance(request.get("vocabulary"), str) else ""
        prompt = p.fit_vocabulary(vocabulary, tokens) if vocabulary else None
        try:
            segments, _info = model.transcribe(samples, language="en", beam_size=5, initial_prompt=prompt)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as error:  # noqa: BLE001
            print(f"stt: transcription failed ({type(error).__name__})", file=sys.stderr, flush=True)
            p.write_frame(out, b"X", {"status": 500, "error": "Whisper could not transcribe the recording"})
            continue
        kept = len(prompt.split(", ")) if prompt else 0
        print(f"stt: {seconds:.1f} s of audio in {time.monotonic() - began:.2f} s, prompt items {kept}",
              file=sys.stderr, flush=True)
        p.write_frame(out, b"E", {"text": text})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

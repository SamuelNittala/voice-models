"""Chatterbox Turbo + Whisper large-v3-turbo (float16) loaded in one process.

Run: ~/voice-models/run.sh python bench/both_bench.py
Measures total GPU memory and repeats the TTS / STT timings to check for slowdown.
Also checks the Perth watermark is detectable in a generated clip.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import REQUESTS, SUMMARIES, VOCAB_PROMPT, Timer, gpu_used_mib, proc_gpu_mib  # noqa: E402

OUT = Path(__file__).parent / "out"
RUNS = 3


def main():
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from faster_whisper import WhisperModel
    import perth

    res = {"gpu_total_before": gpu_used_mib()}
    tts = ChatterboxTurboTTS.from_pretrained(device="cuda")
    stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    res["proc_gpu_mib_both_loaded"] = proc_gpu_mib()
    tts.generate("Warming up the model now.")
    list(stt.transcribe(str(OUT / "clips" / "r1.wav"), language="en")[0])

    tts_rows = {}
    for key, text in SUMMARIES.items():
        times = []
        for _ in range(RUNS):
            with Timer() as t:
                wav = tts.generate(text)
            times.append(t.s)
        tts_rows[key] = round(statistics.median(times), 2)
    res["tts_gen_s_median"] = tts_rows

    stt_rows = {}
    for key in REQUESTS:
        times = []
        for _ in range(RUNS):
            with Timer() as t:
                segs, _ = stt.transcribe(str(OUT / "clips" / f"{key}.wav"), language="en",
                                         beam_size=5, initial_prompt=VOCAB_PROMPT)
                " ".join(s.text for s in segs)
            times.append(t.s)
        stt_rows[key] = round(statistics.median(times), 3)
    res["stt_prompt_s_median"] = stt_rows
    res["proc_gpu_mib_peak"] = proc_gpu_mib()
    res["gpu_total_peak"] = gpu_used_mib()

    # Watermark check: is the Perth mark detectable in a generated clip?
    wav = tts.generate(SUMMARIES["s120"]).squeeze(0).numpy()
    wm = perth.PerthImplicitWatermarker()
    res["watermark_score_generated"] = float(wm.get_watermark(wav, sample_rate=tts.sr))
    import librosa
    ref, sr = librosa.load(str(OUT / "clips" / "r1.wav"), sr=None)
    res["watermark_score_saved_wav"] = float(wm.get_watermark(ref, sample_rate=sr))
    noise = (torch.randn(tts.sr * 4) * 0.05).numpy()
    res["watermark_score_noise"] = float(wm.get_watermark(noise, sample_rate=tts.sr))

    (OUT / "both_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

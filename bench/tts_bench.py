"""Chatterbox Turbo benchmark: load time, GPU memory, generation latency.

Run: ~/voice-models/run.sh python bench/tts_bench.py
Writes WAVs to bench/out/tts/ and JSON results to bench/out/tts_results.json.
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from common import SUMMARIES, REQUESTS, Timer, proc_gpu_mib  # noqa: E402

OUT = Path(__file__).parent / "out"
(OUT / "tts").mkdir(parents=True, exist_ok=True)
(OUT / "clips").mkdir(parents=True, exist_ok=True)
RUNS = 3


def first_sentence(text):
    parts = re.split(r"(?<=[.!?;])\s+", text.strip())
    return parts[0]


def main():
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    res = {"gpu_mib_before": proc_gpu_mib()}
    with Timer() as t:
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        torch.cuda.synchronize()
    res["load_s"] = round(t.s, 2)
    res["gpu_mib_after_load_proc"] = proc_gpu_mib()
    res["torch_alloc_mib_after_load"] = round(torch.cuda.memory_allocated() / 2**20)
    sr = model.sr
    res["sample_rate"] = sr

    # warm-up
    with Timer() as t:
        model.generate("Warming up the model now.")
    res["warmup_s"] = round(t.s, 2)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for key, text in SUMMARIES.items():
        gen_times, wm_times, first_times = [], [], []
        wav = None
        for _ in range(RUNS):
            with Timer() as t:
                wav = model.generate(text)
            gen_times.append(t.s)
            # watermark cost on its own (generate() already applied it once)
            raw = wav.squeeze(0).numpy()
            with Timer() as tw:
                model.watermarker.apply_watermark(raw, sample_rate=sr)
            wm_times.append(tw.s)
            fs = first_sentence(text)
            with Timer() as tf:
                model.generate(fs)
            first_times.append(tf.s)
        dur = wav.shape[-1] / sr
        torchaudio.save(str(OUT / "tts" / f"{key}.wav"), wav, sr)
        med = statistics.median(gen_times)
        rows.append({
            "key": key,
            "chars": len(text),
            "audio_s": round(dur, 2),
            "gen_s_runs": [round(x, 2) for x in gen_times],
            "gen_s_median": round(med, 2),
            "rtf": round(med / dur, 3),
            "watermark_s_median": round(statistics.median(wm_times), 3),
            "first_sentence": fs,
            "first_sentence_chars": len(fs),
            "first_sentence_s_median": round(statistics.median(first_times), 2),
        })
        print(json.dumps(rows[-1]), flush=True)
    res["summaries"] = rows
    res["gpu_mib_peak_proc"] = proc_gpu_mib()
    res["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)

    # Synthesise the 8 request clips for the Whisper test (built-in voice).
    clip_rows = []
    for key, text in REQUESTS.items():
        with Timer() as t:
            wav = model.generate(text)
        torchaudio.save(str(OUT / "clips" / f"{key}.wav"), wav, sr)
        clip_rows.append({"key": key, "audio_s": round(wav.shape[-1] / sr, 2), "gen_s": round(t.s, 2)})
    res["request_clips"] = clip_rows

    (OUT / "tts_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("summaries",)}, indent=2))


if __name__ == "__main__":
    main()

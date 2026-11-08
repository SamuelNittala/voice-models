"""Whisper large-v3-turbo (faster-whisper) benchmark.

Run: ~/voice-models/run.sh python bench/stt_bench.py [float16|int8_float16]
Needs bench/out/clips/*.wav from tts_bench.py. Writes bench/out/stt_<compute>.json.
"""
import json
import re
import statistics
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from common import REQUESTS, SUMMARIES, REPOS, BRANCHES, VOCAB_PROMPT, Timer, proc_gpu_mib  # noqa: E402

OUT = Path(__file__).parent / "out"
RUNS = 3
MODEL = "large-v3-turbo"


def norm(s):
    """Lowercase, drop sentence punctuation, keep hyphens and dots inside tokens."""
    s = s.lower().strip()
    s = re.sub(r"[,!?;:\"]", " ", s)
    s = re.sub(r"\.(\s|$)", r" ", s)  # sentence-final dots, not 4.6.2
    return " ".join(s.split())


def loose(s):
    """Hyphens to spaces, for a lenient name match."""
    return " ".join(norm(s).replace("-", " ").split())


def names_in(text):
    t = norm(text)
    return [n for n in REPOS + BRANCHES if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", t)]


def name_hits(ref, hyp):
    wanted = names_in(ref)
    exact = [n for n in wanted if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w.-])", norm(hyp))]
    loose_hits = [n for n in wanted if loose(n) in loose(hyp)]
    return wanted, exact, loose_hits


def transcribe(model, path, prompt):
    segs, _ = model.transcribe(str(path), language="en", beam_size=5, initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segs).strip()


def run_set(model, items, prompt):
    rows = []
    for key, ref in items.items():
        path = OUT / ("clips" if key.startswith("r") else "tts") / f"{key}.wav"
        times, hyp = [], ""
        for _ in range(RUNS):
            with Timer() as t:
                hyp = transcribe(model, path, prompt)
            times.append(t.s)
        wanted, exact, lh = name_hits(ref, hyp)
        rows.append({
            "key": key, "ref": ref, "hyp": hyp,
            "wer": round(jiwer.wer(norm(ref), norm(hyp)), 3),
            "t_median_s": round(statistics.median(times), 3),
            "names": wanted, "names_exact": exact, "names_loose": lh,
        })
    return rows


def summarise(rows):
    refs = [norm(r["ref"]) for r in rows]
    hyps = [norm(r["hyp"]) for r in rows]
    n = sum(len(r["names"]) for r in rows)
    return {
        "corpus_wer": round(jiwer.wer(refs, hyps), 3),
        "corpus_wer_loose": round(jiwer.wer([loose(r["ref"]) for r in rows], [loose(r["hyp"]) for r in rows]), 3),
        "names_total": n,
        "names_exact": sum(len(r["names_exact"]) for r in rows),
        "names_loose": sum(len(r["names_loose"]) for r in rows),
        "t_median_s": round(statistics.median(r["t_median_s"] for r in rows), 3),
    }


def main():
    from faster_whisper import WhisperModel

    compute = sys.argv[1] if len(sys.argv) > 1 else "float16"
    res = {"compute_type": compute, "model": MODEL}
    with Timer() as t:
        model = WhisperModel(MODEL, device="cuda", compute_type=compute)
    res["load_s"] = round(t.s, 2)
    res["gpu_mib_after_load_proc"] = proc_gpu_mib()

    with Timer() as t:
        transcribe(model, OUT / "clips" / "r1.wav", None)
    res["warmup_s"] = round(t.s, 2)

    res["requests_no_prompt"] = run_set(model, REQUESTS, None)
    res["requests_prompt"] = run_set(model, REQUESTS, VOCAB_PROMPT)
    res["summaries_no_prompt"] = run_set(model, SUMMARIES, None)
    res["gpu_mib_peak_proc"] = proc_gpu_mib()
    res["summary"] = {
        "requests_no_prompt": summarise(res["requests_no_prompt"]),
        "requests_prompt": summarise(res["requests_prompt"]),
        "summaries_no_prompt": summarise(res["summaries_no_prompt"]),
    }
    (OUT / f"stt_{compute}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(("requests", "summaries"))}, indent=2))
    for k in ("requests_no_prompt", "requests_prompt"):
        print(f"--- {k}")
        for r in res[k]:
            print(f"{r['key']} wer={r['wer']} t={r['t_median_s']} exact={len(r['names_exact'])}/{len(r['names'])} | {r['hyp']}")


if __name__ == "__main__":
    main()

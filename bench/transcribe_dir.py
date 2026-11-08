"""Transcribe every audio file in a folder, with and without the repo/branch vocabulary prompt.

Usage (via transcribe_dir.sh): transcribe_dir.sh <folder> [compute_type]
Optional: <folder>/refs.tsv with lines "<filename>\t<what you actually said>" adds WER and
exact name checks. Results are printed and saved to <folder>/transcripts.json and .tsv.
"""
import json
import sys
import time
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from common import VOCAB_PROMPT  # noqa: E402
from stt_bench import name_hits, norm  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm", ".3gp", ".amr", ".mp4"}


def load_refs(folder):
    p = folder / "refs.tsv"
    if not p.exists():
        return {}
    refs = {}
    for line in p.read_text().splitlines():
        if "\t" in line:
            name, text = line.split("\t", 1)
            refs[name.strip()] = text.strip()
    return refs


def main():
    from faster_whisper import WhisperModel

    folder = Path(sys.argv[1]).expanduser().resolve()
    compute = sys.argv[2] if len(sys.argv) > 2 else "float16"
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"no audio files in {folder}")
    refs = load_refs(folder)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type=compute)

    rows = []
    for f in files:
        row = {"file": f.name}
        for label, prompt in (("plain", None), ("prompt", VOCAB_PROMPT)):
            t0 = time.perf_counter()
            segs, info = model.transcribe(str(f), language="en", beam_size=5, initial_prompt=prompt)
            text = " ".join(s.text.strip() for s in segs).strip()
            row[f"{label}_s"] = round(time.perf_counter() - t0, 3)
            row[label] = text
            row["audio_s"] = round(info.duration, 2)
            ref = refs.get(f.name)
            if ref:
                wanted, exact, _ = name_hits(ref, text)
                row[f"{label}_wer"] = round(jiwer.wer(norm(ref), norm(text)), 3)
                row[f"{label}_names"] = f"{len(exact)}/{len(wanted)}"
        if f.name in refs:
            row["ref"] = refs[f.name]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    (folder / "transcripts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    cols = ["file", "audio_s", "plain_s", "plain", "prompt_s", "prompt"]
    if refs:
        cols += ["plain_wer", "prompt_wer", "plain_names", "prompt_names", "ref"]
    lines = ["\t".join(cols)] + ["\t".join(str(r.get(c, "")) for c in cols) for r in rows]
    (folder / "transcripts.tsv").write_text("\n".join(lines) + "\n")
    if refs:
        scored = [r for r in rows if "ref" in r]
        for label in ("plain", "prompt"):
            w = jiwer.wer([norm(r["ref"]) for r in scored], [norm(r[label]) for r in scored])
            print(f"{label}: corpus WER {w:.3f} over {len(scored)} files")
    print(f"saved {folder / 'transcripts.tsv'}")


if __name__ == "__main__":
    main()

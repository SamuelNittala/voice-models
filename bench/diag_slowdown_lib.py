import statistics, time
from pathlib import Path
from common import VOCAB_PROMPT
OUT = Path(__file__).parent / "out"
def bench(stt, label):
    ts = []
    for k in ["r1", "r3", "r5", "r7"] * 2:
        t0 = time.perf_counter()
        segs, _ = stt.transcribe(str(OUT / "clips" / f"{k}.wav"), language="en", beam_size=5, initial_prompt=VOCAB_PROMPT)
        " ".join(s.text for s in segs)
        ts.append(time.perf_counter() - t0)
    print(f"{label}: median {statistics.median(ts):.3f}s", flush=True)

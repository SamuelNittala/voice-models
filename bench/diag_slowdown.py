"""Diagnose why Whisper is slower when Chatterbox is loaded in the same process."""
import statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
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

from faster_whisper import WhisperModel
stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
bench(stt, "whisper alone")
import torch
torch.zeros(1).cuda()
bench(stt, "after torch CUDA init")
from chatterbox.tts_turbo import ChatterboxTurboTTS
tts = ChatterboxTurboTTS.from_pretrained(device="cuda")
bench(stt, "after chatterbox loaded")
tts.generate("Warming up the model now.")
bench(stt, "after one chatterbox generate")
torch.cuda.empty_cache()
bench(stt, "after empty_cache")
print("torch threads", torch.get_num_threads())
torch.set_num_threads(1)
bench(stt, "torch threads=1")

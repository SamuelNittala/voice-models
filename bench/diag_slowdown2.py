"""Second slowdown check: import-only, and Whisper in its own process next to an idle Chatterbox process."""
import statistics, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from diag_slowdown_lib import bench
mode = sys.argv[1]
from faster_whisper import WhisperModel
stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
bench(stt, "whisper alone")
if mode == "import":
    import chatterbox.tts_turbo  # noqa
    bench(stt, "after importing chatterbox (no model)")
elif mode == "sidecar":
    p = subprocess.Popen([sys.executable, "-c", "from chatterbox.tts_turbo import ChatterboxTurboTTS as T; import time,sys; m=T.from_pretrained(device=\"cuda\"); m.generate(\"Warm up.\"); print(\"ready\", flush=True); time.sleep(120)"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    while p.stdout.readline().strip() != "ready":
        pass
    bench(stt, "chatterbox idle in a separate process")
    p.kill(); p.wait()

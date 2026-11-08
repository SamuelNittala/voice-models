# voice-models

A speech service for the voice gateway: Chatterbox Turbo text to speech and faster-whisper (large-v3-turbo) speech
to text, behind a small standard-library HTTP front. Each model runs in its own worker process that the front starts,
warms up and restarts if it fails.

## Requirements

- Linux with an NVIDIA GPU and driver
- [uv](https://docs.astral.sh/uv/)
- ffmpeg (used to encode the MP3 answers)

The scripts expect the repo at `~/voice-models` and keep every cache, model and the virtual environment inside it
(see `env.sh`). On NixOS, `run.sh` points the library path at the venv's NVIDIA libraries and the driver so the
wheels work through nix-ld; see the comments there.

## Setup

```sh
git clone <this repo> ~/voice-models
cd ~/voice-models
source env.sh && uv sync
cp speech/local.env.example speech/local.env   # then set your own addresses
```

The models are downloaded on first use (the workers run with the Hugging Face hub offline, so fetch them once first,
for example by running `bench/tts_bench.py` and `bench/stt_bench.py` through `run.sh`).

## Settings

Both are required; the service refuses to start with a one-line error if either is unset or invalid.
`start.sh` reads them from `speech/local.env`, which is not committed.

| Variable                 | Meaning                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| `SPEECH_BIND_IP`         | Address to listen on: this machine's tailnet address (100.64.0.0/10) or 127.0.0.1 |
| `SPEECH_ALLOWED_CLIENTS` | Comma separated IPv4 addresses allowed to connect; 127.0.0.1 is always allowed  |

It never binds 0.0.0.0, drops other clients before reading anything, and requires `Authorization: Bearer <token>`
on every request. The token is made on first start in `speech/token` (mode 0600); copy it to the client.

## Run

```sh
~/voice-models/speech/start.sh                   # listens on SPEECH_BIND_IP:8790
~/voice-models/speech/start.sh --bind 127.0.0.1  # for an SSH tunnel: ssh -N -L 8790:127.0.0.1:8790 <host>
```

It runs in the foreground; Ctrl+C stops the front and both workers.

## Start at boot (NixOS)

`nixos/voice-models.nix` is a module that runs the service as a system service. It starts after tailscale and
restarts on failure.

```nix
imports = [ /path/to/voice-models/nixos/voice-models.nix ];
services.voice-models = { enable = true; user = "alice"; };  # dir defaults to /home/alice/voice-models
```

Then `sudo nixos-rebuild switch`. Check with `systemctl status voice-models` and `journalctl -u voice-models -f`.
It needs `speech/local.env`, the models downloaded once, and nix-ld enabled.

## Endpoints

- `GET /health`: each worker's state (starting, ready, busy, restarting or down).
- `POST /tts`: JSON `{"text": "..."}`, 1-300 characters. Answers a chunked `audio/mpeg` stream, first sentence first.
- `POST /stt`: the recording as the body (at most 2 MiB; webm, ogg, mp4/m4a or wav) and an optional `X-Vocabulary`
  header used as Whisper's prompt. Answers `{"text": "..."}`.

## Tests

The tests use fake workers, so no GPU or models are needed; any Python 3.11+ works:

```sh
python3 -m unittest discover -s speech -p 'test_*.py'
```

`bench/` holds the latency and accuracy benchmarks; they write results to `bench/out/` (not committed).

## License

MIT

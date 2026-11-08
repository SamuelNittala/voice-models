#!/usr/bin/env bash
# Transcribe every audio file in a folder with Whisper large-v3-turbo, with and without
# the repo/branch vocabulary prompt.
#   ~/voice-models/bench/transcribe_dir.sh ~/voice-models/recordings [float16|int8_float16]
# Optional refs.tsv in the folder ("<filename><TAB><what you said>") adds WER and name checks.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 <folder> [float16|int8_float16]" >&2
  exit 1
fi
DIR="$(cd "$1" && pwd)"
export HF_HUB_OFFLINE=1
exec "$HOME/voice-models/run.sh" python bench/transcribe_dir.py "$DIR" "${2:-float16}"

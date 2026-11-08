#!/usr/bin/env bash
# Runs the speech service in the foreground: the HTTP front and its two model workers (Chatterbox, Whisper).
# Ctrl+C stops all of them. Nothing starts it automatically: it runs only while this terminal does.
#
# Settings come from ~/voice-models/speech/local.env (not committed; copy local.env.example and edit it):
#   SPEECH_BIND_IP          this machine's tailnet address (100.64.0.0/10), or 127.0.0.1
#   SPEECH_ALLOWED_CLIENTS  comma separated client addresses; 127.0.0.1 is always allowed
# The service refuses to start if either is unset or invalid.
#
# Usage: ~/voice-models/speech/start.sh                  binds SPEECH_BIND_IP:8790
#        ~/voice-models/speech/start.sh --bind 127.0.0.1  for an SSH tunnel from the client instead:
#            ssh -N -L 8790:127.0.0.1:8790 <this-host>   (on the client; then VOICE_GATEWAY_SPEECH_URL=http://127.0.0.1:8790)
# The token the client needs is made on first start in ~/voice-models/speech/token (0600).
set -euo pipefail
LOCAL_ENV="$HOME/voice-models/speech/local.env"
if [ -f "$LOCAL_ENV" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$LOCAL_ENV"
  set +a
fi
exec "$HOME/voice-models/run.sh" python "$HOME/voice-models/speech/server.py" "$@"

# Tasks

Spec: publish the speech service as a public repo. Source only (no secrets, caches, audio or results); no
machine-specific addresses or host names; work-specific repo names replaced with generic ones.

- `SPEECH_BIND_IP`: required; an IPv4 address that is 127.0.0.1 or in 100.64.0.0/10. The server binds only this
  address or 127.0.0.1, never 0.0.0.0.
- `SPEECH_ALLOWED_CLIENTS`: required; comma separated IPv4 addresses; 127.0.0.1 is always allowed. Other clients are
  dropped before anything is read.
- Either unset or invalid: the server exits with a one-line `speech: ...` error naming the variable, before the
  token is made or anything starts.
- Unchanged: bearer token compared in constant time; token file must be a 0600 regular file of this user.
- `start.sh` reads both from the untracked `speech/local.env` (`speech/local.env.example` is committed).

## Current

- [ ] Review the working tree, then create the GitHub repo, commit and push (left to the owner)

## Completed

- [x] NixOS module nixos/voice-models.nix (system service, after tailscale) and README section; not built on a NixOS machine

- [x] Copy source only from the host (speech/, bench scripts, refs example, run.sh, env.sh, pyproject.toml, uv.lock)
- [x] Tests first for the two settings (bind validation takes the configured address, allowlist uses configured
      clients, refusal when unset or invalid); seen failing (5 of 31)
- [x] Implement the settings in speech/server.py; all 31 tests pass
- [x] start.sh reads speech/local.env; usage comments name no real hosts or addresses; local.env.example added
- [x] Replace work and client repo names in bench/common.py and recordings/refs.tsv.example
- [x] Add LICENSE (MIT), README.md, .gitignore
- [x] `git init` with branch main; nothing staged or committed
- [x] Audit for addresses, host names, work names and secrets

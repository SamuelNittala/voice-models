# NixOS module: runs the speech service at boot as a system service.
#
#   imports = [ /path/to/voice-models/nixos/voice-models.nix ];
#   services.voice-models = { enable = true; user = "alice"; };
#
# Needs speech/local.env (SPEECH_BIND_IP, SPEECH_ALLOWED_CLIENTS) and the models downloaded once beforehand,
# and nix-ld enabled (run.sh relies on it). See the README.
{ config, lib, pkgs, ... }:
let
  cfg = config.services.voice-models;
in
{
  options.services.voice-models = {
    enable = lib.mkEnableOption "the voice-models speech service (Whisper STT and Chatterbox TTS)";
    user = lib.mkOption {
      type = lib.types.str;
      description = "Account that owns the checkout and runs the service. systemd sets its HOME, which the scripts use.";
    };
    dir = lib.mkOption {
      type = lib.types.str;
      default = "/home/${cfg.user}/voice-models";
      defaultText = lib.literalExpression ''"/home/''${config.services.voice-models.user}/voice-models"'';
      description = "Path of the checkout. The scripts expect it at ~/voice-models.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.voice-models = {
      description = "Voice models speech service (Whisper STT and Chatterbox TTS)";
      wantedBy = [ "multi-user.target" ];
      # The service binds the tailnet address, so tailscale must be up first.
      after = [ "network-online.target" "tailscaled.service" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.bash pkgs.coreutils ]; # start.sh and run.sh use #!/usr/bin/env bash
      serviceConfig = {
        User = cfg.user;
        SupplementaryGroups = [ "video" "render" ]; # GPU access
        ExecStart = "${cfg.dir}/speech/start.sh";
        Restart = "always";
        RestartSec = 10; # retries until the tailnet address exists
      };
    };
  };
}

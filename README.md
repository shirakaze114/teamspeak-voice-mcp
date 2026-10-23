# tsvoice_mcp

Headless TeamSpeak 3 voice bot that will expose per-user transcripts and voice tools over MCP.

## Status: phase 1 (TeamSpeak prototype)

- Connects with a persistent identity (`data/identity.json`, mode 0600) and joins a channel
- Splits incoming voice per client before mixing, resamples to 16 kHz mono, writes one WAV per utterance to `data/recordings/`
- Plays WAV files into the channel (resampled to 48 kHz, Opus 20 ms frames, paced at 50 fps)
- Sends channel text messages, including a transcription notice on join
- Reconnects with exponential backoff (2 s up to 60 s)

Next phases: Silero VAD, Qwen3-ASR-Flash-Realtime, MCP server (rmcp, Streamable HTTP), TTS adapter.

## Run

```sh
cp config.example.toml config.toml   # edit address/channel
cargo run --release -- -c config.toml
```

Console commands on stdin: `users`, `status`, `say <file.wav>`, `msg <text>`, `stop`.
Log level: `RUST_LOG=debug,tsproto=warn`.

Build requirements: Rust 1.88+ (edition 2024), libopus is built from source by `audiopus_sys` (needs cmake or pkg-config + libopus).

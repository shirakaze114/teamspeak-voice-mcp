//! Phase-1 prototype: connect to TeamSpeak, split voice per user, record each
//! utterance to WAV, and play WAV files / send text from a stdin console.

mod audio;
mod config;
mod ts;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use time::OffsetDateTime;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::audio::{ASR_RATE, TS_RATE};
use crate::config::{Config, RecordingConfig};
use crate::ts::{SpeakerInfo, TrackEvent, TsHandle};

#[derive(Parser, Debug)]
#[command(about = "TeamSpeak 3 voice bot (phase-1 prototype)")]
struct Args {
    /// Path to the TOML config file.
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tsproto=warn,tsproto::packet_codec=error,tsclientlib=warn".into()),
        )
        .init();

    let args = Args::parse();
    let cfg = Config::load(&args.config)?;

    let (ts, tracks, ts_task) = ts::spawn(cfg.teamspeak.clone())?;
    tokio::spawn(record_tracks(tracks, cfg.recording.clone()));
    tokio::spawn(console(ts.clone()));

    tokio::signal::ctrl_c().await?;
    info!("ctrl-c received");
    ts.shutdown().await.ok();
    ts_task.await.ok();
    Ok(())
}

struct Utterance {
    speaker: SpeakerInfo,
    started: OffsetDateTime,
    pcm: Vec<f32>,
}

/// Writes one WAV file per user utterance and logs the per-user boundaries.
async fn record_tracks(mut rx: mpsc::Receiver<TrackEvent>, cfg: RecordingConfig) {
    if cfg.enabled
        && let Err(e) = tokio::fs::create_dir_all(&cfg.dir).await
    {
        warn!("cannot create recording dir: {e}");
    }
    let mut active: HashMap<u16, Utterance> = HashMap::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            TrackEvent::Start { speaker, at } => {
                info!(client_id = speaker.client_id, name = %speaker.name, channel = speaker.channel_id, "▶ talking");
                active.insert(speaker.client_id, Utterance { speaker, started: at, pcm: Vec::new() });
            }
            TrackEvent::Audio { client_id, samples } => {
                if let Some(u) = active.get_mut(&client_id) {
                    u.pcm.extend_from_slice(&samples);
                }
            }
            TrackEvent::End { client_id, at } => {
                let Some(u) = active.remove(&client_id) else { continue };
                let ms = u.pcm.len() as u64 * 1000 / ASR_RATE as u64;
                let rms = rms(&u.pcm);
                info!(client_id, name = %u.speaker.name, ms, rms = format!("{rms:.4}"), "■ stopped");
                if !cfg.enabled || ms < cfg.min_utterance_ms as u64 {
                    continue;
                }
                let stamp = u.started.unix_timestamp_nanos() / 1_000_000;
                let path = cfg.dir.join(format!("{stamp}_c{}_{}.wav", client_id, sanitize(&u.speaker.name)));
                let _ = at;
                let pcm = u.pcm;
                let res = tokio::task::spawn_blocking({
                    let path = path.clone();
                    move || audio::write_wav_mono(&path, &pcm, ASR_RATE)
                })
                .await;
                match res {
                    Ok(Ok(())) => info!("saved {}", path.display()),
                    Ok(Err(e)) => warn!("saving wav failed: {e:#}"),
                    Err(e) => warn!("saving wav panicked: {e}"),
                }
            }
        }
    }
}

fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32).sqrt()
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(32)
        .collect()
}

/// Minimal stdin console for manual testing.
async fn console(ts: TsHandle) {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    println!("commands: users | status | say <file.wav> | msg <text> | stop");
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
        let rest = rest.trim();
        let res: Result<()> = async {
            match cmd {
                "" => {}
                "users" => {
                    for u in ts.list_users().await? {
                        println!("{:>5}  {}{}{}", u.client_id, u.name,
                            if u.is_self { "  (bot)" } else { "" },
                            if u.talking { "  [talking]" } else { "" });
                    }
                }
                "status" => println!("{}", serde_json::to_string_pretty(&ts.status().await?)?),
                "say" => {
                    let path = PathBuf::from(rest);
                    let pcm = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
                        let (mono, rate) = audio::load_wav_mono(&path)?;
                        audio::resample_clip(&mono, rate, TS_RATE)
                    })
                    .await??;
                    println!("queued {:.1}s", pcm.len() as f32 / TS_RATE as f32);
                    ts.play(pcm).await?;
                }
                "msg" => ts.send_channel_message(rest.to_string()).await?,
                "stop" => ts.stop_playback().await?,
                other => println!("unknown command: {other}"),
            }
            Ok(())
        }
        .await;
        if let Err(e) = res {
            println!("error: {e:#}");
        }
    }
}

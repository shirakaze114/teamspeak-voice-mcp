//! TeamSpeak 3 voice bot: per-user VAD + ASR, MCP server and control panel.

mod app;
mod asr;
mod audio;
mod config;
mod mcp;
mod pipeline;
mod transcripts;
mod ts;
mod tts;
mod usage;
mod web;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::app::App;
use crate::config::Config;
use crate::pipeline::Pipeline;
use crate::transcripts::TranscriptStore;
use crate::usage::Usage;

#[derive(Parser, Debug)]
#[command(about = "TeamSpeak 3 voice bot with MCP server and control panel")]
struct Args {
    /// Path to the TOML config file.
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    /// Also enable the small stdin console (users/status/say/msg/stop/asr).
    #[arg(long)]
    console: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "info,tsproto=warn,tsproto::packet_codec=error,tsclientlib=warn".into()
            }),
        )
        .init();

    let args = Args::parse();
    let cfg = Config::load(&args.config)?;
    let shutdown = CancellationToken::new();

    let (ts, tracks, mut ts_task) = ts::spawn(cfg.teamspeak.clone())?;
    let usage = Arc::new(Usage::load(cfg.asr.budget.clone()));
    let asr = Arc::new(asr::Asr::new(cfg.asr.clone(), usage)?);
    let store = Arc::new(TranscriptStore::new(cfg.transcripts.clone()));
    let speaker = Arc::new(tts::Speaker::new(cfg.tts.clone(), ts.clone())?);
    if !speaker.available() {
        warn!("TTS not configured: ts_speak will fail until [tts] is set up");
    }
    if asr.state().enabled && !asr.state().has_api_key {
        warn!("DASHSCOPE_API_KEY is not set: transcription disabled");
    }

    let pipeline = Pipeline::new(
        cfg.vad.clone(),
        asr.clone(),
        store.clone(),
        ts.clone(),
        cfg.recording.clone(),
        cfg.asr.context_messages,
    )?;
    let pipeline_task = tokio::spawn(pipeline.run(tracks));

    let app = Arc::new(App {
        ts: ts.clone(),
        asr,
        store: store.clone(),
        tts: speaker,
        started: std::time::Instant::now(),
        tokens: cfg.server.tokens.clone(),
        auth_disabled: cfg.server.auth_disabled,
    });

    let mut web_task = tokio::spawn(web::serve(
        app.clone(),
        cfg.server.clone(),
        shutdown.clone(),
    ));
    let retention_task = tokio::spawn({
        let store = store.clone();
        async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                store.compact_file();
            }
        }
    });
    if args.console {
        tokio::spawn(console(ts.clone(), store.clone()));
    }

    tokio::signal::ctrl_c().await?;
    info!("ctrl-c received, shutting down");
    shutdown.cancel();
    ts.shutdown().await.ok();
    // A busy TS stream or a long-lived MCP/SSE connection must not prevent
    // Ctrl-C from terminating the process indefinitely.
    if tokio::time::timeout(std::time::Duration::from_secs(4), &mut ts_task)
        .await
        .is_err()
    {
        warn!("TS shutdown timed out; aborting connection task");
        ts_task.abort();
    }
    pipeline_task.abort();
    retention_task.abort();
    if tokio::time::timeout(std::time::Duration::from_secs(3), &mut web_task)
        .await
        .is_err()
    {
        warn!("HTTP shutdown timed out; aborting web task");
        web_task.abort();
    }
    Ok(())
}

/// Minimal stdin console for manual testing.
async fn console(ts: ts::TsHandle, store: Arc<TranscriptStore>) {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    println!("commands: users | status | say <audio file> | msg <text> | stop | transcripts");
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
        let rest = rest.trim();
        let res: Result<()> = async {
            match cmd {
                "" => {}
                "users" => {
                    for u in ts.list_users().await? {
                        println!(
                            "{:>5}  {}{}{}",
                            u.client_id,
                            u.name,
                            if u.is_self { "  (bot)" } else { "" },
                            if u.talking { "  [talking]" } else { "" }
                        );
                    }
                }
                "status" => println!("{}", serde_json::to_string_pretty(&ts.status().await?)?),
                "say" => {
                    let path = PathBuf::from(rest);
                    let pcm = tokio::task::spawn_blocking(move || audio::load_for_playback(&path))
                        .await??;
                    println!("queued {:.1}s", pcm.len() as f32 / audio::TS_RATE as f32);
                    ts.play(pcm).await?;
                }
                "msg" => ts.send_channel_message(rest.to_string()).await?,
                "stop" => ts.stop_playback().await?,
                "transcripts" => {
                    for e in store.query(&transcripts::Query {
                        limit: 10,
                        ..Default::default()
                    }) {
                        println!("[{}] {}: {}", e.timestamp, e.speaker.name, e.text);
                    }
                }
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

//! Text-to-speech via an existing OpenAI-compatible `/audio/speech` service.
//!
//! Text is split into sentences and synthesised one by one, so playback of
//! the first sentence starts while later ones are still being generated.
//! Speak requests are played in order; `stop` drops everything queued.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::audio::{CreateSpeechRequest, SpeechModel, SpeechResponseFormat, Voice};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::audio::{self, TS_RATE};
use crate::config::{TtsConfig, TtsFormat, TtsKind};
use crate::ts::TsHandle;

pub struct Speaker {
    cfg: TtsConfig,
    client: Option<Client<OpenAIConfig>>,
    ts: TsHandle,
    /// Bumped by `stop`; queued work from older generations is discarded.
    generation: AtomicU64,
    /// Serialises speak requests so sentences of different requests don't interleave.
    turn: Mutex<()>,
}

impl Speaker {
    pub fn new(cfg: TtsConfig, ts: TsHandle) -> Result<Self> {
        let client = match cfg.kind {
            TtsKind::None | TtsKind::Local => None,
            TtsKind::Openai => {
                let mut c = OpenAIConfig::new().with_api_base(cfg.base_url.trim_end_matches('/'));
                if let Some(k) = &cfg.api_key {
                    c = c.with_api_key(k);
                }
                let http = reqwest::Client::builder()
                    .timeout(Duration::from_secs(cfg.request_timeout_s))
                    .build()?;
                Some(Client::with_config(c).with_http_client(http))
            }
        };
        Ok(Self {
            cfg,
            client,
            ts,
            generation: AtomicU64::new(0),
            turn: Mutex::new(()),
        })
    }

    pub fn available(&self) -> bool {
        match self.cfg.kind {
            TtsKind::None => false,
            TtsKind::Openai => self.client.is_some(),
            TtsKind::Local => self
                .cfg
                .command
                .as_deref()
                .is_some_and(|c| !c.trim().is_empty()),
        }
    }

    /// Queue `text` for playback. Returns once accepted; synthesis runs in the background.
    pub fn speak(self: &Arc<Self>, text: &str, interrupt: bool) -> Result<usize> {
        if !self.available() {
            bail!("TTS is not configured; set [tts] kind = \"openai\" or \"local\"");
        }
        let text = text.trim();
        if text.is_empty() {
            bail!("text is empty");
        }
        if text.chars().count() > self.cfg.max_chars {
            bail!("text longer than {} characters", self.cfg.max_chars);
        }
        if interrupt {
            self.stop_now();
        }
        let sentences = split_sentences(text);
        let n = sentences.len();
        let me = self.clone();
        let generation = self.generation.load(Ordering::SeqCst);
        tokio::spawn(async move {
            let _turn = me.turn.lock().await;
            for s in sentences {
                if me.generation.load(Ordering::SeqCst) != generation {
                    return;
                }
                match me.synthesize(&s).await {
                    Ok(pcm) if me.generation.load(Ordering::SeqCst) == generation => {
                        if let Err(e) = me.ts.play(pcm).await {
                            warn!("queue playback failed: {e:#}");
                            return;
                        }
                    }
                    Ok(_) => return,
                    Err(e) => {
                        warn!("TTS failed for {s:?}: {e:#}");
                        return;
                    }
                }
            }
            // Keep the turn until playback drained, so the next request starts after it.
            me.ts.wait_idle().await;
        });
        info!(sentences = n, "speak queued");
        Ok(n)
    }

    fn stop_now(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        let ts = self.ts.clone();
        tokio::spawn(async move { ts.stop_playback().await });
    }

    pub async fn stop(&self) -> Result<()> {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.ts.stop_playback().await
    }

    /// Synthesize one sentence to 48 kHz mono PCM.
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let bytes = match self.cfg.kind {
            TtsKind::Openai => {
                let client = self
                    .client
                    .as_ref()
                    .ok_or_else(|| anyhow!("TTS not configured"))?;
                let req = CreateSpeechRequest {
                    input: text.to_string(),
                    model: SpeechModel::Other(self.cfg.model.clone()),
                    voice: Voice::Other(self.cfg.voice.clone()),
                    response_format: Some(match self.cfg.format {
                        TtsFormat::Wav => SpeechResponseFormat::Wav,
                        TtsFormat::Pcm => SpeechResponseFormat::Pcm,
                    }),
                    speed: self.cfg.speed,
                    ..Default::default()
                };
                client.audio().speech().create(req).await?.bytes.to_vec()
            }
            TtsKind::Local => self.run_local(text).await?,
            TtsKind::None => bail!("TTS not configured"),
        };
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let (mono, rate) = match cfg.format {
                TtsFormat::Pcm => (audio::from_s16le(&bytes), cfg.sample_rate),
                TtsFormat::Wav => audio::decode_mono(bytes)?,
            };
            let mut pcm = audio::resample_clip(&mono, rate, TS_RATE)?;
            if cfg.volume != 1.0 {
                pcm.iter_mut()
                    .for_each(|s| *s = (*s * cfg.volume).clamp(-1.0, 1.0));
            }
            Ok(pcm)
        })
        .await?
    }

    /// Run a local synthesizer without a shell, so text cannot inject commands.
    async fn run_local(&self, text: &str) -> Result<Vec<u8>> {
        let program = self
            .cfg
            .command
            .as_deref()
            .filter(|p| !p.trim().is_empty())
            .ok_or_else(|| anyhow!("[tts].command is required for local TTS"))?;
        let output_path =
            std::env::temp_dir().join(format!("tsvoice-tts-{}.wav", uuid::Uuid::new_v4()));
        let output_arg = output_path.to_string_lossy().into_owned();
        let mut command = Command::new(program);
        let mut has_text = false;
        let mut writes_file = false;
        for arg in &self.cfg.args {
            has_text |= arg.contains("{text}");
            writes_file |= arg.contains("{output}");
            command.arg(arg.replace("{text}", text).replace("{output}", &output_arg));
        }
        if !has_text {
            bail!("local TTS args must contain {{text}}");
        }
        let output = timeout(
            Duration::from_secs(self.cfg.request_timeout_s),
            command.output(),
        )
        .await
        .context("local TTS command timed out")??;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "local TTS exited with {}: {}",
                output.status,
                stderr.chars().take(300).collect::<String>()
            );
        }
        let bytes = if writes_file {
            let result = tokio::fs::read(&output_path).await;
            let _ = tokio::fs::remove_file(&output_path).await;
            result?
        } else {
            output.stdout
        };
        if bytes.is_empty() {
            bail!("local TTS returned no audio");
        }
        Ok(bytes)
    }
}

/// Split at sentence punctuation, merging short pieces up to ~60 characters.
fn split_sentences(text: &str) -> Vec<String> {
    const ENDS: &[char] = &['。', '！', '？', '!', '?', '；', ';', '\n', '…'];
    let mut parts = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if ENDS.contains(&c) || (c == '.' && cur.chars().count() > 1) {
            parts.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    let mut merged: Vec<String> = Vec::new();
    for p in parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
    {
        match merged.last_mut() {
            Some(last) if last.chars().count() + p.chars().count() <= 60 => last.push_str(&p),
            _ => merged.push(p),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::split_sentences;

    #[test]
    fn splits_chinese_sentences() {
        let text = "你好！今天吃什么？我推荐火锅。还有烧烤。明天去爬山。后天去看电影。你觉得怎么样？我们明天再详细讨论这个问题。".repeat(3);
        let s = split_sentences(&text);
        assert!(
            s.len() > 1,
            "long text should be split for streaming playback"
        );
        assert_eq!(s.concat(), text);
        assert!(s.iter().all(|part| !part.is_empty()));
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(
            split_sentences("你好！今天吃什么？"),
            vec!["你好！今天吃什么？"]
        );
    }
}

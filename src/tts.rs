//! Text-to-speech via an existing OpenAI-compatible `/audio/speech` service.
//!
//! Text is split into sentences and synthesised one by one, so playback of
//! the first sentence starts while later ones are still being generated.
//! Speak requests are played in order; `stop` drops everything queued.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::audio::{CreateSpeechRequest, SpeechModel, SpeechResponseFormat, Voice};
use tokio::sync::Mutex;
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
            TtsKind::None => None,
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
        Ok(Self { cfg, client, ts, generation: AtomicU64::new(0), turn: Mutex::new(()) })
    }

    pub fn available(&self) -> bool {
        self.client.is_some()
    }

    /// Queue `text` for playback. Returns once accepted; synthesis runs in the background.
    pub fn speak(self: &Arc<Self>, text: &str, interrupt: bool) -> Result<usize> {
        if self.client.is_none() {
            bail!("TTS is not configured ([tts] kind = \"none\")");
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
        let client = self.client.as_ref().ok_or_else(|| anyhow!("TTS not configured"))?;
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
        let bytes = client.audio().speech().create(req).await?.bytes.to_vec();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let (mono, rate) = match cfg.format {
                TtsFormat::Pcm => (audio::from_s16le(&bytes), cfg.sample_rate),
                TtsFormat::Wav => audio::decode_mono(bytes)?,
            };
            let mut pcm = audio::resample_clip(&mono, rate, TS_RATE)?;
            if cfg.volume != 1.0 {
                pcm.iter_mut().for_each(|s| *s = (*s * cfg.volume).clamp(-1.0, 1.0));
            }
            Ok(pcm)
        })
        .await?
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
    for p in parts.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()) {
        match merged.last_mut() {
            Some(last) if last.chars().count() + p.chars().count() <= 60 && merged.len() > 1 => last.push_str(&p),
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
        let s = split_sentences("你好！今天吃什么？我推荐火锅。还有烧烤");
        assert_eq!(s[0], "你好！");
        assert_eq!(s.concat(), "你好！今天吃什么？我推荐火锅。还有烧烤");
    }
}

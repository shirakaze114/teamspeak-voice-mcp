use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub teamspeak: TeamSpeakConfig,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub asr: AsrConfig,
    #[serde(default)]
    pub tts: TtsConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub transcripts: TranscriptConfig,
    #[serde(default)]
    pub recording: RecordingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamSpeakConfig {
    /// `host`, `host:port` or a TS3 server nickname.
    pub address: String,
    #[serde(default = "d_nickname")]
    pub nickname: String,
    /// Channel to join by id. Takes precedence over `channel`.
    pub channel_id: Option<u64>,
    /// Channel path such as `Lobby` or `Parent/Child`.
    pub channel: Option<String>,
    pub channel_password: Option<String>,
    pub server_password: Option<String>,
    #[serde(default = "d_identity_file")]
    pub identity_file: PathBuf,
    /// Channel message sent after joining. Empty string disables it.
    #[serde(default = "d_announcement")]
    pub announcement: String,
    /// Client nicknames or unique ids whose voice is never transcribed (e.g. music bots).
    #[serde(default)]
    pub ignore_clients: Vec<String>,
    #[serde(default)]
    pub log_commands: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct VadConfig {
    /// Speech probability that opens a segment (Silero).
    pub threshold: f32,
    /// Probability below which a frame counts as silence. Default: threshold - 0.15.
    pub neg_threshold: Option<f32>,
    /// Segments shorter than this are dropped (clicks, coughs, noise bursts).
    pub min_speech_ms: u32,
    /// Silence that ends a segment. Shorter pauses stay inside one utterance.
    pub silence_end_ms: u32,
    /// Audio kept before and after detected speech.
    pub pad_ms: u32,
    /// Hard cap; longer speech is split.
    pub max_segment_s: u32,
    /// Drop user audio while the bot is talking (loudspeaker echo guard).
    pub suppress_while_bot_speaking: bool,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: None,
            min_speech_ms: 300,
            silence_end_ms: 700,
            pad_ms: 200,
            max_segment_s: 30,
            suppress_while_bot_speaking: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrMode {
    /// qwen-audio-3.0-asr-flash: one HTTP request per finished utterance (cheaper).
    Http,
    /// qwen-audio-3.0-asr-flash-streaming: WebSocket while speaking, partial results.
    Streaming,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AsrConfig {
    pub enabled: bool,
    pub mode: AsrMode,
    /// Read from `api_key_file`, then `DASHSCOPE_API_KEY`, when unset.
    pub api_key: Option<String>,
    /// Optional local file containing only the API key (not committed).
    pub api_key_file: Option<PathBuf>,
    pub http_model: String,
    pub streaming_model: String,
    pub http_url: String,
    pub ws_url: String,
    pub language_hints: Vec<String>,
    /// Instant hot words (HTTP mode only): word -> weight 1..=5.
    pub hotwords: BTreeMap<String, u8>,
    /// Number of previous channel transcripts sent as context (HTTP mode).
    pub context_messages: usize,
    /// Server-side sentence split inside one streaming task.
    pub max_sentence_silence_ms: u32,
    pub request_timeout_s: u64,
    pub budget: BudgetConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: AsrMode::Http,
            api_key: None,
            api_key_file: None,
            http_model: "qwen-audio-3.0-asr-flash".into(),
            streaming_model: "qwen-audio-3.0-asr-flash-streaming".into(),
            http_url: "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation".into(),
            ws_url: "wss://dashscope.aliyuncs.com/api-ws/v1/inference".into(),
            language_hints: vec!["zh".into()],
            hotwords: BTreeMap::new(),
            context_messages: 0,
            max_sentence_silence_ms: 2000,
            request_timeout_s: 30,
            budget: BudgetConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BudgetConfig {
    /// Max billed audio seconds per local day across all models. 0 = unlimited.
    pub daily_limit_seconds: u64,
    /// Max billed audio seconds ever (tracked in `usage_file`). 0 = unlimited.
    pub total_limit_seconds: u64,
    pub usage_file: PathBuf,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_limit_seconds: 3600,
            total_limit_seconds: 0,
            usage_file: "data/usage.json".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsKind {
    None,
    /// OpenAI-compatible `POST {base_url}/audio/speech`.
    Openai,
    /// Run a local executable; arguments support `{text}` and `{output}`.
    #[serde(alias = "command")]
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsFormat {
    Wav,
    /// Raw s16le mono at `sample_rate`; streamed into the channel as it arrives.
    Pcm,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TtsConfig {
    pub kind: TtsKind,
    pub base_url: String,
    /// Read from `TTS_API_KEY` when unset.
    pub api_key: Option<String>,
    pub model: String,
    pub voice: String,
    pub format: TtsFormat,
    /// Sample rate of `pcm` responses.
    pub sample_rate: u32,
    pub speed: Option<f32>,
    pub max_chars: usize,
    pub volume: f32,
    pub request_timeout_s: u64,
    /// Executable for `kind = "local"` (for example `say`, `espeak-ng`, or `piper`).
    pub command: Option<String>,
    /// Arguments for the executable. `{text}` is replaced by the input text;
    /// `{output}` is replaced by a temporary WAV path. Without `{output}`,
    /// the command's stdout is used as audio bytes.
    pub args: Vec<String>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            kind: TtsKind::None,
            base_url: "http://127.0.0.1:8880/v1".into(),
            api_key: None,
            model: "tts-1".into(),
            voice: "alloy".into(),
            format: TtsFormat::Wav,
            sample_rate: 24_000,
            speed: None,
            max_chars: 500,
            volume: 1.0,
            request_timeout_s: 60,
            command: None,
            args: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Read status, users and transcripts.
    Read,
    /// Speak, stop speaking, send text messages.
    Speak,
    /// Everything, plus ASR controls in the panel.
    Admin,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    pub name: String,
    pub token: String,
    pub scopes: Vec<Scope>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub bind: String,
    /// Extra `Host` header values accepted by the MCP endpoint (DNS-rebinding guard).
    pub allowed_hosts: Vec<String>,
    pub tokens: Vec<TokenConfig>,
    /// Only honoured when bound to a loopback address.
    pub auth_disabled: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8787".into(),
            allowed_hosts: vec![],
            tokens: vec![],
            auth_disabled: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TranscriptConfig {
    pub retention_hours: u64,
    pub max_events: usize,
    /// Persist final transcripts to `file` (JSON lines) so they survive restarts.
    pub persist: bool,
    pub file: PathBuf,
}

impl Default for TranscriptConfig {
    fn default() -> Self {
        Self {
            retention_hours: 24,
            max_events: 5000,
            persist: false,
            file: "data/transcripts.jsonl".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RecordingConfig {
    /// Save every VAD segment as 16 kHz mono WAV (debugging aid).
    pub enabled: bool,
    pub dir: PathBuf,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: "data/recordings".into(),
        }
    }
}

fn d_nickname() -> String {
    "VoiceBot".into()
}
fn d_identity_file() -> PathBuf {
    "data/identity.json".into()
}
fn d_announcement() -> String {
    "[提示] 本频道的语音可能会被机器人转写并交由 AI 处理。".into()
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut cfg: Config =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        // Secrets may come from the environment instead of the file.
        if let Some(v) = env("TS_SERVER_PASSWORD") {
            cfg.teamspeak.server_password = Some(v);
        }
        if let Some(v) = env("TS_CHANNEL_PASSWORD") {
            cfg.teamspeak.channel_password = Some(v);
        }
        if cfg.asr.api_key.is_none()
            && let Some(key_file) = cfg.asr.api_key_file.clone()
        {
            let key = std::fs::read_to_string(&key_file)
                .with_context(|| format!("reading ASR key file {}", key_file.display()))?;
            let key = key.trim();
            if key.is_empty() {
                bail!("ASR key file {} is empty", key_file.display());
            }
            cfg.asr.api_key = Some(key.to_owned());
        }
        if cfg.asr.api_key.is_none() {
            cfg.asr.api_key = env("DASHSCOPE_API_KEY");
        }
        if cfg.tts.api_key.is_none() {
            cfg.tts.api_key = env("TTS_API_KEY");
        }
        if let Some(t) = env("TSVOICE_ADMIN_TOKEN") {
            cfg.server.tokens.push(TokenConfig {
                name: "env-admin".into(),
                token: t,
                scopes: vec![Scope::Admin],
            });
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        for t in &self.server.tokens {
            if t.token.len() < 16 {
                bail!("token '{}' is shorter than 16 characters", t.name);
            }
        }
        if self.server.tokens.is_empty() && !self.server.auth_disabled {
            bail!("no API tokens configured: add [[server.tokens]] or set TSVOICE_ADMIN_TOKEN");
        }
        if self.server.auth_disabled {
            let addr: std::net::SocketAddr = self
                .server
                .bind
                .parse()
                .context("server.bind must be ip:port")?;
            if !addr.ip().is_loopback() {
                bail!("server.auth_disabled is only allowed with a loopback bind address");
            }
        }
        let v = &self.vad;
        if !(0.01..=1.0).contains(&v.threshold)
            || v.neg_threshold.is_some_and(|n| n > v.threshold || n < 0.01)
        {
            bail!("vad thresholds must satisfy 0.01 <= neg_threshold <= threshold <= 1");
        }
        if self.asr.hotwords.values().any(|w| !(1..=5).contains(w)) {
            bail!("asr.hotwords weights must be within 1..=5");
        }
        Ok(())
    }
}

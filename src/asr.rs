//! DashScope (Bailian) ASR clients.
//!
//! * HTTP: `qwen-audio-3.0-asr-flash`, one request per finished utterance.
//! * Streaming: `qwen-audio-3.0-asr-flash-streaming`, WebSocket duplex task
//!   opened while the user is talking, with partial results.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::{self, Message, client::IntoClientRequest};
use tracing::{debug, warn};

use crate::audio::{self, ASR_RATE};
use crate::config::{AsrConfig, AsrMode};
use crate::usage::Usage;

#[derive(Debug, Clone)]
pub struct AsrResult {
    pub text: String,
    pub billed_seconds: f64,
    pub model: String,
}

/// Runtime switches, changeable from the control panel.
pub struct AsrRuntime {
    enabled: AtomicBool,
    mode: Mutex<AsrMode>,
    /// Set when the provider refuses service (free tier used up, bad key).
    blocked: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AsrState {
    pub enabled: bool,
    pub mode: AsrMode,
    pub blocked: Option<String>,
    pub has_api_key: bool,
    pub http_model: String,
    pub streaming_model: String,
}

pub struct Asr {
    cfg: AsrConfig,
    http: reqwest::Client,
    pub usage: Arc<Usage>,
    runtime: AsrRuntime,
}

/// Feed audio with `audio`; drop it to finish the task. `result` resolves
/// once the provider returned the final text.
pub struct StreamHandle {
    pub audio: mpsc::UnboundedSender<Vec<f32>>,
    pub result: oneshot::Receiver<Result<AsrResult>>,
}

impl Asr {
    pub fn new(cfg: AsrConfig, usage: Arc<Usage>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.request_timeout_s))
            .build()?;
        let runtime = AsrRuntime {
            enabled: AtomicBool::new(cfg.enabled),
            mode: Mutex::new(cfg.mode),
            blocked: Mutex::new(None),
        };
        Ok(Self { cfg, http, usage, runtime })
    }

    /// The mode to use for a new utterance, or `None` if ASR is off.
    pub fn active_mode(&self) -> Option<AsrMode> {
        if !self.runtime.enabled.load(Ordering::Relaxed)
            || self.cfg.api_key.is_none()
            || self.runtime.blocked.lock().unwrap().is_some()
        {
            return None;
        }
        Some(*self.runtime.mode.lock().unwrap())
    }

    pub fn state(&self) -> AsrState {
        AsrState {
            enabled: self.runtime.enabled.load(Ordering::Relaxed),
            mode: *self.runtime.mode.lock().unwrap(),
            blocked: self.runtime.blocked.lock().unwrap().clone(),
            has_api_key: self.cfg.api_key.is_some(),
            http_model: self.cfg.http_model.clone(),
            streaming_model: self.cfg.streaming_model.clone(),
        }
    }

    pub fn set(&self, enabled: Option<bool>, mode: Option<AsrMode>, clear_block: bool) {
        if let Some(e) = enabled {
            self.runtime.enabled.store(e, Ordering::Relaxed);
        }
        if let Some(m) = mode {
            *self.runtime.mode.lock().unwrap() = m;
        }
        if clear_block {
            *self.runtime.blocked.lock().unwrap() = None;
        }
    }

    pub fn max_segment_seconds(&self) -> f64 {
        // qwen-audio-3.0-asr-flash accepts up to 5 minutes.
        290.0
    }

    fn key(&self) -> Result<&str> {
        self.cfg.api_key.as_deref().ok_or_else(|| anyhow!("DASHSCOPE_API_KEY not set"))
    }

    /// Inspect a provider error; block ASR for errors that will not go away.
    fn on_provider_error(&self, status: Option<u16>, code: &str, message: &str) -> anyhow::Error {
        self.usage.record_failure();
        let fatal = code.contains("FreeTierOnly")
            || code.contains("Arrearage")
            || code == "InvalidApiKey"
            || status == Some(401);
        let text = format!("ASR error {}{code}: {message}", status.map(|s| format!("{s} ")).unwrap_or_default());
        if fatal {
            warn!("disabling ASR: {text}");
            *self.runtime.blocked.lock().unwrap() = Some(text.clone());
        }
        anyhow!(text)
    }

    /// Transcribe one finished utterance over HTTP.
    pub async fn transcribe(&self, pcm: &[f32], context: &[String]) -> Result<AsrResult> {
        let seconds = pcm.len() as f64 / ASR_RATE as f64;
        self.usage.check(seconds).map_err(|e| anyhow!(e))?;
        let wav = audio::wav_bytes(pcm, ASR_RATE)?;
        let data = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(wav));

        let mut messages: Vec<Value> = context
            .iter()
            .map(|t| json!({"role": "user", "content": [{"type": "input_text", "text": t}]}))
            .collect();
        messages.push(json!({"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data}}]}));
        let mut parameters = json!({"format": "wav", "sample_rate": ASR_RATE.to_string()});
        if !self.cfg.language_hints.is_empty() {
            parameters["language_hints"] = json!(self.cfg.language_hints);
        }
        if !self.cfg.hotwords.is_empty() {
            parameters["vocabulary"] = json!(self.cfg.hotwords);
        }
        let body = json!({
            "model": self.cfg.http_model,
            "input": {"messages": messages},
            "parameters": parameters,
        });

        let mut attempt = 0;
        let resp: Value = loop {
            attempt += 1;
            let res = self
                .http
                .post(&self.cfg.http_url)
                .bearer_auth(self.key()?)
                .header("X-DashScope-SSE", "disable")
                .json(&body)
                .send()
                .await;
            let resp = match res {
                Ok(r) => r,
                Err(e) if attempt < 3 => {
                    warn!("ASR request failed ({e}), retrying");
                    tokio::time::sleep(Duration::from_millis(500 * attempt)).await;
                    continue;
                }
                Err(e) => {
                    self.usage.record_failure();
                    return Err(e).context("ASR request");
                }
            };
            let status = resp.status();
            let v: Value = resp.json().await.unwrap_or(Value::Null);
            if status.is_success() {
                break v;
            }
            let code = v["code"].as_str().unwrap_or_default().to_string();
            let msg = v["message"].as_str().unwrap_or_default().to_string();
            if (status.as_u16() == 429 || status.is_server_error()) && attempt < 3 {
                warn!("ASR {status} {code}, retrying");
                tokio::time::sleep(Duration::from_millis(1000 * attempt)).await;
                continue;
            }
            return Err(self.on_provider_error(Some(status.as_u16()), &code, &msg));
        };

        let text = resp["output"]["text"]
            .as_str()
            .or_else(|| resp["output"]["sentence"]["text"].as_str())
            .unwrap_or_default()
            .trim()
            .to_string();
        // Billed duration is reported in whole seconds; fall back to the sent length.
        let billed = resp["usage"]["duration"].as_f64().unwrap_or(seconds.ceil());
        self.usage.record(&self.cfg.http_model, billed);
        Ok(AsrResult { text, billed_seconds: billed, model: self.cfg.http_model.clone() })
    }

    /// Open a streaming recognition task.
    pub fn start_stream(self: &Arc<Self>, on_partial: impl Fn(String) + Send + 'static) -> Result<StreamHandle> {
        self.usage.check(1.0).map_err(|e| anyhow!(e))?;
        self.key()?;
        let (audio_tx, audio_rx) = mpsc::unbounded_channel();
        let (res_tx, res_rx) = oneshot::channel();
        let me = self.clone();
        tokio::spawn(async move {
            let res = me.run_stream(audio_rx, on_partial).await;
            if let Err(e) = &res {
                debug!("streaming ASR ended with error: {e:#}");
            }
            let _ = res_tx.send(res);
        });
        Ok(StreamHandle { audio: audio_tx, result: res_rx })
    }

    async fn run_stream(
        &self,
        mut audio_rx: mpsc::UnboundedReceiver<Vec<f32>>,
        on_partial: impl Fn(String),
    ) -> Result<AsrResult> {
        let mut req = self.cfg.ws_url.as_str().into_client_request()?;
        req.headers_mut().insert("Authorization", format!("bearer {}", self.key()?).parse()?);
        let (ws, _) = match tokio::time::timeout(Duration::from_secs(10), tokio_tungstenite::connect_async(req))
            .await
            .context("ASR websocket connect timeout")?
        {
            Ok(ok) => ok,
            Err(tungstenite::Error::Http(resp)) => {
                let body = resp.body().as_ref().map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
                let code = serde_json::from_str::<Value>(&body)
                    .ok()
                    .and_then(|v| v["code"].as_str().map(str::to_string))
                    .unwrap_or_default();
                return Err(self.on_provider_error(Some(resp.status().as_u16()), &code, &body));
            }
            Err(e) => return Err(e.into()),
        };
        let (mut tx, mut rx) = ws.split();

        let task_id = uuid::Uuid::new_v4().simple().to_string();
        let mut parameters = json!({
            "format": "pcm",
            "sample_rate": ASR_RATE,
            "max_sentence_silence": self.cfg.max_sentence_silence_ms.clamp(200, 6000),
        });
        if !self.cfg.language_hints.is_empty() {
            parameters["language_hints"] = json!(self.cfg.language_hints);
        }
        let run_task = json!({
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio", "task": "asr", "function": "recognition",
                "model": self.cfg.streaming_model, "parameters": parameters, "input": {}
            }
        });
        tx.send(Message::text(run_task.to_string())).await?;

        let mut started = false;
        let mut audio_done = false;
        let mut finals: Vec<(u64, String)> = Vec::new();
        let mut billed = 0.0;
        let timeout = Duration::from_secs(self.cfg.request_timeout_s);
        loop {
            tokio::select! {
                chunk = audio_rx.recv(), if started && !audio_done => match chunk {
                    Some(pcm) => tx.send(Message::binary(audio::to_s16le(&pcm))).await?,
                    None => {
                        audio_done = true;
                        let finish = json!({
                            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": {"input": {}}
                        });
                        tx.send(Message::text(finish.to_string())).await?;
                    }
                },
                msg = tokio::time::timeout(timeout, rx.next()) => {
                    let msg = match msg {
                        Err(_) if audio_done => bail!("ASR stream timed out waiting for result"),
                        Err(_) => continue,
                        Ok(None) => bail!("ASR websocket closed early"),
                        Ok(Some(m)) => m?,
                    };
                    let Message::Text(t) = msg else { continue };
                    let v: Value = serde_json::from_str(&t)?;
                    match v["header"]["event"].as_str().unwrap_or_default() {
                        "task-started" => started = true,
                        "result-generated" => {
                            let s = &v["payload"]["output"]["sentence"];
                            if s["heartbeat"].as_bool() == Some(true) || s["sentence_id"].as_u64() == Some(0) {
                                continue;
                            }
                            let id = s["sentence_id"].as_u64().unwrap_or(0);
                            let text = s["text"].as_str().unwrap_or_default().to_string();
                            if s["sentence_end"].as_bool() == Some(true) {
                                finals.retain(|(i, _)| *i != id);
                                finals.push((id, text));
                            } else {
                                let mut so_far: Vec<&str> = finals.iter().map(|(_, t)| t.as_str()).collect();
                                so_far.push(&text);
                                on_partial(so_far.concat());
                            }
                            if let Some(d) = v["payload"]["usage"]["duration"].as_f64() {
                                billed = d;
                            }
                        }
                        "task-finished" => {
                            if let Some(d) = v["payload"]["usage"]["duration"].as_f64() {
                                billed = d;
                            }
                            break;
                        }
                        "task-failed" => {
                            let code = v["header"]["error_code"].as_str().unwrap_or_default();
                            let msg = v["header"]["error_message"].as_str().unwrap_or_default();
                            return Err(self.on_provider_error(None, code, msg));
                        }
                        other => debug!(event = other, "ignored ASR event"),
                    }
                }
            }
        }
        let _ = tx.send(Message::Close(None)).await;
        self.usage.record(&self.cfg.streaming_model, billed);
        let text = finals.into_iter().map(|(_, t)| t).collect::<String>().trim().to_string();
        Ok(AsrResult { text, billed_seconds: billed, model: self.cfg.streaming_model.clone() })
    }
}

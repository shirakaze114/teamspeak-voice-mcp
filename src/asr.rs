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
        Ok(Self {
            cfg,
            http,
            usage,
            runtime,
        })
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

    fn key(&self) -> Result<&str> {
        self.cfg
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow!("DASHSCOPE_API_KEY not set"))
    }

    /// Inspect a provider error; block ASR for errors that will not go away.
    fn on_provider_error(&self, status: Option<u16>, code: &str, message: &str) -> anyhow::Error {
        self.usage.record_failure();
        let fatal = code.contains("FreeTierOnly")
            || code.contains("Arrearage")
            || code == "InvalidApiKey"
            || matches!(status, Some(401..=403));
        let text = format!(
            "ASR error {}{code}: {message}",
            status.map(|s| format!("{s} ")).unwrap_or_default()
        );
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
        let data = format!(
            "data:audio/wav;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(wav)
        );

        let context = sanitize_context(context);
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
        Ok(AsrResult {
            text,
            billed_seconds: billed,
            model: self.cfg.http_model.clone(),
        })
    }

    /// Open a streaming recognition task.
    pub fn start_stream(
        self: &Arc<Self>,
        on_partial: impl Fn(String) + Send + 'static,
    ) -> Result<StreamHandle> {
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
        Ok(StreamHandle {
            audio: audio_tx,
            result: res_rx,
        })
    }

    async fn run_stream(
        &self,
        mut audio_rx: mpsc::UnboundedReceiver<Vec<f32>>,
        on_partial: impl Fn(String),
    ) -> Result<AsrResult> {
        let mut req = self.cfg.ws_url.as_str().into_client_request()?;
        req.headers_mut()
            .insert("Authorization", format!("bearer {}", self.key()?).parse()?);
        let (ws, _) = match tokio::time::timeout(
            Duration::from_secs(10),
            tokio_tungstenite::connect_async(req),
        )
        .await
        .context("ASR websocket connect timeout")?
        {
            Ok(ok) => ok,
            Err(tungstenite::Error::Http(resp)) => {
                let body = resp
                    .body()
                    .as_ref()
                    .map(|b| String::from_utf8_lossy(b).into_owned())
                    .unwrap_or_default();
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
        let mut finish_sent = false;
        let mut pending_pcm: Vec<f32> = Vec::new();
        let mut sent_seconds = 0.0;
        let mut finals: Vec<(u64, String)> = Vec::new();
        let mut billed = 0.0;
        let timeout = Duration::from_secs(self.cfg.request_timeout_s);

        // Audio may arrive before task-started; buffer it and flush once the
        // server is ready. Frames are 100 ms (1600 samples at 16 kHz).
        macro_rules! flush_audio {
            ($all:expr) => {{
                while pending_pcm.len() >= 1600 || ($all && !pending_pcm.is_empty()) {
                    let n = if pending_pcm.len() >= 1600 {
                        1600
                    } else {
                        pending_pcm.len()
                    };
                    let chunk: Vec<f32> = pending_pcm.drain(..n).collect();
                    tx.send(Message::binary(audio::to_s16le(&chunk))).await?;
                }
            }};
        }
        macro_rules! finish_task {
            () => {{
                if !finish_sent {
                    let finish = json!({
                        "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                        "payload": {"input": {}}
                    });
                    tx.send(Message::text(finish.to_string())).await?;
                    finish_sent = true;
                }
            }};
        }

        loop {
            tokio::select! {
                chunk = audio_rx.recv(), if !audio_done => match chunk {
                    Some(pcm) => {
                        sent_seconds += pcm.len() as f64 / ASR_RATE as f64;
                        pending_pcm.extend_from_slice(&pcm);
                        if started {
                            flush_audio!(false);
                        }
                    }
                    None => {
                        audio_done = true;
                        if started {
                            flush_audio!(true);
                            finish_task!();
                        }
                    }
                },
                msg = tokio::time::timeout(timeout, rx.next()) => {
                    let msg = match msg {
                        Err(_) if audio_done || !started => bail!("ASR stream timed out waiting for server"),
                        Err(_) => continue,
                        Ok(None) => bail!("ASR websocket closed early"),
                        Ok(Some(m)) => m?,
                    };
                    let Message::Text(t) = msg else { continue };
                    let v: Value = serde_json::from_str(&t)?;
                    match v["header"]["event"].as_str().unwrap_or_default() {
                        "task-started" => {
                            started = true;
                            flush_audio!(audio_done);
                            if audio_done {
                                finish_task!();
                            }
                        }
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
        let billed = if billed > 0.0 {
            billed
        } else {
            sent_seconds.ceil()
        };
        self.usage.record(&self.cfg.streaming_model, billed);
        let text = finals
            .into_iter()
            .map(|(_, t)| t)
            .collect::<String>()
            .trim()
            .to_string();
        Ok(AsrResult {
            text,
            billed_seconds: billed,
            model: self.cfg.streaming_model.clone(),
        })
    }
}

/// Keep at most five recent context messages and 400 characters, as required by
/// the HTTP ASR API. The audio message itself is not included here.
fn sanitize_context(context: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = 0;
    for text in context.iter().rev() {
        if out.len() >= 5 || chars >= 400 {
            break;
        }
        let clipped: String = text.chars().take(400 - chars).collect();
        if clipped.is_empty() {
            continue;
        }
        chars += clipped.chars().count();
        out.push(clipped);
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    #[test]
    fn context_is_bounded() {
        let input: Vec<String> = (0..8).map(|i| format!("message-{i}")).collect();
        let out = sanitize_context(&input);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], "message-3");
        let long = vec!["x".repeat(500)];
        assert_eq!(sanitize_context(&long)[0].chars().count(), 400);
    }

    fn test_config() -> AsrConfig {
        AsrConfig {
            api_key: Some("test-key".into()),
            request_timeout_s: 2,
            budget: crate::config::BudgetConfig {
                daily_limit_seconds: 0,
                total_limit_seconds: 0,
                usage_file: std::env::temp_dir()
                    .join(format!("tsv-asr-test-{}.json", uuid::Uuid::new_v4())),
            },
            ..Default::default()
        }
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut data = Vec::new();
        let mut expected = None;
        let mut buf = [0u8; 8192];
        loop {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buf[..n]);
            if expected.is_none()
                && let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n")
            {
                let headers = String::from_utf8_lossy(&data[..pos]);
                let len = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                    })
                    .flatten()
                    .unwrap_or(0);
                expected = Some(pos + 4 + len);
            }
            if expected.is_some_and(|end| data.len() >= end) {
                break;
            }
        }
        data
    }

    #[tokio::test]
    async fn http_protocol_and_usage() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = r#"{"output":{"text":"你好"},"usage":{"duration":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        let mut cfg = test_config();
        cfg.http_url = format!("http://{addr}/");
        let usage = Arc::new(Usage::load(cfg.budget.clone()));
        let asr = Asr::new(cfg, usage.clone()).unwrap();
        let result = asr
            .transcribe(&vec![0.0; 16_000], &["前文".into()])
            .await
            .unwrap();
        assert_eq!(result.text, "你好");
        assert_eq!(result.billed_seconds, 1.0);
        let request = String::from_utf8_lossy(&server.await.unwrap()).into_owned();
        assert!(request.contains("qwen-audio-3.0-asr-flash"));
        assert!(request.contains("\"sample_rate\":\"16000\""));
        assert!(request.contains("input_audio"));
        assert!(request.contains("input_text"));
        assert_eq!(usage.snapshot().total_seconds, 1.0);
    }

    #[tokio::test]
    async fn streaming_protocol_and_final_result() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            let mut binary_frames = 0;
            while let Some(msg) = ws.next().await {
                match msg.unwrap() {
                    Message::Text(text) => {
                        let v: Value = serde_json::from_str(&text).unwrap();
                        match v["header"]["action"].as_str() {
                            Some("run-task") => {
                                ws.send(Message::text(
                                    r#"{"header":{"event":"task-started"},"payload":{}}"#,
                                ))
                                .await
                                .unwrap();
                            }
                            Some("finish-task") => {
                                ws.send(Message::text(r#"{"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"sentence_id":1,"sentence_end":true,"text":"你好"}},"usage":{"duration":1}}}"#)).await.unwrap();
                                ws.send(Message::text(r#"{"header":{"event":"task-finished"},"payload":{"usage":{"duration":1}}}"#)).await.unwrap();
                                break;
                            }
                            _ => {}
                        }
                    }
                    Message::Binary(bytes) => {
                        assert!(bytes.len() == 3200 || bytes.len() == 1600);
                        binary_frames += 1;
                    }
                    _ => {}
                }
            }
            binary_frames
        });
        let mut cfg = test_config();
        cfg.mode = AsrMode::Streaming;
        cfg.ws_url = format!("ws://{addr}/");
        let usage = Arc::new(Usage::load(cfg.budget.clone()));
        let asr = Arc::new(Asr::new(cfg, usage.clone()).unwrap());
        let StreamHandle { audio, result } = asr.start_stream(|_| {}).unwrap();
        audio.send(vec![0.0; 3200]).unwrap();
        audio.send(vec![0.0; 800]).unwrap();
        drop(audio);
        let result = tokio::time::timeout(Duration::from_secs(5), result)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(result.text, "你好");
        assert_eq!(server.await.unwrap(), 3);
        assert_eq!(usage.snapshot().total_seconds, 1.0);
    }
}

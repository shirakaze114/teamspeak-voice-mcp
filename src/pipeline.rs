//! Per-user pipeline: TS track → Silero VAD (silero crate) → ASR → transcript store.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use silero::{SampleRate, Session, SpeechOptions, SpeechSegmenter, SpeechSegmenterExt, StreamState};
use time::OffsetDateTime;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::asr::{Asr, StreamHandle};
use crate::audio::{self, ASR_RATE};
use crate::config::{AsrMode, RecordingConfig, VadConfig};
use crate::transcripts::{LiveEvent, Query, Speaker, TranscriptEvent, TranscriptStore, rfc3339};
use crate::ts::{SpeakerInfo, TrackEvent, TsHandle};

/// An utterance that VAD has opened.
struct Utterance {
    id: String,
    started_at: OffsetDateTime,
    /// Streaming mode: live ASR task, fed from `fed_until`.
    stream: Option<StreamHandle>,
    fed_until: u64,
}

struct UserState {
    speaker: SpeakerInfo,
    stream: StreamState,
    seg: SpeechSegmenter,
    /// Audio kept for slicing segments; `buf[0]` is sample `buf_start`.
    buf: Vec<f32>,
    buf_start: u64,
    /// TS stopped sending; finish VAD after `silence_end_ms`.
    idle_since: Option<Instant>,
    active_since: Option<u64>,
    utt: Option<Utterance>,
}

impl UserState {
    fn pushed(&self) -> u64 {
        self.buf_start + self.buf.len() as u64
    }
    fn slice(&self, start: u64, end: u64) -> &[f32] {
        let s = start.saturating_sub(self.buf_start) as usize;
        let e = (end.saturating_sub(self.buf_start) as usize).min(self.buf.len());
        &self.buf[s.min(e)..e]
    }
}

pub struct Pipeline {
    vad_cfg: VadConfig,
    opts: SpeechOptions,
    session: Session,
    users: HashMap<u16, UserState>,
    asr: Arc<Asr>,
    store: Arc<TranscriptStore>,
    ts: TsHandle,
    recording: RecordingConfig,
    context_messages: usize,
}

impl Pipeline {
    pub fn new(
        vad_cfg: VadConfig,
        asr: Arc<Asr>,
        store: Arc<TranscriptStore>,
        ts: TsHandle,
        recording: RecordingConfig,
        context_messages: usize,
    ) -> Result<Self> {
        let mut opts = SpeechOptions::default()
            .with_sample_rate(SampleRate::Rate16k)
            .with_start_threshold(vad_cfg.threshold)
            .with_min_run_duration(Duration::from_millis(vad_cfg.min_speech_ms as u64))
            .with_min_gap_duration(Duration::from_millis(vad_cfg.silence_end_ms as u64))
            .with_pad(Duration::from_millis(vad_cfg.pad_ms as u64))
            .with_max_run_duration(Duration::from_secs(vad_cfg.max_segment_s as u64));
        if let Some(n) = vad_cfg.neg_threshold {
            opts = opts.with_end_threshold(n);
        }
        if recording.enabled {
            std::fs::create_dir_all(&recording.dir)?;
        }
        Ok(Self {
            vad_cfg,
            opts,
            session: Session::bundled()?,
            users: HashMap::new(),
            asr,
            store,
            ts,
            recording,
            context_messages,
        })
    }

    pub async fn run(mut self, mut rx: mpsc::Receiver<TrackEvent>) {
        let mut tick = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                ev = rx.recv() => match ev {
                    Some(ev) => self.on_event(ev),
                    None => break,
                },
                _ = tick.tick() => self.on_tick(),
            }
        }
    }

    fn on_event(&mut self, ev: TrackEvent) {
        match ev {
            TrackEvent::Start { speaker, .. } => {
                let id = speaker.client_id;
                let opts = self.opts.clone();
                let u = self.users.entry(id).or_insert_with(|| UserState {
                    speaker: speaker.clone(),
                    stream: StreamState::new(SampleRate::Rate16k),
                    seg: SpeechSegmenter::new(opts),
                    buf: Vec::new(),
                    buf_start: 0,
                    idle_since: None,
                    active_since: None,
                    utt: None,
                });
                u.speaker = speaker;
                u.idle_since = None;
            }
            TrackEvent::Audio { client_id, samples } => {
                if self.vad_cfg.suppress_while_bot_speaking && self.ts.is_playing() {
                    return;
                }
                self.feed(client_id, &samples);
            }
            TrackEvent::End { client_id, .. } => {
                if let Some(u) = self.users.get_mut(&client_id) {
                    u.idle_since = Some(Instant::now());
                }
            }
            TrackEvent::UsersChanged => self.store.emit(LiveEvent::UsersChanged),
        }
    }

    fn on_tick(&mut self) {
        let timeout = Duration::from_millis(self.vad_cfg.silence_end_ms as u64);
        let idle: Vec<u16> = self
            .users
            .iter()
            .filter(|(_, u)| u.idle_since.is_some_and(|t| t.elapsed() >= timeout))
            .map(|(id, _)| *id)
            .collect();
        for id in idle {
            self.finish_user(id);
        }
    }

    fn feed(&mut self, id: u16, samples: &[f32]) {
        let Some(u) = self.users.get_mut(&id) else { return };
        u.buf.extend_from_slice(samples);
        let mut segments = Vec::new();
        match u.seg.push_samples(&mut self.session, &mut u.stream, samples) {
            Ok(Some(s)) => {
                segments.push(s);
                while let Ok(Some(s)) = u.seg.push_samples(&mut self.session, &mut u.stream, &[]) {
                    segments.push(s);
                }
            }
            Ok(None) => {}
            Err(e) => warn!("VAD failed: {e}"),
        }
        for s in segments {
            self.on_segment(id, s.start_sample(), s.end_sample());
        }
        self.track_activity(id);
    }

    /// Opens utterances when VAD becomes active, streams audio, trims the buffer.
    fn track_activity(&mut self, id: u16) {
        let mode = self.asr.active_mode();
        let min_speech = ms_to_samples(self.vad_cfg.min_speech_ms);
        let pad = ms_to_samples(self.vad_cfg.pad_ms);
        let Some(u) = self.users.get_mut(&id) else { return };
        let now = u.pushed();
        if u.seg.is_active() {
            let since = *u.active_since.get_or_insert(now);
            if u.utt.is_none() && now - since >= min_speech {
                let utt_id = format!("utt_{}", uuid::Uuid::new_v4().simple());
                let mut utt = Utterance {
                    id: utt_id.clone(),
                    started_at: OffsetDateTime::now_utc() - time::Duration::milliseconds(((now - since) * 1000 / ASR_RATE as u64) as i64),
                    stream: None,
                    fed_until: since.saturating_sub(pad),
                };
                if mode == Some(AsrMode::Streaming) {
                    let store = self.store.clone();
                    let partial = partial_template(&u.speaker, &utt_id, utt.started_at, &self.asr.state().streaming_model);
                    match self.asr.start_stream(move |text| {
                        let mut ev = partial.clone();
                        ev.text = text;
                        store.push_partial(ev);
                    }) {
                        Ok(h) => utt.stream = Some(h),
                        Err(e) => warn!("cannot start streaming ASR: {e:#}"),
                    }
                }
                self.store.emit(LiveEvent::Speech { client_id: id, name: u.speaker.name.clone(), active: true });
                u.utt = Some(utt);
            }
        } else {
            u.active_since = None;
        }
        // Stream new audio to the live ASR task.
        if let Some(utt) = &mut u.utt
            && let Some(h) = &utt.stream
        {
            let start = utt.fed_until;
            let chunk = u.slice(start, now).to_vec();
            if !chunk.is_empty() {
                let _ = h.audio.send(chunk);
            }
            utt.fed_until = now;
        }
        // Keep only what a future segment may still need.
        // `active_since` lags the real onset by up to one VAD frame; keep a margin.
        let margin = pad + 2 * silero::SampleRate::Rate16k.chunk_samples() as u64;
        let keep_from = match u.active_since {
            Some(s) => s.saturating_sub(margin),
            None => now.saturating_sub(margin),
        };
        if keep_from > u.buf_start {
            let n = ((keep_from - u.buf_start) as usize).min(u.buf.len());
            u.buf.drain(..n);
            u.buf_start += n as u64;
        }
    }

    fn finish_user(&mut self, id: u16) {
        let Some(u) = self.users.get_mut(&id) else { return };
        let mut segments = Vec::new();
        if let Ok(Some(s)) = u.seg.finish_stream(&mut self.session, &mut u.stream) {
            segments.push(s);
        }
        while let Ok(Some(s)) = u.seg.push_samples(&mut self.session, &mut u.stream, &[]) {
            segments.push(s);
        }
        for s in segments {
            self.on_segment(id, s.start_sample(), s.end_sample());
        }
        if let Some(u) = self.users.get_mut(&id) {
            if let Some(utt) = u.utt.take() {
                // VAD dropped the run (too short): close a live stream without a segment.
                drop(utt.stream);
                self.store.emit(LiveEvent::Speech { client_id: id, name: u.speaker.name.clone(), active: false });
            }
            u.seg.reset();
            u.stream.reset();
            u.buf.clear();
            u.buf_start = 0;
            u.active_since = None;
            u.idle_since = None;
        }
    }

    fn on_segment(&mut self, id: u16, start: u64, end: u64) {
        let mode = self.asr.active_mode();
        let Some(u) = self.users.get_mut(&id) else { return };
        let pcm = u.slice(start, end).to_vec();
        let utt = u.utt.take();
        u.active_since = None;
        // Streaming: send the audio between the last feed and the segment end.
        if let Some(utt) = &utt
            && let Some(h) = &utt.stream
            && end > utt.fed_until
        {
            let _ = h.audio.send(u.slice(utt.fed_until, end).to_vec());
        }
        let speaker = u.speaker.clone();
        let duration_ms = (pcm.len() as u64 * 1000 / ASR_RATE as u64) as u32;
        let ended_at = OffsetDateTime::now_utc();
        let (utt_id, started_at, stream) = match utt {
            Some(utt) => (utt.id, utt.started_at, utt.stream),
            None => (
                format!("utt_{}", uuid::Uuid::new_v4().simple()),
                ended_at - time::Duration::milliseconds(duration_ms as i64),
                None,
            ),
        };
        debug!(client_id = id, name = %speaker.name, duration_ms, "segment");
        self.store.emit(LiveEvent::Speech { client_id: id, name: speaker.name.clone(), active: false });

        if self.recording.enabled {
            let path = self.recording.dir.join(format!("{}_c{}.wav", started_at.unix_timestamp(), id));
            let pcm = pcm.clone();
            tokio::task::spawn_blocking(move || audio::write_wav_mono(&path, &pcm, ASR_RATE));
        }

        let asr = self.asr.clone();
        let store = self.store.clone();
        let context: Vec<String> = if self.context_messages > 0 {
            self.store
                .query(&Query { limit: self.context_messages, channel_id: Some(speaker.channel_id), ..Default::default() })
                .into_iter()
                .map(|e| e.text)
                .collect()
        } else {
            Vec::new()
        };
        tokio::spawn(async move {
            let res = match (stream, mode) {
                (Some(h), _) => {
                    // Dropping the sender finishes the task.
                    let StreamHandle { audio, result } = h;
                    drop(audio);
                    result.await.unwrap_or_else(|_| Err(anyhow::anyhow!("stream task dropped")))
                }
                (None, Some(_)) => asr.transcribe(&pcm, &context).await,
                (None, None) => return,
            };
            match res {
                Ok(r) if r.text.is_empty() => debug!(client_id = id, "empty transcript"),
                Ok(r) => {
                    info!(client_id = id, name = %speaker.name, billed = r.billed_seconds, "📝 {}", r.text);
                    let seq = store.next_seq();
                    store.push_final(TranscriptEvent {
                        event_id: format!("evt_{seq}"),
                        seq,
                        kind: "transcript.final".into(),
                        utterance_id: utt_id,
                        channel_id: speaker.channel_id,
                        channel_name: speaker.channel_name.clone(),
                        speaker: Speaker {
                            client_id: speaker.client_id,
                            name: speaker.name.clone(),
                            uid: speaker.uid.clone(),
                            database_id: speaker.database_id,
                        },
                        text: r.text,
                        timestamp: rfc3339(started_at),
                        end_timestamp: Some(rfc3339(ended_at)),
                        duration_ms,
                        model: r.model,
                    });
                }
                Err(e) => {
                    warn!(client_id = id, "ASR failed: {e:#}");
                    store.emit(LiveEvent::Log { level: "warn".into(), message: format!("ASR: {e:#}") });
                }
            }
        });
    }
}

fn partial_template(s: &SpeakerInfo, utt_id: &str, started_at: OffsetDateTime, model: &str) -> TranscriptEvent {
    TranscriptEvent {
        event_id: String::new(),
        seq: 0,
        kind: "transcript.partial".into(),
        utterance_id: utt_id.to_string(),
        channel_id: s.channel_id,
        channel_name: s.channel_name.clone(),
        speaker: Speaker { client_id: s.client_id, name: s.name.clone(), uid: s.uid.clone(), database_id: s.database_id },
        text: String::new(),
        timestamp: rfc3339(started_at),
        end_timestamp: None,
        duration_ms: 0,
        model: model.to_string(),
    }
}

fn ms_to_samples(ms: u32) -> u64 {
    ms as u64 * ASR_RATE as u64 / 1000
}

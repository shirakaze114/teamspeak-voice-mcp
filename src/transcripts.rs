//! Transcript events, in-memory store with retention, and live broadcast.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::sync::broadcast;
use tracing::warn;

use crate::config::TranscriptConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speaker {
    pub client_id: u16,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEvent {
    pub event_id: String,
    /// Monotonic sequence number, usable as a cursor (`after_seq`).
    pub seq: u64,
    /// `transcript.final` or `transcript.partial`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Stable id of the utterance; partials and the final share it.
    pub utterance_id: String,
    pub channel_id: u64,
    pub channel_name: String,
    pub speaker: Speaker,
    pub text: String,
    /// RFC 3339 time the utterance started.
    pub timestamp: String,
    /// RFC 3339 time the utterance ended (finals only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_timestamp: Option<String>,
    /// Audio length sent to ASR.
    pub duration_ms: u32,
    pub model: String,
}

/// Events pushed to the panel and MCP notifier.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LiveEvent {
    Transcript(TranscriptEvent),
    /// Utterance started (VAD) or ended without a transcript.
    Speech { client_id: u16, name: String, active: bool },
    UsersChanged,
    Status,
    Log { level: String, message: String },
}

pub struct TranscriptStore {
    cfg: TranscriptConfig,
    seq: AtomicU64,
    events: Mutex<VecDeque<TranscriptEvent>>,
    pub live: broadcast::Sender<LiveEvent>,
}

pub fn rfc3339(t: OffsetDateTime) -> String {
    t.format(&time::format_description::well_known::Rfc3339).unwrap_or_default()
}

impl TranscriptStore {
    pub fn new(cfg: TranscriptConfig) -> Self {
        let (live, _) = broadcast::channel(512);
        let mut events = VecDeque::new();
        let mut seq = 0;
        if cfg.persist
            && let Ok(text) = std::fs::read_to_string(&cfg.file)
        {
            for line in text.lines() {
                if let Ok(ev) = serde_json::from_str::<TranscriptEvent>(line) {
                    seq = seq.max(ev.seq);
                    events.push_back(ev);
                }
            }
        }
        let store = Self { cfg, seq: AtomicU64::new(seq), events: Mutex::new(events), live };
        store.prune();
        store
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Store a final transcript and broadcast it.
    pub fn push_final(&self, ev: TranscriptEvent) {
        if self.cfg.persist {
            self.append_file(&ev);
        }
        self.events.lock().unwrap().push_back(ev.clone());
        self.prune();
        let _ = self.live.send(LiveEvent::Transcript(ev));
    }

    /// Broadcast a partial result (not stored).
    pub fn push_partial(&self, ev: TranscriptEvent) {
        let _ = self.live.send(LiveEvent::Transcript(ev));
    }

    pub fn emit(&self, ev: LiveEvent) {
        let _ = self.live.send(ev);
    }

    pub fn get(&self, event_id: &str) -> Option<TranscriptEvent> {
        self.events.lock().unwrap().iter().find(|e| e.event_id == event_id).cloned()
    }

    pub fn latest_seq(&self) -> u64 {
        self.events.lock().unwrap().back().map(|e| e.seq).unwrap_or(0)
    }

    /// Query final transcripts, oldest first, limited to the newest `limit`.
    pub fn query(&self, q: &Query) -> Vec<TranscriptEvent> {
        let events = self.events.lock().unwrap();
        let since = q.since.as_deref().and_then(|s| {
            OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
        });
        let mut out: Vec<TranscriptEvent> = events
            .iter()
            .rev()
            .filter(|e| q.after_seq.is_none_or(|s| e.seq > s))
            .filter(|e| q.client_id.is_none_or(|c| e.speaker.client_id == c))
            .filter(|e| q.channel_id.is_none_or(|c| e.channel_id == c))
            .filter(|e| {
                since.is_none_or(|s| {
                    OffsetDateTime::parse(&e.timestamp, &time::format_description::well_known::Rfc3339)
                        .map(|t| t >= s)
                        .unwrap_or(true)
                })
            })
            .take(q.limit.clamp(1, 500))
            .cloned()
            .collect();
        out.reverse();
        out
    }

    fn prune(&self) {
        let cutoff = OffsetDateTime::now_utc() - time::Duration::hours(self.cfg.retention_hours as i64);
        let mut events = self.events.lock().unwrap();
        while events.len() > self.cfg.max_events {
            events.pop_front();
        }
        while let Some(front) = events.front() {
            let t = OffsetDateTime::parse(&front.timestamp, &time::format_description::well_known::Rfc3339);
            if t.map(|t| t < cutoff).unwrap_or(true) {
                events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Rewrite the persisted file without expired events. Called periodically.
    pub fn compact_file(&self) {
        if !self.cfg.persist {
            return;
        }
        self.prune();
        let events = self.events.lock().unwrap();
        let mut buf = Vec::new();
        for e in events.iter() {
            if let Ok(line) = serde_json::to_string(e) {
                buf.extend_from_slice(line.as_bytes());
                buf.push(b'\n');
            }
        }
        let tmp = self.cfg.file.with_extension("jsonl.tmp");
        if let Err(e) = std::fs::write(&tmp, buf).and_then(|_| std::fs::rename(&tmp, &self.cfg.file)) {
            warn!("compacting transcript file failed: {e}");
        }
    }

    fn append_file(&self, ev: &TranscriptEvent) {
        if let Some(dir) = self.cfg.file.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let res = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.cfg.file)
            .and_then(|mut f| writeln!(f, "{}", serde_json::to_string(ev).unwrap_or_default()));
        if let Err(e) = res {
            warn!("appending transcript failed: {e}");
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
pub struct Query {
    /// Max events to return (newest N, returned oldest first). Default 20, max 500.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Only events with `seq` greater than this cursor.
    pub after_seq: Option<u64>,
    /// Only events at or after this RFC 3339 time.
    pub since: Option<String>,
    /// Only this speaker.
    pub client_id: Option<u16>,
    /// Only this channel.
    pub channel_id: Option<u64>,
}

fn default_limit() -> usize {
    20
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(store: &TranscriptStore, client: u16, text: &str) -> TranscriptEvent {
        let seq = store.next_seq();
        TranscriptEvent {
            event_id: format!("evt_{seq}"),
            seq,
            kind: "transcript.final".into(),
            utterance_id: format!("utt_{seq}"),
            channel_id: 5,
            channel_name: "c".into(),
            speaker: Speaker { client_id: client, name: "a".into(), uid: None, database_id: None },
            text: text.into(),
            timestamp: rfc3339(OffsetDateTime::now_utc()),
            end_timestamp: None,
            duration_ms: 1000,
            model: "m".into(),
        }
    }

    #[test]
    fn query_filters_and_cursor() {
        let store = TranscriptStore::new(TranscriptConfig { max_events: 3, ..Default::default() });
        for i in 0..5 {
            store.push_final(ev(&store, i % 2, &format!("t{i}")));
        }
        let all = store.query(&Query { limit: 10, ..Default::default() });
        assert_eq!(all.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(), ["t2", "t3", "t4"]);
        let after = store.query(&Query { limit: 10, after_seq: Some(4), ..Default::default() });
        assert_eq!(after.len(), 1);
        let c1 = store.query(&Query { limit: 10, client_id: Some(1), ..Default::default() });
        assert_eq!(c1.len(), 1);
        assert!(store.get("evt_5").is_some());
    }
}

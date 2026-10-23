//! Shared application state and authorization.

use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::asr::{Asr, AsrState};
use crate::config::{Scope, TokenConfig};
use crate::transcripts::TranscriptStore;
use crate::ts::{BotStatus, TsHandle};
use crate::tts::Speaker;
use crate::usage::UsageSnapshot;

pub struct App {
    pub ts: TsHandle,
    pub asr: Arc<Asr>,
    pub store: Arc<TranscriptStore>,
    pub tts: Arc<Speaker>,
    pub started: Instant,
    pub tokens: Vec<TokenConfig>,
    pub auth_disabled: bool,
}

/// Identity of an authenticated caller, stored in request extensions.
#[derive(Debug, Clone)]
pub struct Caller {
    pub name: String,
    pub scopes: Vec<Scope>,
}

impl Caller {
    pub fn has(&self, scope: Scope) -> bool {
        self.scopes.contains(&Scope::Admin) || self.scopes.contains(&scope)
    }
}

#[derive(Debug, Serialize)]
pub struct FullStatus {
    pub bot: Option<BotStatus>,
    pub asr: AsrState,
    pub usage: UsageSnapshot,
    pub tts_available: bool,
    pub latest_transcript_seq: u64,
    pub uptime_s: u64,
}

impl App {
    pub fn authenticate(&self, token: Option<&str>) -> Option<Caller> {
        if self.auth_disabled {
            return Some(Caller { name: "anonymous".into(), scopes: vec![Scope::Admin] });
        }
        let token = token?;
        self.tokens
            .iter()
            .find(|t| ct_eq(t.token.as_bytes(), token.as_bytes()))
            .map(|t| Caller { name: t.name.clone(), scopes: t.scopes.clone() })
    }

    pub async fn status(&self) -> FullStatus {
        FullStatus {
            bot: self.ts.status().await.ok(),
            asr: self.asr.state(),
            usage: self.asr.usage.snapshot(),
            tts_available: self.tts.available(),
            latest_transcript_seq: self.store.latest_seq(),
            uptime_s: self.started.elapsed().as_secs(),
        }
    }
}

/// Constant-time comparison for API tokens.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

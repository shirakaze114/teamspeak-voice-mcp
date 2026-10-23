//! ASR usage accounting and spending guard.
//!
//! Billed audio seconds are persisted per model and per local day. A request
//! is refused locally when the configured daily or total limit would be
//! exceeded, independent of the provider-side "stop when free tier is used up"
//! switch.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tracing::warn;

use crate::config::BudgetConfig;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct UsageFile {
    /// model -> billed seconds since tracking started
    total: BTreeMap<String, f64>,
    /// "YYYY-MM-DD" -> model -> billed seconds
    daily: BTreeMap<String, BTreeMap<String, f64>>,
    requests: u64,
    failures: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageSnapshot {
    pub today_seconds: f64,
    pub total_seconds: f64,
    pub daily_limit_seconds: u64,
    pub total_limit_seconds: u64,
    pub per_model_total: BTreeMap<String, f64>,
    pub today_per_model: BTreeMap<String, f64>,
    pub requests: u64,
    pub failures: u64,
    /// Estimated list price in CNY (before free tier).
    pub estimated_cost_cny: f64,
}

pub struct Usage {
    cfg: BudgetConfig,
    path: PathBuf,
    data: Mutex<UsageFile>,
}

/// List prices in CNY per second (Beijing region, 2026-10).
pub fn price_per_second(model: &str) -> f64 {
    if model.contains("streaming") || model.contains("realtime") {
        0.00033
    } else {
        0.00022
    }
}

fn today() -> String {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        now.month() as u8,
        now.day()
    )
}

impl Usage {
    pub fn load(cfg: BudgetConfig) -> Self {
        let path = cfg.usage_file.clone();
        let data = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        Self {
            cfg,
            path,
            data: Mutex::new(data),
        }
    }

    /// Returns `Err(reason)` if `seconds` more audio would exceed a limit.
    pub fn check(&self, seconds: f64) -> Result<(), String> {
        let d = self.data.lock().unwrap();
        let today_used: f64 = d
            .daily
            .get(&today())
            .map(|m| m.values().fold(0.0, |a, b| a + b))
            .unwrap_or(0.0);
        let total_used: f64 = d.total.values().fold(0.0, |a, b| a + b);
        if self.cfg.daily_limit_seconds > 0
            && today_used + seconds > self.cfg.daily_limit_seconds as f64
        {
            return Err(format!(
                "daily ASR limit reached ({today_used:.0}/{} s)",
                self.cfg.daily_limit_seconds
            ));
        }
        if self.cfg.total_limit_seconds > 0
            && total_used + seconds > self.cfg.total_limit_seconds as f64
        {
            return Err(format!(
                "total ASR limit reached ({total_used:.0}/{} s)",
                self.cfg.total_limit_seconds
            ));
        }
        Ok(())
    }

    pub fn record(&self, model: &str, seconds: f64) {
        let snapshot = {
            let mut d = self.data.lock().unwrap();
            *d.total.entry(model.to_string()).or_default() += seconds;
            *d.daily
                .entry(today())
                .or_default()
                .entry(model.to_string())
                .or_default() += seconds;
            d.requests += 1;
            // Keep 60 days of daily history.
            while d.daily.len() > 60 {
                let first = d.daily.keys().next().cloned().unwrap();
                d.daily.remove(&first);
            }
            d.clone()
        };
        self.save(&snapshot);
    }

    pub fn record_failure(&self) {
        let snapshot = {
            let mut d = self.data.lock().unwrap();
            d.failures += 1;
            d.clone()
        };
        self.save(&snapshot);
    }

    fn save(&self, d: &UsageFile) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("json.tmp");
        let res = serde_json::to_vec_pretty(d)
            .map_err(std::io::Error::other)
            .and_then(|b| std::fs::write(&tmp, b))
            .and_then(|_| std::fs::rename(&tmp, &self.path));
        if let Err(e) = res {
            warn!("saving usage file failed: {e}");
        }
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        let d = self.data.lock().unwrap();
        let today_per_model = d.daily.get(&today()).cloned().unwrap_or_default();
        let cost = d
            .total
            .iter()
            .fold(0.0, |acc, (m, s)| acc + price_per_second(m) * s);
        UsageSnapshot {
            today_seconds: today_per_model.values().fold(0.0, |a, b| a + b),
            total_seconds: d.total.values().fold(0.0, |a, b| a + b),
            daily_limit_seconds: self.cfg.daily_limit_seconds,
            total_limit_seconds: self.cfg.total_limit_seconds,
            per_model_total: d.total.clone(),
            today_per_model,
            requests: d.requests,
            failures: d.failures,
            estimated_cost_cny: cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_enforced() {
        let dir = std::env::temp_dir().join(format!("tsv_usage_{}", std::process::id()));
        let cfg = BudgetConfig {
            daily_limit_seconds: 10,
            total_limit_seconds: 0,
            usage_file: dir.join("u.json"),
        };
        let u = Usage::load(cfg.clone());
        assert!(u.check(5.0).is_ok());
        u.record("qwen-audio-3.0-asr-flash", 8.0);
        assert!(u.check(3.0).is_err());
        let reloaded = Usage::load(cfg);
        assert_eq!(reloaded.snapshot().today_seconds, 8.0);
        let _ = std::fs::remove_dir_all(dir);
    }
}

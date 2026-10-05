//! Claude API client for bulk labelling (SPEC §4.8 step 2): raw HTTPS with `reqwest`, since
//! there's no official Rust SDK. Message Batches by default (50% price), `--sync` for a quick
//! check of a few frames through `POST /v1/messages`.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{FrameLabel, Usage, frame_label_schema};

const API: &str = "https://api.anthropic.com/v1";
/// Batches are capped at 100k requests / 256 MB; frames are ~50–80 KB of base64 each.
pub const MAX_BATCH_REQUESTS: usize = 2000;

enum Auth {
    ApiKey(String),
    Bearer(String),
}

/// Resolves credentials: `ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`, then the `ant`
/// CLI's active profile.
fn auth() -> Result<Auth> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
        && !key.is_empty()
    {
        return Ok(Auth::ApiKey(key));
    }
    if let Ok(tok) = std::env::var("ANTHROPIC_AUTH_TOKEN")
        && !tok.is_empty()
    {
        return Ok(Auth::Bearer(tok));
    }
    let out = std::process::Command::new("ant").args(["auth", "print-credentials", "--access-token"]).output();
    match out {
        Ok(o) if o.status.success() => Ok(Auth::Bearer(String::from_utf8_lossy(&o.stdout).trim().to_string())),
        _ => bail!("no Anthropic credentials: set ANTHROPIC_API_KEY, or run `ant auth login`"),
    }
}

#[derive(Debug, Clone)]
pub struct Pricing {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// $ per million tokens (SPEC Appendix B). Cache writes are 1.25× input (5-minute TTL).
pub fn pricing(model: &str) -> Pricing {
    let (input, output, cache_read) = match model {
        m if m.starts_with("claude-sonnet-5") => (2.0, 10.0, 0.20),
        m if m.starts_with("claude-haiku-4-5") => (1.0, 5.0, 0.10),
        _ => (4.0, 20.0, 0.20),
    };
    Pricing { input, output, cache_read, cache_write: input * 1.25 }
}

pub fn cost_usd(model: &str, u: &Usage, batch: bool) -> f64 {
    let p = pricing(model);
    let raw = (u.input_tokens as f64 * p.input
        + u.output_tokens as f64 * p.output
        + u.cache_read_input_tokens as f64 * p.cache_read
        + u.cache_creation_input_tokens as f64 * p.cache_write)
        / 1e6;
    if batch { raw * 0.5 } else { raw }
}

/// The request `params` (SPEC §4.8). `system` is byte-identical across requests and marked
/// for caching; the zone list order is the only text that varies.
pub fn label_params(model: &str, effort: &str, guide: &str, jpeg_b64: &str, zone_list: &str) -> Value {
    json!({
        "model": model,
        "max_tokens": 4000,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": frame_label_schema()}
        },
        "system": [{"type": "text", "text": guide, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}},
            {"type": "text", "text": format!(
                "Label this Subway Surfers frame. Zones are outlined and named: {zone_list}. \
                 Give one entry per zone.")}
        ]}]
    })
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Label(FrameLabel),
    Refused(String),
    Invalid(String),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Answer {
    pub outcome: Outcome,
    pub request_id: Option<String>,
    pub usage: Option<Usage>,
}

/// Reads a Messages API response: refusal first, then the structured-output text block.
pub fn parse_message(msg: &Value) -> Answer {
    let usage = msg.get("usage").and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok());
    let request_id = msg.get("id").and_then(|v| v.as_str()).map(String::from);
    let stop = msg.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("");
    let outcome = if stop == "refusal" {
        let cat = msg.pointer("/stop_details/category").and_then(|v| v.as_str()).unwrap_or("unspecified");
        Outcome::Refused(format!("refusal ({cat})"))
    } else if stop == "max_tokens" {
        Outcome::Invalid("hit max_tokens".into())
    } else {
        let text = msg
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|blocks| blocks.iter().find(|b| b.get("type").and_then(|t| t.as_str()) == Some("text")))
            .and_then(|b| b.get("text").and_then(|t| t.as_str()));
        match text.map(serde_json::from_str::<FrameLabel>) {
            Some(Ok(label)) => match label.validate() {
                Ok(()) => Outcome::Label(label),
                Err(e) => Outcome::Invalid(e),
            },
            Some(Err(e)) => Outcome::Invalid(format!("bad JSON: {e}")),
            None => Outcome::Invalid("no text block".into()),
        }
    };
    Answer { outcome, request_id, usage }
}

pub struct Client {
    http: reqwest::Client,
    headers: HeaderMap,
}

#[derive(Debug, Deserialize)]
struct BatchStatus {
    id: String,
    processing_status: String,
    #[serde(default)]
    results_url: Option<String>,
    #[serde(default)]
    request_counts: Value,
}

impl Client {
    pub fn new() -> Result<Client> {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        match auth()? {
            Auth::ApiKey(k) => {
                headers.insert("x-api-key", HeaderValue::from_str(&k)?);
            }
            Auth::Bearer(t) => {
                headers.insert("authorization", HeaderValue::from_str(&format!("Bearer {t}"))?);
                headers.insert("anthropic-beta", HeaderValue::from_static("oauth-2025-04-20"));
            }
        }
        let http = reqwest::Client::builder().timeout(Duration::from_secs(600)).build()?;
        Ok(Client { http, headers })
    }

    async fn check(resp: reqwest::Response) -> Result<reqwest::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let body = resp.text().await.unwrap_or_default();
        bail!("Claude API {status}: {body}")
    }

    /// `POST /v1/messages` with server-side refusal fallback (`fallbacks: "default"`); the
    /// Batches API rejects that parameter, so only the sync path sends it.
    pub async fn message(&self, mut params: Value) -> Result<Answer> {
        params["fallbacks"] = json!("default");
        let mut headers = self.headers.clone();
        let beta = match headers.get("anthropic-beta").and_then(|v| v.to_str().ok()) {
            Some(existing) => format!("{existing},server-side-fallback-2026-07-01"),
            None => "server-side-fallback-2026-07-01".into(),
        };
        headers.insert("anthropic-beta", HeaderValue::from_str(&beta)?);
        for attempt in 0..4u32 {
            let resp = self.http.post(format!("{API}/messages")).headers(headers.clone()).json(&params).send().await;
            match resp {
                Ok(r) if r.status().as_u16() == 429 || r.status().is_server_error() => {
                    let wait = r.headers().get("retry-after").and_then(|v| v.to_str().ok()?.parse().ok()).unwrap_or(2u64 << attempt);
                    tracing::warn!("Claude API {}; retrying in {wait}s", r.status());
                    tokio::time::sleep(Duration::from_secs(wait)).await;
                }
                Ok(r) => return Ok(parse_message(&Self::check(r).await?.json::<Value>().await?)),
                Err(e) if attempt < 3 => {
                    tracing::warn!("network error {e}; retrying");
                    tokio::time::sleep(Duration::from_secs(2 << attempt)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
        bail!("Claude API kept failing")
    }

    /// Creates a batch from `(custom_id, params)` pairs and returns its id.
    pub async fn create_batch(&self, requests: &[(String, Value)]) -> Result<String> {
        let body = json!({
            "requests": requests.iter().map(|(id, p)| json!({"custom_id": id, "params": p})).collect::<Vec<_>>()
        });
        let r = self.http.post(format!("{API}/messages/batches")).headers(self.headers.clone()).json(&body).send().await?;
        let status: BatchStatus = Self::check(r).await?.json().await?;
        Ok(status.id)
    }

    /// Polls until `processing_status == "ended"`, then downloads the results JSONL and keys
    /// it by `custom_id` (results come back in any order).
    pub async fn wait_results(&self, batch_id: &str, poll: Duration) -> Result<HashMap<String, Answer>> {
        let status = loop {
            let r = self.http.get(format!("{API}/messages/batches/{batch_id}")).headers(self.headers.clone()).send().await?;
            let s: BatchStatus = Self::check(r).await?.json().await?;
            if s.processing_status == "ended" {
                break s;
            }
            tracing::info!("batch {batch_id}: {} {}", s.processing_status, s.request_counts);
            tokio::time::sleep(poll).await;
        };
        let url = status.results_url.context("ended batch has no results_url")?;
        let r = self.http.get(url).headers(self.headers.clone()).send().await?;
        let text = Self::check(r).await?.text().await?;
        let mut out = HashMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line)?;
            let id = v["custom_id"].as_str().unwrap_or_default().to_string();
            let kind = v.pointer("/result/type").and_then(|t| t.as_str()).unwrap_or("errored");
            let answer = match kind {
                "succeeded" => parse_message(&v["result"]["message"]),
                other => Answer {
                    outcome: Outcome::Failed(format!("{other}: {}", v.pointer("/result/error").cloned().unwrap_or(Value::Null))),
                    request_id: None,
                    usage: None,
                },
            };
            out.insert(id, answer);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::all_zone_ids;

    #[test]
    fn params_shape() {
        let p = label_params("claude-opus-5-5", "low", "GUIDE", "AAAA", "L-near, C-near");
        assert_eq!(p["output_config"]["effort"], "low");
        assert_eq!(p["output_config"]["format"]["type"], "json_schema");
        assert_eq!(p["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(p["messages"][0]["content"][0]["source"]["media_type"], "image/jpeg");
        assert!(p.get("thinking").is_none(), "Opus 5.5 rejects disabled thinking; leave the default");
        assert!(p.get("fallbacks").is_none(), "Batches API rejects fallbacks");
    }

    #[test]
    fn parses_success_refusal_and_bad_output() {
        let label = json!({
            "game_state": "Running", "player_lane": "L", "player_action": "jumping",
            "zones": all_zone_ids().iter().map(|z| json!({"zone": z, "obstacle": "TrainBody", "coins": true, "powerup": false, "sure": true})).collect::<Vec<_>>(),
            "notes": "ok"
        });
        let msg = json!({
            "id": "msg_1", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": label.to_string()}],
            "usage": {"input_tokens": 300, "output_tokens": 400, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}
        });
        let a = parse_message(&msg);
        assert!(matches!(a.outcome, Outcome::Label(ref l) if l.player_lane == "L"));
        assert_eq!(a.usage.unwrap().cache_read_input_tokens, 1000);
        let refused = parse_message(&json!({"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []}));
        assert!(matches!(refused.outcome, Outcome::Refused(ref s) if s.contains("cyber")));
        let bad = parse_message(&json!({"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]}));
        assert!(matches!(bad.outcome, Outcome::Invalid(_)));
    }

    #[test]
    fn cost_estimate_in_spec_range() {
        // ~300 image + ~50 text tokens fresh, ~1k cached system, ~600 output, batch discount.
        let u = Usage { input_tokens: 350, output_tokens: 600, cache_read_input_tokens: 1000, cache_creation_input_tokens: 0 };
        let per_frame = cost_usd("claude-opus-5-5", &u, true);
        assert!((0.005..=0.015).contains(&per_frame), "{per_frame}");
    }
}

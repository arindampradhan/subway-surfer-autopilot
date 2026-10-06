//! OpenJev sidecar protocol (SPEC §4.6): the JSON-lines requests and replies, and the channel
//! link the advisor talks through, so it never blocks on the process (`Sidecar`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarmItem {
    pub premise: String,
    pub options: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Request {
    Decide { id: u64, premise: String, options: BTreeMap<String, String> },
    Warm { items: Vec<WarmItem> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Reply {
    Ready {
        model: String,
        backend: String,
    },
    Decision {
        id: u64,
        probs: BTreeMap<String, f32>,
        ms: f32,
    },
    Error {
        #[serde(default)]
        id: Option<u64>,
        message: String,
    },
}

/// The advisor's side of the sidecar: send requests, poll replies without blocking.
pub struct SidecarLink {
    pub requests: UnboundedSender<Request>,
    pub replies: UnboundedReceiver<Reply>,
}

impl SidecarLink {
    /// A link wired to in-memory channels instead of a process (tests, replay).
    pub fn channels() -> (SidecarLink, UnboundedReceiver<Request>, UnboundedSender<Reply>) {
        let (req_tx, req_rx) = mpsc::unbounded_channel();
        let (rep_tx, rep_rx) = mpsc::unbounded_channel();
        (SidecarLink { requests: req_tx, replies: rep_rx }, req_rx, rep_tx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_matches_spec() {
        let req = Request::Decide {
            id: 42,
            premise: "Subway run.".into(),
            options: BTreeMap::from([("left".into(), "Left is the best move because it ...".into())]),
        };
        assert_eq!(
            serde_json::to_string(&req).unwrap(),
            r#"{"type":"decide","id":42,"premise":"Subway run.","options":{"left":"Left is the best move because it ..."}}"#
        );
        let r: Reply = serde_json::from_str(r#"{"type":"decision","id":42,"probs":{"left":0.61,"stay":0.22,"roll":0.17},"ms":231}"#).unwrap();
        assert!(matches!(r, Reply::Decision { id: 42, ms, .. } if ms == 231.0));
        let r: Reply = serde_json::from_str(r#"{"type":"ready","model":"OpenJev 0.8B","backend":"mlx"}"#).unwrap();
        assert!(matches!(r, Reply::Ready { .. }));
        let r: Reply = serde_json::from_str(r#"{"type":"error","id":42,"message":"boom"}"#).unwrap();
        assert!(matches!(r, Reply::Error { id: Some(42), .. }));
    }
}

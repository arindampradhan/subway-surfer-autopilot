//! Offline labelling with Claude (SPEC §4.8): frame selection, zone overlays, the Batches
//! client, automatic checks and the HTML review gallery.

pub mod batch;
pub mod boxes;
pub mod checks;
pub mod cmd;
pub mod overlay;
pub mod review;
pub mod select;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use ssbot_core::perception::zones::{BAND_NAMES, LANE_NAMES, parse_zone_id, zone_id};
use ssbot_core::perception::{GameState, Obstacle};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneLabel {
    pub zone: String,
    pub obstacle: Obstacle,
    pub coins: bool,
    pub powerup: bool,
    pub sure: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameLabel {
    pub game_state: GameState,
    /// "L", "C", "R" or "unknown".
    pub player_lane: String,
    /// "running", "jumping", "rolling", "switching" or "unknown".
    pub player_action: String,
    pub zones: Vec<ZoneLabel>,
    pub notes: String,
}

impl FrameLabel {
    pub fn zone(&self, lane: usize, band: usize) -> Option<&ZoneLabel> {
        let id = zone_id(lane, band);
        self.zones.iter().find(|z| z.zone == id)
    }

    /// Problems that make a label unusable: missing or unknown zones.
    pub fn validate(&self) -> Result<(), String> {
        for z in &self.zones {
            if parse_zone_id(&z.zone).is_none() {
                return Err(format!("unknown zone {}", z.zone));
            }
        }
        let missing: Vec<String> =
            all_zone_ids().into_iter().filter(|id| !self.zones.iter().any(|z| &z.zone == id)).collect();
        if !missing.is_empty() && self.game_state == GameState::Running {
            return Err(format!("missing zones {}", missing.join(", ")));
        }
        Ok(())
    }
}

pub fn all_zone_ids() -> Vec<String> {
    (0..3).flat_map(|l| (0..3).map(move |b| zone_id(l, b))).collect()
}

/// The `FrameLabel` JSON schema for structured outputs: every object closed, every field
/// required, no numeric constraints (SPEC §4.8).
pub fn frame_label_schema() -> Value {
    let states: Vec<String> = GameState::ALL.iter().map(|s| format!("{s:?}")).collect();
    let obstacles: Vec<String> = Obstacle::ALL.iter().map(|o| format!("{o:?}")).collect();
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["game_state", "player_lane", "player_action", "zones", "notes"],
        "properties": {
            "game_state": {"type": "string", "enum": states},
            "player_lane": {"type": "string", "enum": ["L", "C", "R", "unknown"]},
            "player_action": {"type": "string", "enum": ["running", "jumping", "rolling", "switching", "unknown"]},
            "zones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["zone", "obstacle", "coins", "powerup", "sure"],
                    "properties": {
                        "zone": {"type": "string", "enum": all_zone_ids()},
                        "obstacle": {"type": "string", "enum": obstacles},
                        "coins": {"type": "boolean"},
                        "powerup": {"type": "boolean"},
                        "sure": {"type": "boolean"}
                    }
                }
            },
            "notes": {"type": "string"}
        }
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, o: &Usage) {
        self.input_tokens += o.input_tokens;
        self.output_tokens += o.output_tokens;
        self.cache_creation_input_tokens += o.cache_creation_input_tokens;
        self.cache_read_input_tokens += o.cache_read_input_tokens;
    }
}

/// One line of `labels/<run>.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelRecord {
    pub custom_id: String,
    pub run: String,
    pub frame_id: u64,
    pub frame: String,
    /// Second labelling of the same frame with the zone list reordered (SPEC §4.8 step 3.1).
    pub repeat: bool,
    pub label: Option<FrameLabel>,
    /// `succeeded`, `refused`, `invalid`, `errored`, `expired` or `canceled`.
    pub status: String,
    pub error: Option<String>,
    pub model: String,
    pub request_id: Option<String>,
    pub prompt_version: String,
    pub usage: Option<Usage>,
}

/// A human correction from the review gallery (`labels/<run>.fixes.jsonl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fix {
    pub frame_id: u64,
    pub zone: Option<String>,
    pub obstacle: Option<Obstacle>,
    pub game_state: Option<GameState>,
    pub player_lane: Option<String>,
}

pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter(|(_, l)| l.as_ref().map(|l| !l.trim().is_empty()).unwrap_or(true))
        .map(|(i, line)| serde_json::from_str(&line?).with_context(|| format!("{}:{}", path.display(), i + 1)))
        .collect()
}

pub fn labels_path(labels_dir: &Path, run: &str) -> PathBuf {
    labels_dir.join(format!("{run}.jsonl"))
}

pub fn fixes_path(labels_dir: &Path, run: &str) -> PathBuf {
    labels_dir.join(format!("{run}.fixes.jsonl"))
}

/// Run name for a frames directory: `runs/<ts>/frames` → `<ts>`, `recordings/x_frames` → `x_frames`.
/// Restricted to `[A-Za-z0-9_-]` so it can go into a batch `custom_id`.
pub fn run_name(frames_dir: &Path) -> String {
    let dir = if frames_dir.file_name().is_some_and(|n| n == "frames") {
        frames_dir.parent().unwrap_or(frames_dir)
    } else {
        frames_dir
    };
    let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "run".into());
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).take(40).collect()
}

/// Final labels per frame: the first successful non-repeat label, with human fixes applied.
pub fn load_final_labels(labels_dir: &Path, run: &str) -> Result<HashMap<u64, FrameLabel>> {
    let records: Vec<LabelRecord> = read_jsonl(&labels_path(labels_dir, run))?;
    let mut out: HashMap<u64, FrameLabel> = HashMap::new();
    for r in records {
        if r.repeat {
            continue;
        }
        if let Some(l) = r.label {
            out.entry(r.frame_id).or_insert(l);
        }
    }
    let fixes = fixes_path(labels_dir, run);
    if fixes.exists() {
        for fix in read_jsonl::<Fix>(&fixes)? {
            let Some(label) = out.get_mut(&fix.frame_id) else { continue };
            if let Some(s) = fix.game_state {
                label.game_state = s;
            }
            if let Some(l) = fix.player_lane {
                label.player_lane = l;
            }
            if let (Some(zone), Some(o)) = (fix.zone, fix.obstacle) {
                match label.zones.iter_mut().find(|z| z.zone == zone) {
                    Some(z) => {
                        z.obstacle = o;
                        z.sure = true;
                    }
                    None => label.zones.push(ZoneLabel { zone, obstacle: o, coins: false, powerup: false, sure: true }),
                }
            }
        }
    }
    Ok(out)
}

/// FNV-1a over the guide text: a stable prompt version without a hashing dependency.
pub fn prompt_version(guide: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in guide.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", h >> 32)
}

/// Zone listing for the user message; repeats list them in a different order.
pub fn zone_list(reordered: bool) -> String {
    let mut ids: Vec<String> = (0..3).flat_map(|b| (0..3).map(move |l| format!("{}-{}", LANE_NAMES[l], BAND_NAMES[b]))).collect();
    if reordered {
        ids.reverse();
        ids.rotate_left(4);
    }
    ids.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_closed_and_complete() {
        let s = frame_label_schema();
        assert_eq!(s["additionalProperties"], false);
        assert_eq!(s["properties"]["zones"]["items"]["additionalProperties"], false);
        assert_eq!(s["properties"]["game_state"]["enum"].as_array().unwrap().len(), 11);
        assert_eq!(s["properties"]["zones"]["items"]["properties"]["zone"]["enum"].as_array().unwrap().len(), 9);
        let label: FrameLabel = serde_json::from_value(json!({
            "game_state": "Running", "player_lane": "C", "player_action": "running",
            "zones": all_zone_ids().iter().map(|z| json!({"zone": z, "obstacle": "Free", "coins": false, "powerup": false, "sure": true})).collect::<Vec<_>>(),
            "notes": ""
        }))
        .unwrap();
        assert!(label.validate().is_ok());
    }

    #[test]
    fn run_names_are_custom_id_safe() {
        assert_eq!(run_name(Path::new("runs/20261005-101010.123/frames")), "20261005-101010_123");
        assert_eq!(run_name(Path::new("recordings/session1_frames")), "session1_frames");
    }

    #[test]
    fn reordered_zone_list_differs() {
        assert_ne!(zone_list(false), zone_list(true));
        let mut a: Vec<_> = zone_list(false).split(", ").map(String::from).collect();
        let mut b: Vec<_> = zone_list(true).split(", ").map(String::from).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}

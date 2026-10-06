//! `ssbot.toml` (SPEC §11). Every field has a default, so a missing file or section is fine.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub browser: BrowserConfig,
    pub capture: CaptureConfig,
    pub policy: PolicyConfig,
    pub advisor: AdvisorConfig,
    pub recorder: RecorderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserConfig {
    pub viewport: [u32; 2],
    pub profile_dir: String,
    pub url: String,
    /// Poki fullscreen gives a bigger canvas and no side ads (SPEC §10.1). Calibrate in the same mode.
    pub fullscreen: bool,
    /// Open the game iframe's own URL top-level, for when key events don't reach the
    /// cross-origin iframe (SPEC §10 risk 1).
    pub direct: bool,
    /// Path to a Chrome binary; empty means let chromiumoxide find one.
    pub chrome: String,
    /// How long each key is held down.
    pub key_hold_ms: u64,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            viewport: [1280, 720],
            profile_dir: "~/.ssbot/chrome".into(),
            url: "https://poki.com/en/g/subway-surfers".into(),
            fullscreen: true,
            direct: false,
            chrome: String::new(),
            key_hold_ms: 40,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum CaptureBackend {
    Screencast,
    Native,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    pub backend: CaptureBackend,
    pub jpeg_quality: u8,
    pub max_width: u32,
    /// Working size perception runs at (SPEC §4.2: ~320×180).
    pub work_size: [u32; 2],
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self { backend: CaptureBackend::Screencast, jpeg_quality: 60, max_width: 640, work_size: [320, 180] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyConfig {
    pub emergency_ms: u64,
    /// Barriers are only reacted to once they are this close in time. Jumping early lands before
    /// the barrier and the next obstacle catches the runner (logged runs: early jumps died 26%,
    /// near ones 13%). Trains keep the longer `emergency_ms`, since a lane change needs lead time.
    pub barrier_ms: u64,
    /// How long a jump and a roll last, for tracking the runner's own motion (see `MotionTracker`).
    pub jump_ms: u64,
    pub roll_ms: u64,
    pub cooldown_ms: u64,
    pub advisor_max_age_ms: u64,
    /// Obstacles further away in time than this don't constrain the safety mask.
    pub lookahead_ms: u64,
    /// Starting run speed in zone bands per second, before any obstacle has been tracked.
    pub initial_speed_bands_per_s: f32,
    /// Press Space for a hoverboard when nothing else is safe. Off by default: with no
    /// hoverboards in stock, Space opens a "Need hoverboards? 300 coins" shop dialog that
    /// pauses the game (seen live, 2026-10-05).
    pub use_hoverboard: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            emergency_ms: 250,
            barrier_ms: 150,
            jump_ms: 600,
            roll_ms: 600,
            cooldown_ms: 180,
            advisor_max_age_ms: 400,
            lookahead_ms: 700,
            initial_speed_bands_per_s: 4.5,
            use_hoverboard: false,
        }
    }
}

/// How effects are phrased: where the action takes you, or what it then runs into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum EffectStyle {
    /// "moves into a lane with a train close ahead", "jumps toward the low barrier ..."
    Position,
    /// Adds the computed consequence: "jumps over the low barrier ...", "keeps you running
    /// toward the train ...". Still facts from geometry, like Appendix A's "blocks X".
    Consequence,
}

/// Option wording (SPEC §4.5, §9): an effect style and a sentence template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Wording {
    pub style: EffectStyle,
    /// 0: "<Action> is the best move because it <effect>." (Appendix A's best)
    /// 1: "The best move is <action>, because it <effect>."
    pub template: usize,
}

/// Default from `ssbot advisor-bench` on OpenJev 0.8B (2026-10-05): Consequence/1 scored
/// 14/27 raw vs 8–10/27 for the position wordings.
impl Default for Wording {
    fn default() -> Self {
        Wording { style: EffectStyle::Consequence, template: 1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdvisorConfig {
    pub enabled: bool,
    pub model: String,
    pub backend: String,
    pub python: String,
    pub script: String,
    /// Recorded runs whose most frequent facts strings are sent as `warm` at startup.
    pub prewarm_from: Vec<String>,
    pub prewarm_top: usize,
    /// Trained decision head directory (`sidecar/train_head.py`); empty = zero-shot OpenJev.
    pub head: String,
    /// Option wording, chosen by `ssbot advisor-bench` (SPEC §9).
    pub wording: Wording,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "0.8b".into(),
            backend: "mlx".into(),
            python: "../.venv/bin/python".into(),
            script: "sidecar/openjev_sidecar.py".into(),
            prewarm_from: Vec::new(),
            prewarm_top: 50,
            wording: Default::default(),
            head: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecorderConfig {
    pub frame_sample_every: u64,
    pub keep_pre_crash_ms: u64,
    pub runs_dir: String,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self { frame_sample_every: 3, keep_pre_crash_ms: 2000, runs_dir: "runs".into() }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Expands a leading `~/` to the home directory.
pub fn expand_home(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example_parses() {
        let text = r#"
[browser]
viewport = [1280, 720]
profile_dir = "~/.ssbot/chrome"
[capture]
backend = "screencast"
jpeg_quality = 60
max_width = 640
[policy]
emergency_ms = 250
cooldown_ms = 180
advisor_max_age_ms = 400
[advisor]
enabled = true
model = "0.8b"
backend = "mlx"
python = "../.venv/bin/python"
[recorder]
frame_sample_every = 3
keep_pre_crash_ms = 2000
"#;
        let c: Config = toml::from_str(text).unwrap();
        assert_eq!(c.policy.emergency_ms, 250);
        assert_eq!(c.capture.backend, CaptureBackend::Screencast);
        assert_eq!(c.recorder.keep_pre_crash_ms, 2000);
    }
}

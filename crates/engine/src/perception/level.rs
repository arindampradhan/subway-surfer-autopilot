//! The runner's level (SPEC-roofs-and-powerups §4.3): a small CNN on the whole frame, trained
//! offline by `sidecar/level_cnn.py` (same layer layout and JSON as the zone CNN), plus
//! hysteresis. Sprite height alone can't tell a roof from a long jump; the camera height can.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};

use super::Level;
use super::cnn::{ConvLayer, Dense, argmax_softmax, logits};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelCnn {
    kind: String,
    classes: Vec<Level>,
    convs: Vec<ConvLayer>,
    fc1: Dense,
    fc2: Dense,
    #[serde(default)]
    pub note: String,
    /// The frame is resized to `crop`×`crop` (aspect not kept).
    pub crop: usize,
    /// Trained on full-resolution frames rather than the 320×180 work image.
    #[serde(default)]
    pub hires: bool,
}

/// The whole frame resized to `size`×`size`, row-major RGB bytes: the network input, shared with
/// `ssbot level-frames`, which writes the training data.
pub fn frame_input(img: &RgbImage, size: usize) -> Vec<u8> {
    image::imageops::resize(img, size as u32, size as u32, image::imageops::FilterType::Triangle).into_raw()
}

impl LevelCnn {
    pub fn load(path: &Path) -> Result<LevelCnn> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let m: LevelCnn = serde_json::from_str(&text)?;
        ensure!(m.kind == "level-cnn", "not a level CNN");
        ensure!(m.convs.len() == 3 && m.convs[0].cin == 3, "unexpected layer layout");
        let side = m.crop / 8;
        ensure!(m.fc1.din == m.convs[2].cout * side * side, "fc1 input size doesn't match the conv stack");
        ensure!(m.classes.len() == m.fc2.dout, "one class per output");
        Ok(m)
    }

    /// A network that always answers `level`, for tests.
    #[cfg(test)]
    pub(crate) fn constant(level: Level) -> Self {
        let conv = |cin: usize, cout: usize| ConvLayer { w: vec![0.0; cout * cin * 9], b: vec![0.0; cout], cin, cout };
        Self {
            kind: "level-cnn".into(),
            classes: vec![level],
            convs: vec![conv(3, 1), conv(1, 1), conv(1, 1)],
            fc1: Dense { w: vec![0.0; 4], b: vec![0.0], din: 4, dout: 1 },
            fc2: Dense { w: vec![0.0], b: vec![1.0], din: 1, dout: 1 },
            note: String::new(),
            crop: 16,
            hires: false,
        }
    }

    pub fn predict(&self, img: &RgbImage) -> (Level, f32) {
        let (i, p) = argmax_softmax(logits(&self.convs, &self.fc1, &self.fc2, self.crop, &frame_input(img, self.crop)), &[]);
        (self.classes[i], p)
    }
}

/// Smooths raw per-frame levels. Getting onto a roof takes `enter_roof` frames in a row and
/// leaving it `leave_roof`: dropping off a roof is the risky moment, so a one-frame "ground"
/// read mid-roof must not switch the zones back. Ground and Air switch at once, as the sprite
/// check did before.
#[derive(Debug, Clone)]
pub struct LevelTracker {
    current: Level,
    candidate: Level,
    streak: u32,
    pub enter_roof: u32,
    pub leave_roof: u32,
}

impl Default for LevelTracker {
    fn default() -> Self {
        Self { current: Level::Ground, candidate: Level::Ground, streak: 0, enter_roof: 3, leave_roof: 5 }
    }
}

impl LevelTracker {
    pub fn update(&mut self, raw: Level) -> Level {
        if raw == self.candidate {
            self.streak += 1;
        } else {
            self.candidate = raw;
            self.streak = 1;
        }
        let needed = match (self.current, raw) {
            (a, b) if a == b => 0,
            (_, Level::Roof) => self.enter_roof,
            (Level::Roof, _) => self.leave_roof,
            _ => 1,
        };
        if self.streak >= needed {
            self.current = raw;
        }
        self.current
    }

    pub fn reset(&mut self) {
        *self = Self { enter_roof: self.enter_roof, leave_roof: self.leave_roof, ..Self::default() };
    }

    pub fn current(&self) -> Level {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roof_needs_three_frames_in_and_five_out() {
        let mut t = LevelTracker::default();
        assert_eq!(t.update(Level::Roof), Level::Ground);
        assert_eq!(t.update(Level::Roof), Level::Ground);
        assert_eq!(t.update(Level::Roof), Level::Roof);
        // A blip of ground mid-roof is ignored.
        assert_eq!(t.update(Level::Ground), Level::Roof);
        assert_eq!(t.update(Level::Roof), Level::Roof);
        for _ in 0..4 {
            assert_eq!(t.update(Level::Ground), Level::Roof);
        }
        assert_eq!(t.update(Level::Ground), Level::Ground);
    }

    #[test]
    fn ground_and_air_switch_at_once() {
        let mut t = LevelTracker::default();
        assert_eq!(t.update(Level::Air), Level::Air);
        assert_eq!(t.update(Level::Ground), Level::Ground);
    }
}

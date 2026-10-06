//! Calibration (`calibration.toml`): lane zones, player strip, screen markers and every
//! classifier threshold (SPEC §4.3). Geometry is stored in normalised canvas coordinates
//! (0..1 on both axes) so it doesn't depend on the capture size.

use std::path::Path;

use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

use super::GameState;

pub const LANE_NAMES: [&str; 3] = ["L", "C", "R"];
pub const BAND_NAMES: [&str; 3] = ["near", "mid", "far"];

/// Zone id as drawn on overlays and used in labels, e.g. `L-near`.
pub fn zone_id(lane: usize, band: usize) -> String {
    format!("{}-{}", LANE_NAMES[lane], BAND_NAMES[band])
}

pub fn parse_zone_id(id: &str) -> Option<(usize, usize)> {
    let (lane, band) = id.split_once('-')?;
    Some((LANE_NAMES.iter().position(|l| *l == lane)?, BAND_NAMES.iter().position(|b| *b == band)?))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    /// Pixel bounds `(x0, y0, x1, y1)` clamped to an image of `w`×`h`.
    pub fn pixels(&self, w: u32, h: u32) -> (u32, u32, u32, u32) {
        let x0 = ((self.x * w as f32).round() as u32).min(w);
        let y0 = ((self.y * h as f32).round() as u32).min(h);
        let x1 = (((self.x + self.w) * w as f32).round() as u32).clamp(x0, w);
        let y1 = (((self.y + self.h) * h as f32).round() as u32).clamp(y0, h);
        (x0, y0, x1, y1)
    }
}

/// A convex quadrilateral, corners in order: top-left, top-right, bottom-right, bottom-left.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quad(pub [[f32; 2]; 4]);

impl Quad {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        // Same sign of the cross product on every edge means inside a convex polygon.
        let mut sign = 0.0f32;
        for i in 0..4 {
            let [ax, ay] = self.0[i];
            let [bx, by] = self.0[(i + 1) % 4];
            let cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if cross != 0.0 {
                if sign != 0.0 && cross.signum() != sign {
                    return false;
                }
                sign = cross.signum();
            }
        }
        true
    }

    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let xs = self.0.map(|p| p[0]);
        let ys = self.0.map(|p| p[1]);
        let min = |v: [f32; 4]| v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = |v: [f32; 4]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min(xs), min(ys), max(xs), max(ys))
    }

    /// Pixel indices (row-major) inside the quad for an image of `w`×`h`, plus each pixel's
    /// vertical position within the quad (0 = top, 1 = bottom).
    pub fn mask(&self, w: u32, h: u32) -> ZoneMask {
        let (x0, y0, x1, y1) = self.bounds();
        let (px0, py0) = ((x0 * w as f32).floor() as u32, (y0 * h as f32).floor() as u32);
        let (px1, py1) = (((x1 * w as f32).ceil() as u32).min(w), ((y1 * h as f32).ceil() as u32).min(h));
        let mut pixels = Vec::new();
        for py in py0..py1 {
            for px in px0..px1 {
                let (nx, ny) = ((px as f32 + 0.5) / w as f32, (py as f32 + 0.5) / h as f32);
                if self.contains(nx, ny) {
                    let rel_y = ((ny - y0) / (y1 - y0).max(1e-6)).clamp(0.0, 1.0);
                    pixels.push((py * w + px, rel_y));
                }
            }
        }
        ZoneMask { pixels }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ZoneMask {
    pub pixels: Vec<(u32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LaneZones {
    pub near: Quad,
    pub mid: Quad,
    pub far: Quad,
}

impl LaneZones {
    pub fn bands(&self) -> [Quad; 3] {
        [self.near, self.mid, self.far]
    }
}

/// A small screen region that identifies a game state. The template is cropped from
/// `reference` (a saved frame) at `rect`, so re-calibrating means saving a new frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub state: GameState,
    pub rect: Rect,
    pub reference: String,
    /// Mean absolute grey-level difference (0..255) below which the marker matches.
    pub max_diff: f32,
    /// Where to click to leave this screen (e.g. the PLAY button), in normalised coordinates.
    #[serde(default)]
    pub click: Option<[f32; 2]>,
}

/// Every threshold the heuristic classifier uses (SPEC §4.3); `ssbot fit` grid-searches these.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    /// Pixels with saturation below this and value in the background band count as track.
    pub bg_sat_max: f32,
    pub bg_val_min: f32,
    pub bg_val_max: f32,
    /// Zone occupancy (non-background fraction) at or below this is Free.
    pub free_occ_max: f32,
    /// Occupancy at or above this with few edges is a train.
    pub train_occ_min: f32,
    pub train_edge_max: f32,
    /// A train whose top-half occupancy is below `ramp_top_ratio` × bottom-half is a ramp.
    pub ramp_top_ratio: f32,
    /// Fraction of red or white pixels (barrier stripes) that marks a barrier.
    pub barrier_stripe_min: f32,
    /// Barrier occupancy at or above this is a high barrier.
    pub high_barrier_occ_min: f32,
    /// Barrier whose top-half occupancy exceeds `overhead_ratio` × bottom-half is an overhead bar.
    pub overhead_ratio: f32,
    pub coin_frac_min: f32,
    pub powerup_frac_min: f32,
    pub powerup_hue: [f32; 2],
    /// Player detection: saturated pixels in the player strip.
    pub player_sat_min: f32,
    pub player_frac_min: f32,
    /// Sprite centroid this far (fraction of strip height) above its baseline means airborne.
    pub airborne_dy: f32,
    /// Sprite height below this fraction of its standing height means rolling.
    pub roll_height_ratio: f32,
    /// Game-state fallbacks when no marker matches.
    pub black_frac_adbreak: f32,
    pub white_frac_ad: f32,
    /// Whole-frame mean change (0..1) below this while running counts as frozen.
    pub frozen_change_max: f32,
    pub frozen_frames_crash: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            bg_sat_max: 0.30,
            bg_val_min: 0.15,
            bg_val_max: 0.80,
            free_occ_max: 0.18,
            train_occ_min: 0.55,
            train_edge_max: 0.22,
            ramp_top_ratio: 0.55,
            barrier_stripe_min: 0.12,
            high_barrier_occ_min: 0.45,
            overhead_ratio: 1.8,
            coin_frac_min: 0.03,
            powerup_frac_min: 0.02,
            powerup_hue: [270.0, 330.0],
            player_sat_min: 0.45,
            player_frac_min: 0.04,
            airborne_dy: 0.15,
            roll_height_ratio: 0.65,
            black_frac_adbreak: 0.85,
            white_frac_ad: 0.60,
            frozen_change_max: 0.004,
            frozen_frames_crash: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Free text: which layout (page or fullscreen) and date this was made for.
    #[serde(default)]
    pub note: String,
    pub lanes: [LaneZones; 3],
    pub player_strip: Rect,
    /// Player sprite's standing baseline (vertical centroid) and height within the strip, 0..1.
    #[serde(default = "default_baseline")]
    pub player_baseline: f32,
    #[serde(default = "default_height")]
    pub player_height: f32,
    #[serde(default)]
    pub markers: Vec<Marker>,
    /// Where to click to give the canvas focus or to dismiss a screen with no marker click.
    #[serde(default = "default_canvas_center")]
    pub canvas_center: [f32; 2],
    #[serde(default)]
    pub thresholds: Thresholds,
    /// Learned zone classifier (`ssbot train-zones`), relative to this file. When set it
    /// replaces the threshold rules for obstacle classes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone_model: Option<String>,
}

fn default_baseline() -> f32 {
    0.55
}
fn default_height() -> f32 {
    0.6
}
fn default_canvas_center() -> [f32; 2] {
    [0.5, 0.5]
}

impl Default for Calibration {
    /// A perspective guess for the fullscreen 16:9 layout: the vanishing point sits at about
    /// (0.5, 0.30) and the three tracks fill about 0.10..0.90 of the width at the bottom edge.
    /// Replace with Claude-proposed zones (SPEC §4.8 step 1) before trusting it.
    fn default() -> Self {
        let vp = [0.5f32, 0.30];
        let bottom = (0.10f32, 0.90f32);
        // x on the line from the vanishing point to the bottom edge at fraction t of the width.
        let x_at = |t: f32, y: f32| {
            let xb = bottom.0 + (bottom.1 - bottom.0) * t;
            vp[0] + (xb - vp[0]) * (y - vp[1]) / (1.0 - vp[1])
        };
        let band_y = [(0.74f32, 0.90f32), (0.56, 0.72), (0.43, 0.54)];
        let quad = |lane: usize, (y0, y1): (f32, f32)| {
            let (t0, t1) = (lane as f32 / 3.0, (lane + 1) as f32 / 3.0);
            // Shrink each lane slightly so neighbouring obstacles don't bleed in.
            let inset = 0.08 / 3.0;
            let (t0, t1) = (t0 + inset, t1 - inset);
            Quad([[x_at(t0, y0), y0], [x_at(t1, y0), y0], [x_at(t1, y1), y1], [x_at(t0, y1), y1]])
        };
        let lanes = [0, 1, 2].map(|l| LaneZones {
            near: quad(l, band_y[0]),
            mid: quad(l, band_y[1]),
            far: quad(l, band_y[2]),
        });
        Self {
            note: "default perspective guess; not yet calibrated".into(),
            lanes,
            player_strip: Rect { x: 0.15, y: 0.62, w: 0.70, h: 0.36 },
            player_baseline: default_baseline(),
            player_height: default_height(),
            markers: Vec::new(),
            canvas_center: default_canvas_center(),
            thresholds: Thresholds::default(),
            zone_model: None,
        }
    }
}

impl Calibration {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Loads `path`, or falls back to the default geometry with a warning.
    pub fn load_or_default(path: &Path) -> Self {
        match Self::load(path) {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!("{err:#}; using default (uncalibrated) zones");
                Self::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, toml::to_string_pretty(self)?).with_context(|| format!("writing {}", path.display()))
    }

    /// x boundaries between lanes along the player strip, in normalised coordinates:
    /// `[left edge, L|C, C|R, right edge]`, taken from the near zones' bottom edges.
    pub fn lane_edges_at_player(&self) -> [f32; 4] {
        let bl = |q: &Quad| q.0[3][0];
        let br = |q: &Quad| q.0[2][0];
        let near = self.lanes.map(|l| l.near);
        [bl(&near[0]), (br(&near[0]) + bl(&near[1])) / 2.0, (br(&near[1]) + bl(&near[2])) / 2.0, br(&near[2])]
    }
}

/// Per-marker grey template cropped from its reference frame, resized to the working size.
#[derive(Debug, Clone)]
pub struct MarkerTemplate {
    pub marker: Marker,
    pub template: GrayImage,
}

pub fn load_marker_templates(calib: &Calibration, base: &Path, work: (u32, u32)) -> Vec<MarkerTemplate> {
    calib
        .markers
        .iter()
        .filter_map(|m| {
            let path = base.join(&m.reference);
            match image::open(&path) {
                Ok(img) => {
                    let rgb = img.to_rgb8();
                    let rgb = image::imageops::resize(&rgb, work.0, work.1, image::imageops::FilterType::Triangle);
                    Some(MarkerTemplate { marker: m.clone(), template: crop_gray(&rgb, &m.rect) })
                }
                Err(err) => {
                    tracing::warn!("marker {:?}: can't open {}: {err}", m.state, path.display());
                    None
                }
            }
        })
        .collect()
}

pub fn crop_gray(img: &RgbImage, rect: &Rect) -> GrayImage {
    let (x0, y0, x1, y1) = rect.pixels(img.width(), img.height());
    let sub = image::imageops::crop_imm(img, x0, y0, (x1 - x0).max(1), (y1 - y0).max(1)).to_image();
    image::DynamicImage::ImageRgb8(sub).to_luma8()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_contains_and_mask() {
        let q = Quad([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]);
        assert!(q.contains(0.5, 0.5));
        assert!(!q.contains(0.1, 0.5));
        let m = q.mask(100, 100);
        assert_eq!(m.pixels.len(), 50 * 50);
    }

    #[test]
    fn default_zones_are_ordered_and_disjoint() {
        let c = Calibration::default();
        for lane in &c.lanes {
            // near is below mid is below far
            assert!(lane.near.bounds().1 > lane.mid.bounds().3 - 1e-6);
            assert!(lane.mid.bounds().1 > lane.far.bounds().3 - 1e-6);
        }
        let e = c.lane_edges_at_player();
        assert!(e[0] < e[1] && e[1] < e[2] && e[2] < e[3]);
        let text = toml::to_string_pretty(&c).unwrap();
        let back: Calibration = toml::from_str(&text).unwrap();
        assert_eq!(back.lanes, c.lanes);
    }

    #[test]
    fn zone_ids_round_trip() {
        assert_eq!(zone_id(1, 2), "C-far");
        assert_eq!(parse_zone_id("R-near"), Some((2, 0)));
        assert_eq!(parse_zone_id("X-near"), None);
    }
}

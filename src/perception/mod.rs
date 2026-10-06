//! Perception: frame → typed `Observation` (SPEC §4.3). Pure Rust, no ML in v1.

pub mod classify;
pub mod cnn;
pub mod model;
pub mod state;
pub mod zones;

use std::fmt;
use std::path::Path;

use image::RgbImage;
use serde::{Deserialize, Serialize};

use classify::{ZoneFeatures, classify_zone, hsv, zone_features};
use state::{FrameStats, StateDetector, frame_stats};
use zones::{Calibration, MarkerTemplate, ZoneMask, load_marker_templates};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameState {
    Loading,
    AdBreak,
    Ad,
    Menu,
    Running,
    Crashed,
    RevivePrompt,
    NewHighScore,
    ScoreScreen,
    Paused,
    Unknown,
}

impl GameState {
    pub const ALL: [GameState; 11] = [
        Self::Loading,
        Self::AdBreak,
        Self::Ad,
        Self::Menu,
        Self::Running,
        Self::Crashed,
        Self::RevivePrompt,
        Self::NewHighScore,
        Self::ScoreScreen,
        Self::Paused,
        Self::Unknown,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Obstacle {
    Free,
    TrainBody,
    TrainRamp,
    LowBarrier,
    HighBarrier,
    OverheadBar,
    Unknown,
}

impl Obstacle {
    pub const ALL: [Obstacle; 7] = [
        Self::Free,
        Self::TrainBody,
        Self::TrainRamp,
        Self::LowBarrier,
        Self::HighBarrier,
        Self::OverheadBar,
        Self::Unknown,
    ];

    /// Something you crash into if you run into it without the right move.
    pub fn is_hazard(self) -> bool {
        matches!(self, Self::TrainBody | Self::LowBarrier | Self::HighBarrier | Self::OverheadBar)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lane {
    L,
    C,
    R,
}

impl Lane {
    pub const ALL: [Lane; 3] = [Lane::L, Lane::C, Lane::R];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<Lane> {
        Self::ALL.get(i).copied()
    }

    pub fn name(self) -> &'static str {
        match self {
            Lane::L => "left",
            Lane::C => "center",
            Lane::R => "right",
        }
    }
}

impl fmt::Display for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaneView {
    pub near: Obstacle,
    pub mid: Obstacle,
    pub far: Obstacle,
    pub coins: bool,
    pub powerup: bool,
}

impl LaneView {
    pub const FREE: LaneView =
        LaneView { near: Obstacle::Free, mid: Obstacle::Free, far: Obstacle::Free, coins: false, powerup: false };

    pub fn bands(&self) -> [Obstacle; 3] {
        [self.near, self.mid, self.far]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub frame_id: u64,
    pub state: GameState,
    pub player_lane: Option<Lane>,
    pub airborne: bool,
    pub rolling: bool,
    pub lanes: [LaneView; 3],
    pub confidence: f32,
}

impl Observation {
    pub fn running(frame_id: u64, player_lane: Lane, lanes: [LaneView; 3]) -> Self {
        Self {
            frame_id,
            state: GameState::Running,
            player_lane: Some(player_lane),
            airborne: false,
            rolling: false,
            lanes,
            confidence: 1.0,
        }
    }
}

/// The camera follows the runner sideways, so once it settles the screen's middle zones show
/// the runner's own lane, the left zones the lane to its left, the right zones the lane to its
/// right. This maps those screen-relative views onto absolute lanes given the runner's lane.
/// A lane two away from the runner is off-screen and reads as Unknown.
pub fn to_absolute(screen: [LaneView; 3], player: Lane) -> [LaneView; 3] {
    let unknown = LaneView { near: Obstacle::Unknown, mid: Obstacle::Unknown, far: Obstacle::Unknown, coins: false, powerup: false };
    [0i32, 1, 2].map(|abs| {
        let idx = abs - player.index() as i32 + 1;
        if (0..3).contains(&idx) { screen[idx as usize] } else { unknown }
    })
}

/// Everything perception measured on one frame, kept for `fit` and debugging.
#[derive(Debug, Clone)]
pub struct Measured {
    pub obs: Observation,
    pub zones: [[ZoneFeatures; 3]; 3],
    pub stats: FrameStats,
    pub marker_click: Option<[f32; 2]>,
}

pub struct Perceiver {
    pub calib: Calibration,
    model: Option<cnn::Classifier>,
    templates: Vec<MarkerTemplate>,
    masks: Option<((u32, u32), [[ZoneMask; 3]; 3])>,
    /// Zone masks for the full-resolution frame, for models trained on native crops.
    hires_masks: Option<((u32, u32), [[ZoneMask; 3]; 3])>,
    prev: Option<RgbImage>,
    detector: StateDetector,
    /// Last few raw zone classes, for a per-zone majority vote that stops one-frame flicker.
    history: std::collections::VecDeque<[[Obstacle; 3]; 3]>,
}

/// Majority vote over recent frames for each zone; ties go to the newest frame.
fn vote(history: &std::collections::VecDeque<[[Obstacle; 3]; 3]>) -> [[Obstacle; 3]; 3] {
    let newest = *history.back().expect("non-empty");
    [0, 1, 2].map(|l| {
        [0, 1, 2].map(|b| {
            let mut best = (newest[l][b], 0usize);
            for o in Obstacle::ALL {
                let n = history.iter().filter(|h| h[l][b] == o).count();
                if n > best.1 || (n == best.1 && o == newest[l][b]) {
                    best = (o, n);
                }
            }
            best.0
        })
    })
}

impl Perceiver {
    /// `base`: directory marker reference paths are relative to; `work`: working frame size.
    pub fn new(calib: Calibration, base: &Path, work: (u32, u32)) -> Self {
        let templates = load_marker_templates(&calib, base, work);
        let model = calib.zone_model.as_ref().and_then(|m| match cnn::Classifier::load(&base.join(m)) {
            Ok(m) => Some(m),
            Err(err) => {
                tracing::warn!("zone model: {err:#}; using threshold rules");
                None
            }
        });
        Self {
            calib,
            model,
            templates,
            masks: None,
            hires_masks: None,
            prev: None,
            detector: StateDetector::default(),
            history: Default::default(),
        }
    }

    fn masks(&mut self, w: u32, h: u32) -> &[[ZoneMask; 3]; 3] {
        if self.masks.as_ref().map(|(size, _)| *size) != Some((w, h)) {
            let m = self.calib.lanes.map(|lane| lane.bands().map(|q| q.mask(w, h)));
            self.masks = Some(((w, h), m));
        }
        &self.masks.as_ref().unwrap().1
    }

    /// Feature vectors for the learned zone model, `[lane][band]`.
    pub fn zone_vectors(&mut self, img: &RgbImage) -> [[Vec<f32>; 3]; 3] {
        let masks = self.masks(img.width(), img.height()).clone();
        [0, 1, 2].map(|l| [0, 1, 2].map(|b| model::zone_vector(img, &masks[l][b])))
    }

    pub fn perceive(&mut self, frame_id: u64, img: &RgbImage) -> Observation {
        self.measure(frame_id, img).obs
    }

    pub fn measure(&mut self, frame_id: u64, img: &RgbImage) -> Measured {
        self.measure_with(frame_id, img, None)
    }

    /// `hires`: the same frame at full resolution, used by zone models trained on native crops.
    pub fn measure_with(&mut self, frame_id: u64, img: &RgbImage, hires: Option<&RgbImage>) -> Measured {
        let th = self.calib.thresholds;
        let masks = self.masks(img.width(), img.height()).clone();
        let prev = self.prev.as_ref();
        let zones = [0, 1, 2].map(|l| [0, 1, 2].map(|b| zone_features(img, prev, &masks[l][b], &th)));
        let stats = frame_stats(img, prev);

        // A model trained on full-resolution crops looks at the full-resolution frame.
        let hires_img = hires.filter(|_| self.model.as_ref().is_some_and(|m| m.hires()));
        if let Some(h) = hires_img {
            let size = (h.width(), h.height());
            if self.hires_masks.as_ref().map(|(sz, _)| *sz) != Some(size) {
                let m = self.calib.lanes.map(|lane| lane.bands().map(|q| q.mask(size.0, size.1)));
                self.hires_masks = Some((size, m));
            }
        }
        let classes = match &self.model {
            Some(m) => match (hires_img, &self.hires_masks) {
                (Some(h), Some((_, hm))) => [0, 1, 2].map(|l| [0, 1, 2].map(|b| m.predict(h, &hm[l][b], b))),
                _ => [0, 1, 2].map(|l| [0, 1, 2].map(|b| m.predict(img, &masks[l][b], b))),
            },
            None => zones.map(|lane| lane.map(|f| classify_zone(&f, &th))),
        };
        self.history.push_back(classes);
        if self.history.len() > 3 {
            self.history.pop_front();
        }
        let classes = vote(&self.history);
        let lanes = [0, 1, 2].map(|l| LaneView {
            near: classes[l][0],
            mid: classes[l][1],
            far: classes[l][2],
            coins: zones[l].iter().any(|f| f.coin >= th.coin_frac_min),
            powerup: zones[l].iter().any(|f| f.powerup >= th.powerup_frac_min),
        });
        let known = classes.iter().flatten().filter(|c| **c != Obstacle::Unknown).count();
        // Track is visible when most zones read as something the classifier knows.
        let track_visible = known >= 6;
        let state = self.detector.detect(img, stats, &self.templates, track_visible, &th);
        let marker_click = state::match_marker(img, &self.templates).and_then(|t| t.marker.click);
        let player = self.player(img);
        self.prev = Some(img.clone());

        let running = state == GameState::Running;
        let obs = Observation {
            frame_id,
            state,
            player_lane: player.map(|p| p.lane),
            airborne: running && player.is_some_and(|p| p.airborne),
            rolling: running && player.is_some_and(|p| p.rolling),
            lanes,
            confidence: known as f32 / 9.0 * if player.is_some() { 1.0 } else { 0.5 },
        };
        Measured { obs, zones, stats, marker_click }
    }

    /// Finds the runner in the player strip as the lane with the most saturated, non-track
    /// pixels, then reads jump and roll from the sprite's vertical centroid and height.
    fn player(&self, img: &RgbImage) -> Option<PlayerFix> {
        let th = &self.calib.thresholds;
        let (w, h) = img.dimensions();
        let (x0, y0, x1, y1) = self.calib.player_strip.pixels(w, h);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let edges = self.calib.lane_edges_at_player().map(|e| (e * w as f32) as u32);
        let mut count = [0u32; 3];
        let mut rows: [Vec<u32>; 3] = Default::default();
        for y in y0..y1 {
            for x in x0..x1 {
                let p = img.get_pixel(x, y);
                let (_, s, v) = hsv(p[0], p[1], p[2]);
                if s < th.player_sat_min || v < 0.25 {
                    continue;
                }
                let lane = if x < edges[1] { 0 } else if x < edges[2] { 1 } else { 2 };
                count[lane] += 1;
                rows[lane].push(y - y0);
            }
        }
        let (lane, &n) = count.iter().enumerate().max_by_key(|(_, n)| **n)?;
        let lane_px = ((edges[lane + 1].saturating_sub(edges[lane])).max(1) * (y1 - y0)) as f32;
        if (n as f32) < th.player_frac_min * lane_px {
            return None;
        }
        let strip_h = (y1 - y0) as f32;
        let r = &mut rows[lane];
        r.sort_unstable();
        let centroid = r.iter().map(|&y| y as f32).sum::<f32>() / r.len() as f32 / strip_h;
        // 5th..95th percentile span, so stray pixels don't stretch the sprite.
        let span = (r[r.len() * 95 / 100] - r[r.len() * 5 / 100]) as f32 / strip_h;
        Some(PlayerFix {
            lane: Lane::from_index(lane)?,
            airborne: centroid < self.calib.player_baseline - th.airborne_dy,
            rolling: span < self.calib.player_height * th.roll_height_ratio,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct PlayerFix {
    lane: Lane,
    airborne: bool,
    rolling: bool,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use image::Rgb;

    pub const TRACK: Rgb<u8> = Rgb([110, 100, 95]);

    /// Paints a synthetic 320×180 frame: grey track, plus a solid fill in the given zones.
    pub fn synthetic(calib: &Calibration, fills: &[(usize, usize, Rgb<u8>)]) -> RgbImage {
        let (w, h) = (320, 180);
        let mut img = RgbImage::from_pixel(w, h, TRACK);
        for &(lane, band, colour) in fills {
            for (idx, _) in calib.lanes[lane].bands()[band].mask(w, h).pixels {
                img.put_pixel(idx % w, idx / w, colour);
            }
        }
        img
    }

    #[test]
    fn synthetic_trains_and_coins() {
        let calib = Calibration::default();
        let mut p = Perceiver::new(calib.clone(), Path::new("."), (320, 180));
        let blue_train = Rgb([30, 60, 200]);
        let coin = Rgb([255, 215, 0]);
        let img = synthetic(&calib, &[(0, 0, blue_train), (0, 1, blue_train), (2, 1, coin)]);
        let obs = p.perceive(1, &img);
        assert_eq!(obs.state, GameState::Running);
        assert_eq!(obs.lanes[0].near, Obstacle::TrainBody);
        assert_eq!(obs.lanes[0].mid, Obstacle::TrainBody);
        assert_eq!(obs.lanes[1].near, Obstacle::Free);
        assert!(obs.lanes[2].coins);
        assert!(!obs.lanes[1].coins);
    }

    #[test]
    fn screen_views_map_to_absolute_lanes() {
        let mk = |o| LaneView { near: o, ..LaneView::FREE };
        let screen = [mk(Obstacle::TrainBody), mk(Obstacle::LowBarrier), mk(Obstacle::Free)];
        // Runner in the centre: screen = absolute.
        assert_eq!(to_absolute(screen, Lane::C), screen);
        // Runner in the right lane: middle zones are its lane, left zones the centre lane.
        let r = to_absolute(screen, Lane::R);
        assert_eq!(r[2].near, Obstacle::LowBarrier);
        assert_eq!(r[1].near, Obstacle::TrainBody);
        assert_eq!(r[0].near, Obstacle::Unknown);
        let l = to_absolute(screen, Lane::L);
        assert_eq!(l[0].near, Obstacle::LowBarrier);
        assert_eq!(l[1].near, Obstacle::Free);
        assert_eq!(l[2].near, Obstacle::Unknown);
    }

    #[test]
    fn one_frame_flicker_is_voted_out() {
        use std::collections::VecDeque;
        let free = [[Obstacle::Free; 3]; 3];
        let mut blip = free;
        blip[2][0] = Obstacle::TrainBody;
        let h: VecDeque<_> = [free, free, blip].into_iter().collect();
        assert_eq!(vote(&h)[2][0], Obstacle::Free);
        let h: VecDeque<_> = [free, blip, blip].into_iter().collect();
        assert_eq!(vote(&h)[2][0], Obstacle::TrainBody, "two frames in a row is real");
    }

    #[test]
    fn finds_player_lane() {
        let calib = Calibration::default();
        let mut p = Perceiver::new(calib.clone(), Path::new("."), (320, 180));
        let mut img = synthetic(&calib, &[]);
        // A saturated sprite in the right lane's part of the player strip.
        let e = calib.lane_edges_at_player();
        let (x0, x1) = (((e[2] + 0.04) * 320.0) as u32, ((e[3] - 0.04) * 320.0) as u32);
        for y in 120..170 {
            for x in x0..x1 {
                img.put_pixel(x, y, Rgb([220, 40, 40]));
            }
        }
        let obs = p.perceive(1, &img);
        assert_eq!(obs.player_lane, Some(Lane::R));
    }
}

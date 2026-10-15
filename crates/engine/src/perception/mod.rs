//! Perception: frame → typed `Observation` (SPEC §4.3). Pure Rust, no ML in v1.

pub mod classify;
pub mod cnn;
pub mod hud;
pub mod level;
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
    /// A walkable train roof at the runner's height (roof zone set only).
    TrainRoof,
    /// The roof the runner is on ends here; the lane drops to the track (roof zone set only).
    RoofEnd,
}

impl Obstacle {
    pub const ALL: [Obstacle; 9] = [
        Self::Free,
        Self::TrainBody,
        Self::TrainRamp,
        Self::LowBarrier,
        Self::HighBarrier,
        Self::OverheadBar,
        Self::Unknown,
        Self::TrainRoof,
        Self::RoofEnd,
    ];

    /// Something you crash into if you run into it without the right move.
    pub fn is_hazard(self) -> bool {
        matches!(self, Self::TrainBody | Self::LowBarrier | Self::HighBarrier | Self::OverheadBar)
    }
}

/// Where the runner is, vertically. On a roof the camera rises and the ground zones look at roofs
/// and buildings, so perception switches to the roof zone set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Level {
    #[default]
    Ground,
    Roof,
    Air,
}

impl Level {
    pub const ALL: [Level; 3] = [Self::Ground, Self::Roof, Self::Air];
}

/// The power-up whose timer shows in the bottom-left HUD. `Other` means the timer is showing but
/// the icon matches no reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerUp {
    Jetpack,
    SuperSneakers,
    PogoStick,
    Magnet,
    Multiplier,
    Hoverboard,
    Other,
}

impl PowerUp {
    pub const ALL: [PowerUp; 7] =
        [Self::Jetpack, Self::SuperSneakers, Self::PogoStick, Self::Magnet, Self::Multiplier, Self::Hoverboard, Self::Other];

    /// Lifts the runner off the track: nothing on it can be hit.
    pub fn flies(self) -> bool {
        matches!(self, Self::Jetpack | Self::PogoStick)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Jetpack => "jetpack",
            Self::SuperSneakers => "super sneakers",
            Self::PogoStick => "pogo stick",
            Self::Magnet => "coin magnet",
            Self::Multiplier => "score multiplier",
            Self::Hoverboard => "hoverboard",
            Self::Other => "unrecognised",
        }
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
    /// `level == Air`, or a jump the sprite shows; kept for runs recorded before `level`.
    pub airborne: bool,
    pub rolling: bool,
    pub lanes: [LaneView; 3],
    pub confidence: f32,
    /// Runs recorded before this field read as `Ground`.
    #[serde(default)]
    pub level: Level,
    #[serde(default)]
    pub active_powerup: Option<PowerUp>,
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
            level: Level::Ground,
            active_powerup: None,
        }
    }

    pub fn on_roof(&self) -> bool {
        self.level == Level::Roof
    }

    /// Jetpack or pogo stick: the runner is out of reach of anything on the track.
    pub fn flying(&self) -> bool {
        self.active_powerup.is_some_and(PowerUp::flies)
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

/// Zone masks `[lane][band]` together with the frame size and zone set (roof or not) they were
/// built for.
type SizedMasks = ((u32, u32, bool), [[ZoneMask; 3]; 3]);

pub struct Perceiver {
    pub calib: Calibration,
    model: Option<cnn::Classifier>,
    level_model: Option<level::LevelCnn>,
    levels: level::LevelTracker,
    hud: hud::Hud,
    templates: Vec<MarkerTemplate>,
    masks: Option<SizedMasks>,
    /// Zone masks for the full-resolution frame, for models trained on native crops.
    hires_masks: Option<SizedMasks>,
    prev: Option<RgbImage>,
    detector: StateDetector,
    /// Last few raw zone classes, for a per-zone majority vote that stops one-frame flicker.
    history: std::collections::VecDeque<[[Obstacle; 3]; 3]>,
    /// Last few HUD reads, for the same vote.
    hud_history: std::collections::VecDeque<Option<PowerUp>>,
    /// Whether the last frame used the roof zone set, and frames left before roof zones are trusted.
    roof_zones: bool,
    settle: u32,
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

/// Majority vote over recent HUD reads; ties go to the newest.
fn vote_hud(history: &std::collections::VecDeque<Option<PowerUp>>) -> Option<PowerUp> {
    let newest = *history.back().expect("non-empty");
    let count = |v: Option<PowerUp>| history.iter().filter(|h| **h == v).count();
    history.iter().copied().fold(newest, |best, v| if count(v) > count(best) { v } else { best })
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
        let level_model = calib.level_model.as_ref().and_then(|m| match level::LevelCnn::load(&base.join(m)) {
            Ok(m) => Some(m),
            Err(err) => {
                tracing::warn!("level model: {err:#}; reading only ground and air from the sprite");
                None
            }
        });
        let hud = hud::Hud::new(calib.hud.clone(), base);
        Self {
            calib,
            model,
            level_model,
            levels: Default::default(),
            hud,
            templates,
            masks: None,
            hires_masks: None,
            prev: None,
            detector: StateDetector::default(),
            history: Default::default(),
            hud_history: Default::default(),
            roof_zones: false,
            settle: 0,
        }
    }

    fn zone_set(&self, roof: bool) -> [zones::LaneZones; 3] {
        self.calib.lanes_for(if roof { Level::Roof } else { Level::Ground })
    }

    fn masks(&mut self, w: u32, h: u32) -> &[[ZoneMask; 3]; 3] {
        let key = (w, h, self.roof_zones);
        if self.masks.as_ref().map(|(k, _)| *k) != Some(key) {
            let m = self.zone_set(self.roof_zones).map(|lane| lane.bands().map(|q| q.mask(w, h)));
            self.masks = Some((key, m));
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
        // Level first: it picks the zone set.
        let player = self.player(img);
        let raw_level = match &self.level_model {
            Some(m) => m.predict(hires.filter(|_| m.hires).unwrap_or(img)).0,
            None if player.is_some_and(|p| p.airborne) => Level::Air,
            None => Level::Ground,
        };
        let level = self.levels.update(raw_level);
        let roof = level == Level::Roof && self.calib.roof_lanes.is_some();
        if roof != self.roof_zones {
            // Classes from the other zone set mustn't vote with these.
            self.history.clear();
            self.roof_zones = roof;
            // Blank the zones only while switching to a roof set with other geometry: with the
            // same quads (as calibrated in 2026-10) the classes stay comparable, and Unknown
            // zones would let the reflex walk into whatever they hide.
            let moved = self.calib.roof_lanes.is_some_and(|r| r != self.calib.lanes);
            self.settle = if roof && moved { th.roof_settle_frames } else { 0 };
        }
        let masks = self.masks(img.width(), img.height()).clone();
        let prev = self.prev.as_ref();
        let zones = [0, 1, 2].map(|l| [0, 1, 2].map(|b| zone_features(img, prev, &masks[l][b], &th)));
        let stats = frame_stats(img, prev);

        // A model trained on full-resolution crops looks at the full-resolution frame.
        let hires_img = hires.filter(|_| self.model.as_ref().is_some_and(|m| m.hires()));
        if let Some(h) = hires_img {
            let key = (h.width(), h.height(), self.roof_zones);
            if self.hires_masks.as_ref().map(|(k, _)| *k) != Some(key) {
                let m = self.zone_set(self.roof_zones).map(|lane| lane.bands().map(|q| q.mask(key.0, key.1)));
                self.hires_masks = Some((key, m));
            }
        }
        let classes = match &self.model {
            Some(m) => match (hires_img, &self.hires_masks) {
                (Some(h), Some((_, hm))) => [0, 1, 2].map(|l| [0, 1, 2].map(|b| m.predict(h, &hm[l][b], b))),
                _ => [0, 1, 2].map(|l| [0, 1, 2].map(|b| m.predict(img, &masks[l][b], b))),
            },
            None => zones.map(|lane| lane.map(|f| classify_zone(&f, &th))),
        };
        // Track is visible when most zones read as something the classifier knows (before the
        // roof settle period blanks them, so state detection doesn't flicker on a climb).
        let known = classes.iter().flatten().filter(|c| **c != Obstacle::Unknown).count();
        self.history.push_back(classes);
        if self.history.len() > 3 {
            self.history.pop_front();
        }
        let classes = if self.settle > 0 {
            self.settle -= 1;
            [[Obstacle::Unknown; 3]; 3]
        } else {
            vote(&self.history)
        };
        let lanes = [0, 1, 2].map(|l| LaneView {
            near: classes[l][0],
            mid: classes[l][1],
            far: classes[l][2],
            coins: zones[l].iter().any(|f| f.coin >= th.coin_frac_min),
            powerup: zones[l].iter().any(|f| f.powerup >= th.powerup_frac_min),
        });
        let track_visible = known >= 6;
        let state = self.detector.detect(img, stats, &self.templates, track_visible, &th);
        let marker_click = state::match_marker(img, &self.templates).and_then(|t| t.marker.click);
        // The timer's thin bar and small icon read best at full resolution.
        self.hud_history.push_back(self.hud.detect(hires.unwrap_or(img)));
        if self.hud_history.len() > 3 {
            self.hud_history.pop_front();
        }
        let active_powerup = vote_hud(&self.hud_history);
        self.prev = Some(img.clone());

        let running = state == GameState::Running;
        if !running {
            self.levels.reset();
        }
        // The sprite sits high on a roof, so its "airborne" read is only trusted on the ground.
        let airborne = running && (level == Level::Air || (level == Level::Ground && player.is_some_and(|p| p.airborne)));
        let obs = Observation {
            frame_id,
            state,
            player_lane: player.map(|p| p.lane),
            airborne,
            rolling: running && player.is_some_and(|p| p.rolling),
            lanes,
            confidence: known as f32 / 9.0 * if player.is_some() { 1.0 } else { 0.5 },
            level: match (running, level) {
                (false, _) => Level::Ground,
                (true, Level::Ground) if airborne => Level::Air,
                (true, l) => l,
            },
            active_powerup: if running { active_powerup } else { None },
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

    fn roof_perceiver(roof_lanes: [zones::LaneZones; 3]) -> Perceiver {
        let calib = Calibration { roof_lanes: Some(roof_lanes), ..Calibration::default() };
        let mut p = Perceiver::new(calib, Path::new("."), (320, 180));
        p.level_model = Some(level::LevelCnn::constant(Level::Roof));
        p
    }

    #[test]
    fn roof_level_needs_three_frames_and_same_geometry_keeps_the_zones() {
        let calib = Calibration::default();
        let mut p = roof_perceiver(calib.lanes);
        let img = synthetic(&calib, &[(0, 0, Rgb([30, 60, 200]))]);
        let levels: Vec<Level> = (0..4).map(|i| p.perceive(i, &img).level).collect();
        assert_eq!(levels, [Level::Ground, Level::Ground, Level::Roof, Level::Roof]);
        // The roof set has the ground geometry: no blanking, the train stays visible.
        assert_eq!(p.perceive(5, &img).lanes[0].near, Obstacle::TrainBody);
    }

    #[test]
    fn moved_roof_zones_read_unknown_while_the_camera_settles() {
        let calib = Calibration::default();
        let mut shifted = calib.lanes;
        for lane in &mut shifted {
            for q in [&mut lane.near, &mut lane.mid, &mut lane.far] {
                for c in &mut q.0 {
                    c[1] -= 0.02;
                }
            }
        }
        let mut p = roof_perceiver(shifted);
        // Alternate the track shade: identical frames would read as a frozen (crashed) screen.
        let frames = [synthetic(&calib, &[]), RgbImage::from_pixel(320, 180, Rgb([122, 104, 98]))];
        let obs: Vec<Observation> = (0..30).map(|i| p.perceive(i, &frames[i as usize % 2])).collect();
        assert!(obs.iter().all(|o| o.state == GameState::Running));
        // On the roof from the third frame: then 18 frames of Unknown, then the roof zones.
        assert!(obs[2..20].iter().all(|o| o.lanes.iter().all(|l| l.near == Obstacle::Unknown)));
        assert!(obs[25].lanes.iter().all(|l| l.near != Obstacle::Unknown));
    }

    #[test]
    fn hud_timer_reaches_the_observation() {
        let calib = Calibration::default();
        let mut p = Perceiver::new(calib.clone(), Path::new("."), (320, 180));
        let mut img = synthetic(&calib, &[]);
        let paint = |img: &mut RgbImage, r: zones::Rect, c: Rgb<u8>| {
            let (x0, y0, x1, y1) = r.pixels(320, 180);
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, c);
                }
            }
        };
        paint(&mut img, calib.hud.bar, Rgb([245, 245, 245]));
        paint(&mut img, calib.hud.icon, Rgb([40, 120, 230]));
        let reads: Vec<Option<PowerUp>> = (0..3).map(|i| p.perceive(i, &img).active_powerup).collect();
        assert_eq!(reads.last().copied().flatten(), Some(PowerUp::Other), "{reads:?}");
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

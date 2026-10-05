//! Reflex layer (SPEC §4.5): per-frame Rust rules. Estimates time to impact from how fast
//! obstacles move between zone bands, builds the safety mask every action must pass, and makes
//! emergency dodges.

use serde::{Deserialize, Serialize};

use super::{Action, ActionSet, possible_actions};
use crate::config::PolicyConfig;
use crate::perception::{GameState, Lane, LaneView, Obstacle, Observation};

/// Distance from the runner to the start of each band, in band lengths.
const BAND_DISTANCE: [f32; 3] = [0.5, 1.5, 2.5];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReflexOut {
    pub mask: Vec<Action>,
    pub emergency: bool,
    pub action: Action,
    pub tti_ms: Option<f32>,
    pub speed: f32,
    /// Best ordinary move when `action` is the hoverboard but it was used recently: the move
    /// whose first unavoidable hazard is furthest away.
    #[serde(default = "stay")]
    pub fallback: Action,
}

fn stay() -> Action {
    Action::Stay
}

pub struct Reflex {
    cfg: PolicyConfig,
    /// Run speed in bands per second (EMA of observed band-to-band transitions).
    speed: f32,
    /// Per lane: band of the first hazard and when it was first seen there (ms).
    tracked: [Option<(usize, f64)>; 3],
}

impl Reflex {
    pub fn new(cfg: PolicyConfig) -> Self {
        let speed = cfg.initial_speed_bands_per_s;
        Self { cfg, speed, tracked: [None; 3] }
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Resets the speed estimate at the start of a new run.
    pub fn reset(&mut self) {
        self.speed = self.cfg.initial_speed_bands_per_s;
        self.tracked = [None; 3];
    }

    /// Updates the speed estimate from obstacles moving one band closer between frames.
    pub fn update(&mut self, obs: &Observation, t_ms: f64) {
        if obs.state != GameState::Running {
            self.tracked = [None; 3];
            return;
        }
        for (lane, view) in obs.lanes.iter().enumerate() {
            let band = view.bands().iter().position(|o| o.is_hazard() || *o == Obstacle::TrainRamp);
            match (self.tracked[lane], band) {
                (Some((prev, since)), Some(b)) if b + 1 == prev => {
                    let dt = (t_ms - since) / 1000.0;
                    // Ignore implausible intervals (detector flicker, dropped frames).
                    if (0.05..=3.0).contains(&dt) {
                        let observed = 1.0 / dt as f32;
                        self.speed = 0.7 * self.speed + 0.3 * observed;
                    }
                    self.tracked[lane] = Some((b, t_ms));
                }
                (Some((prev, _)), Some(b)) if b == prev => {}
                (_, Some(b)) => self.tracked[lane] = Some((b, t_ms)),
                (_, None) => self.tracked[lane] = None,
            }
        }
    }

    pub fn tti_ms(&self, band: usize) -> f32 {
        BAND_DISTANCE[band] / self.speed.max(0.1) * 1000.0
    }

    /// Whether `action` survives every obstacle within `window_ms`, given what the runner is doing.
    fn survives(&self, obs: &Observation, lane: Lane, action: Action, window_ms: f32) -> bool {
        if action == Action::Hoverboard {
            return true;
        }
        let Some(target) = action.target_lane(lane) else { return false };
        let view = &obs.lanes[target.index()];
        let jumping = action == Action::Jump || obs.airborne;
        let rolling = action == Action::Roll || obs.rolling;
        let mut on_ramp = false;
        for (band, o) in view.bands().into_iter().enumerate() {
            let tti = self.tti_ms(band);
            if tti > window_ms {
                break;
            }
            let handled_later = tti > self.cfg.emergency_ms as f32;
            let ok = match o {
                Obstacle::Free | Obstacle::Unknown => true,
                Obstacle::TrainRamp => {
                    on_ramp = true;
                    true
                }
                // Running up a ramp puts you on top of the train; anything else hits it.
                Obstacle::TrainBody => on_ramp,
                Obstacle::LowBarrier => handled_later || jumping || rolling,
                Obstacle::HighBarrier | Obstacle::OverheadBar => handled_later || rolling,
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// The set of actions that won't crash within the look-ahead window (SPEC §4.5).
    /// Falls back to `{Hoverboard}` when nothing else is safe.
    pub fn safety_mask(&self, obs: &Observation) -> ActionSet {
        let Some(lane) = obs.player_lane else { return [Action::Stay].into_iter().collect() };
        let window = self.cfg.lookahead_ms as f32;
        let mask: ActionSet = possible_actions(obs).iter().filter(|a| self.survives(obs, lane, *a, window)).collect();
        if mask.is_empty() { [Action::Hoverboard].into_iter().collect() } else { mask }
    }

    /// Time to the first hazard in the player's lane.
    pub fn tti_current(&self, obs: &Observation) -> Option<f32> {
        let view = obs.lanes[obs.player_lane?.index()];
        view.bands().iter().position(|o| o.is_hazard()).map(|b| self.tti_ms(b))
    }

    /// True when staying put crashes within the emergency window.
    pub fn emergency(&self, obs: &Observation) -> bool {
        let Some(lane) = obs.player_lane else { return false };
        obs.state == GameState::Running && !self.survives(obs, lane, Action::Stay, self.cfg.emergency_ms as f32)
    }

    /// Must-act rules in priority order (SPEC §4.5): train → nearest free lane; low barrier →
    /// jump or roll; high barrier or overhead bar → roll. Always returns a masked action.
    pub fn emergency_action(&self, obs: &Observation, mask: ActionSet) -> Action {
        let Some(lane) = obs.player_lane else { return Action::Stay };
        let threat = obs.lanes[lane.index()].bands().into_iter().find(|o| o.is_hazard());
        let lane_changes = self.lane_changes_by_preference(obs, lane);
        let preferred: Vec<Action> = match threat {
            Some(Obstacle::TrainBody) => lane_changes.into_iter().chain([Action::Jump, Action::Roll]).collect(),
            Some(Obstacle::LowBarrier) => [Action::Jump, Action::Roll].into_iter().chain(lane_changes).collect(),
            Some(Obstacle::HighBarrier | Obstacle::OverheadBar) => {
                [Action::Roll].into_iter().chain(lane_changes).chain([Action::Jump]).collect()
            }
            _ => lane_changes.into_iter().chain([Action::Jump, Action::Roll]).collect(),
        };
        preferred.into_iter().chain([Action::Stay, Action::Hoverboard]).find(|a| mask.contains(*a)).unwrap_or(Action::Hoverboard)
    }

    /// Left/Right ordered by how clear the target lane is (free or ramp at near and mid first).
    fn lane_changes_by_preference(&self, obs: &Observation, lane: Lane) -> Vec<Action> {
        let mut v: Vec<(Action, i32)> = [Action::Left, Action::Right]
            .into_iter()
            .filter_map(|a| {
                let t = a.target_lane(lane)?;
                let view = &obs.lanes[t.index()];
                let clear = |o: Obstacle| matches!(o, Obstacle::Free | Obstacle::TrainRamp);
                Some((a, clear(view.near) as i32 * 2 + clear(view.mid) as i32 + view.coins as i32))
            })
            .collect();
        v.sort_by_key(|(_, score)| -score);
        v.into_iter().map(|(a, _)| a).collect()
    }

    /// Best safe action by rule when there's no fresh advice: prefer clear lanes with coins,
    /// stay centred, and don't move without a reason.
    pub fn default_action(&self, obs: &Observation, mask: ActionSet) -> Action {
        let Some(lane) = obs.player_lane else { return Action::Stay };
        let score = |a: Action| -> f32 {
            let Some(t) = a.target_lane(lane) else { return f32::NEG_INFINITY };
            let view: &LaneView = &obs.lanes[t.index()];
            let mut s = 0.0;
            s += view.bands().iter().filter(|o| **o == Obstacle::Free).count() as f32 * 0.5;
            s -= view.bands().iter().filter(|o| o.is_hazard()).count() as f32 * 0.4;
            s -= view.bands().iter().filter(|o| **o == Obstacle::Unknown).count() as f32 * 0.3;
            s += if view.coins { 1.0 } else { 0.0 } + if view.powerup { 1.0 } else { 0.0 };
            s += if t == Lane::C { 0.2 } else { 0.0 };
            s += match a {
                Action::Stay => 0.5,
                Action::Jump | Action::Roll => -1.0,
                Action::Hoverboard => -10.0,
                _ => 0.0,
            };
            s
        };
        let best = mask.iter().max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap_or(Action::Stay);
        // Staying is allowed and the best move is only slightly better: stay. Lane changes on
        // small differences make the runner flip back and forth as the view shifts.
        if mask.contains(Action::Stay) && score(best) - score(Action::Stay) < 0.25 { Action::Stay } else { best }
    }

    /// The ordinary move that survives longest: the largest window it still survives, with
    /// the emergency priorities breaking ties.
    pub fn least_bad(&self, obs: &Observation) -> Action {
        let Some(lane) = obs.player_lane else { return Action::Stay };
        let candidates: Vec<Action> = possible_actions(obs).iter().collect();
        let horizon = |a: Action| -> f32 {
            let mut lo = 0.0;
            for w in [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0] {
                if self.survives(obs, lane, a, w) {
                    lo = w;
                } else {
                    break;
                }
            }
            lo
        };
        let best = candidates.iter().map(|a| horizon(*a)).fold(0.0, f32::max);
        let tied: ActionSet = candidates.into_iter().filter(|a| horizon(*a) >= best).collect();
        let pick = self.emergency_action(obs, tied);
        if pick == Action::Hoverboard { Action::Stay } else { pick }
    }

    pub fn evaluate(&self, obs: &Observation) -> ReflexOut {
        let mask = self.safety_mask(obs);
        let emergency = self.emergency(obs);
        let action = if emergency { self.emergency_action(obs, mask) } else { self.default_action(obs, mask) };
        let fallback = if action == Action::Hoverboard { self.least_bad(obs) } else { action };
        ReflexOut { mask: mask.iter().collect(), emergency, action, tti_ms: self.tti_current(obs), speed: self.speed, fallback }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reflex() -> Reflex {
        Reflex::new(PolicyConfig::default())
    }

    fn lanes(l: LaneView, c: LaneView, r: LaneView) -> [LaneView; 3] {
        [l, c, r]
    }

    const FREE: LaneView = LaneView::FREE;

    fn near(o: Obstacle) -> LaneView {
        LaneView { near: o, ..FREE }
    }
    fn mid(o: Obstacle) -> LaneView {
        LaneView { mid: o, ..FREE }
    }

    #[test]
    fn train_ahead_switches_to_free_lane() {
        let r = reflex();
        let obs = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::TrainBody), FREE));
        let mask = r.safety_mask(&obs);
        assert!(r.emergency(&obs));
        assert!(!mask.contains(Action::Stay) && !mask.contains(Action::Left) && !mask.contains(Action::Jump));
        assert!(mask.contains(Action::Right));
        assert_eq!(r.emergency_action(&obs, mask), Action::Right);
    }

    #[test]
    fn train_in_two_lanes_from_edge() {
        // Left lane, trains in left and centre: the only way out is two lanes away, so nothing
        // ordinary is safe and the hoverboard is the last resort.
        let r = reflex();
        let obs = Observation::running(1, Lane::L, lanes(near(Obstacle::TrainBody), near(Obstacle::TrainBody), FREE));
        let mask = r.safety_mask(&obs);
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![Action::Hoverboard]);
        assert_eq!(r.evaluate(&obs).action, Action::Hoverboard);
        // Once the hoverboard is used up, the fallback is still an ordinary move.
        assert_ne!(r.evaluate(&obs).fallback, Action::Hoverboard);
    }

    #[test]
    fn least_bad_prefers_the_later_hazard() {
        let r = reflex();
        // Trains close ahead in left and centre, a train only at mid distance on the right:
        // moving right buys time.
        let obs = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::TrainBody), mid(Obstacle::TrainBody)));
        let out = r.evaluate(&obs);
        assert_eq!(out.action, Action::Hoverboard);
        assert_eq!(out.fallback, Action::Right);
    }

    #[test]
    fn ramp_lane_is_an_escape() {
        let r = reflex();
        let ramp_then_train = LaneView { near: Obstacle::TrainRamp, mid: Obstacle::TrainBody, ..FREE };
        let obs = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::TrainBody), ramp_then_train));
        assert_eq!(r.evaluate(&obs).action, Action::Right);
    }

    #[test]
    fn low_barrier_jumps_high_barrier_rolls() {
        let r = reflex();
        let low = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::LowBarrier), near(Obstacle::TrainBody)));
        let out = r.evaluate(&low);
        assert!(out.emergency);
        assert_eq!(out.action, Action::Jump);
        let high = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::HighBarrier), near(Obstacle::TrainBody)));
        assert_eq!(r.evaluate(&high).action, Action::Roll);
        let bar = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::OverheadBar), near(Obstacle::TrainBody)));
        assert!(!r.safety_mask(&bar).contains(Action::Jump));
    }

    #[test]
    fn barrier_while_airborne_is_already_handled() {
        let r = reflex();
        let mut obs = Observation::running(1, Lane::C, lanes(FREE, near(Obstacle::LowBarrier), FREE));
        obs.airborne = true;
        assert!(!r.emergency(&obs));
        let mask = r.safety_mask(&obs);
        assert!(mask.contains(Action::Stay));
        assert!(!mask.contains(Action::Jump), "no double jump");
        // Airborne doesn't clear a high barrier.
        obs.lanes[1] = near(Obstacle::HighBarrier);
        assert!(r.emergency(&obs));
        assert_eq!(r.evaluate(&obs).action, Action::Roll);
    }

    #[test]
    fn far_obstacles_are_not_emergencies() {
        let r = reflex();
        let obs = Observation::running(1, Lane::C, lanes(FREE, mid(Obstacle::LowBarrier), FREE));
        assert!(!r.emergency(&obs));
        // A barrier at mid distance can still be jumped later, so staying is allowed.
        assert!(r.safety_mask(&obs).contains(Action::Stay));
        // A train at mid distance can't be: staying is masked out and the default moves early.
        let obs = Observation::running(1, Lane::C, lanes(FREE, mid(Obstacle::TrainBody), FREE));
        assert!(!r.emergency(&obs));
        assert!(!r.safety_mask(&obs).contains(Action::Stay));
        assert!(matches!(r.evaluate(&obs).action, Action::Left | Action::Right));
    }

    #[test]
    fn default_prefers_coins_then_staying() {
        let r = reflex();
        let coins = LaneView { coins: true, ..FREE };
        let obs = Observation::running(1, Lane::C, lanes(FREE, FREE, coins));
        assert_eq!(r.evaluate(&obs).action, Action::Right);
        let obs = Observation::running(1, Lane::C, lanes(FREE, FREE, FREE));
        assert_eq!(r.evaluate(&obs).action, Action::Stay);
    }

    #[test]
    fn speed_tracks_band_transitions() {
        let mut r = reflex();
        let before = r.speed();
        // A train moves far → mid → near in 150 ms steps: ~6.7 bands/s, faster than the default.
        for (i, view) in [LaneView { far: Obstacle::TrainBody, ..FREE }, mid(Obstacle::TrainBody), near(Obstacle::TrainBody)]
            .into_iter()
            .enumerate()
        {
            r.update(&Observation::running(i as u64, Lane::C, lanes(view, FREE, FREE)), i as f64 * 150.0);
        }
        assert!(r.speed() > before);
    }
}

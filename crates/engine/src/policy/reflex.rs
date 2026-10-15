//! Reflex layer (SPEC §4.5): per-frame Rust rules. Estimates time to impact from how fast
//! obstacles move between zone bands, builds the safety mask every action must pass, and makes
//! emergency dodges.

use serde::{Deserialize, Serialize};

use super::{Action, ActionSet, possible_actions};
use crate::config::PolicyConfig;
use crate::perception::{GameState, Lane, LaneView, Obstacle, Observation, PowerUp};

/// Distance from the runner to the start of each band, in band lengths.
const BAND_DISTANCE: [f32; 3] = [0.5, 1.5, 2.5];
/// Observations the speed median is taken over, and the range of speeds the game runs at in
/// bands per second (measured from live runs: about 5 early, 7 later).
const SPEED_WINDOW: usize = 9;
const SPEED_RANGE: (f32, f32) = (3.0, 8.0);
/// After a jetpack or pogo stick ends, the runner drops back onto a track perception hasn't
/// tracked for a while: no lane changes for this long, only the emergency rules.
const LANDING_MS: f64 = 1000.0;
/// Score for a lane whose ramp leads over the train blocking the current lane.
const RAMP_BONUS: f32 = 2.0;

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
    /// Run speed in bands per second: the median of recent band-to-band transitions. Classifier
    /// flicker makes some look far too fast (seen live: an average drifted from 3 to 13), so the
    /// median is used and the result is kept in the range the game really runs at.
    speed: f32,
    recent: std::collections::VecDeque<f32>,
    /// Per lane: band of the first hazard and when it was first seen there (ms).
    tracked: [Option<(usize, f64)>; 3],
    was_flying: bool,
    /// Time of the last update and the end of the post-flight landing window (ms).
    now: f64,
    landing_until: f64,
}

impl Reflex {
    pub fn new(cfg: PolicyConfig) -> Self {
        let speed = cfg.initial_speed_bands_per_s;
        Self { cfg, speed, recent: Default::default(), tracked: [None; 3], was_flying: false, now: 0.0, landing_until: f64::NEG_INFINITY }
    }

    /// Within the landing window after a jetpack or pogo stick.
    pub fn landing(&self) -> bool {
        self.now < self.landing_until
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Resets the speed estimate at the start of a new run.
    pub fn reset(&mut self) {
        self.speed = self.cfg.initial_speed_bands_per_s;
        self.recent.clear();
        self.tracked = [None; 3];
        self.was_flying = false;
        self.landing_until = f64::NEG_INFINITY;
    }

    /// Updates the speed estimate from obstacles moving one band closer between frames.
    pub fn update(&mut self, obs: &Observation, t_ms: f64) {
        self.now = t_ms;
        let flying = obs.state == GameState::Running && obs.flying();
        if self.was_flying && !flying && obs.state == GameState::Running {
            self.landing_until = t_ms + LANDING_MS;
        }
        self.was_flying = flying;
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
                        self.recent.push_back(1.0 / dt as f32);
                        if self.recent.len() > SPEED_WINDOW {
                            self.recent.pop_front();
                        }
                        let mut sorted: Vec<f32> = self.recent.iter().copied().collect();
                        sorted.sort_by(|a, b| a.total_cmp(b));
                        self.speed = sorted[sorted.len() / 2].clamp(SPEED_RANGE.0, SPEED_RANGE.1);
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
        // A jetpack or pogo stick keeps the runner above everything on the track.
        if obs.flying() {
            return true;
        }
        let view = &obs.lanes[target.index()];
        let jumping = action == Action::Jump || obs.airborne;
        let rolling = action == Action::Roll || obs.rolling;
        // Super sneakers jump high enough to land on top of a train.
        let over_trains = jumping && obs.active_powerup == Some(PowerUp::SuperSneakers);
        let mut on_ramp = false;
        for (band, o) in view.bands().into_iter().enumerate() {
            let tti = self.tti_ms(band);
            if tti > window_ms {
                break;
            }
            let barrier_later = tti > self.cfg.barrier_ms as f32;
            let ok = match o {
                // From a roof, a lane with no train or the roof's end is a drop to the track.
                Obstacle::Free | Obstacle::Unknown | Obstacle::RoofEnd => true,
                Obstacle::TrainRamp => {
                    on_ramp = true;
                    true
                }
                // Running up a ramp puts you on top of the train; anything else hits it. From a
                // roof, a train body (above roof height) is a wall.
                Obstacle::TrainBody => on_ramp || over_trains,
                // A roof is walkable from a roof; from the ground it is the train under it.
                Obstacle::TrainRoof => obs.on_roof() || on_ramp || over_trains,
                // Only a jump clears a low barrier: rolling at one crashed 4 of 5 times in logged runs.
                Obstacle::LowBarrier => barrier_later || jumping,
                Obstacle::HighBarrier | Obstacle::OverheadBar => barrier_later || rolling,
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
        let mut mask: ActionSet = possible_actions(obs).iter().filter(|a| self.survives(obs, lane, *a, window)).collect();
        // Just landed: no sidestepping into lanes perception has only just started reading.
        if self.landing() {
            let held: ActionSet = mask.iter().filter(|a| !matches!(a, Action::Left | Action::Right)).collect();
            if !held.is_empty() {
                mask = held;
            }
        }
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
                let clear = |o: Obstacle| {
                    matches!(o, Obstacle::Free | Obstacle::TrainRamp | Obstacle::RoofEnd) || (o == Obstacle::TrainRoof && obs.on_roof())
                };
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
        // A train ahead in this lane, within the look-ahead: a ramp beside it is the way over.
        let train_ahead = obs.lanes[lane.index()]
            .bands()
            .iter()
            .position(|o| *o == Obstacle::TrainBody)
            .is_some_and(|b| self.tti_ms(b) <= self.cfg.lookahead_ms as f32);
        let score = |a: Action| -> f32 {
            let Some(t) = a.target_lane(lane) else { return f32::NEG_INFINITY };
            let view: &LaneView = &obs.lanes[t.index()];
            let mut s = 0.0;
            s += if view.coins { 1.0 } else { 0.0 } + if view.powerup { self.cfg.powerup_weight } else { 0.0 };
            s += if t == Lane::C { 0.2 } else { 0.0 };
            if obs.flying() {
                // Nothing on the track matters while flying: follow the coins.
                return s + if a == Action::Stay { 0.5 } else if matches!(a, Action::Jump | Action::Roll) { -1.0 } else { 0.0 };
            }
            let walkable = |o: &Obstacle| *o == Obstacle::Free || (obs.on_roof() && *o == Obstacle::TrainRoof);
            s += view.bands().iter().filter(|o| walkable(o)).count() as f32 * 0.5;
            s -= view.bands().iter().filter(|o| o.is_hazard()).count() as f32 * 0.4;
            s -= view.bands().iter().filter(|o| **o == Obstacle::Unknown).count() as f32 * 0.3;
            if obs.on_roof() && matches!(view.near, Obstacle::Free | Obstacle::RoofEnd) && t != lane {
                s -= self.cfg.roof_drop_penalty;
            }
            if train_ahead && t != lane && (view.near == Obstacle::TrainRamp || view.mid == Obstacle::TrainRamp) {
                s += RAMP_BONUS;
            }
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
        // Never sidestep into a lane that already has a train or barrier right beside the runner
        // (the near band): that's a side collision (a stumble, and the second one ends the run).
        // A hazard further up the lane is only ahead and moving there still buys time. When boxed
        // in between two trains, jumping, rolling or holding the lane is the only non-crash.
        let blocked_side = |a: Action| {
            matches!(a, Action::Left | Action::Right)
                && a.target_lane(lane).is_some_and(|t| obs.lanes[t.index()].near.is_hazard())
        };
        let mut candidates: Vec<Action> = possible_actions(obs).iter().filter(|a| !blocked_side(*a)).collect();
        if candidates.is_empty() {
            candidates = possible_actions(obs).iter().collect();
        }
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
    fn boxed_in_between_trains_never_sidesteps_into_one() {
        let r = reflex();
        let train = LaneView { near: Obstacle::TrainBody, mid: Obstacle::TrainBody, far: Obstacle::TrainBody, ..FREE };
        let obs = Observation::running(1, Lane::C, lanes(train, near(Obstacle::TrainBody), train));
        let pick = r.least_bad(&obs);
        assert!(!matches!(pick, Action::Left | Action::Right), "{pick:?}");
        let out = r.evaluate(&obs);
        assert!(!matches!(out.fallback, Action::Left | Action::Right), "{:?}", out.fallback);
        // With a free neighbour the dodge is still allowed.
        let obs = Observation::running(1, Lane::C, lanes(train, near(Obstacle::TrainBody), FREE));
        assert_eq!(r.evaluate(&obs).action, Action::Right);
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

    // SPEC-roofs-and-powerups §4.4: one test per row of the roof and power-up tables.

    fn on_roof(lanes: [LaneView; 3]) -> Observation {
        Observation { level: crate::perception::Level::Roof, ..Observation::running(1, Lane::C, lanes) }
    }

    fn with_powerup(mut obs: Observation, p: PowerUp) -> Observation {
        obs.active_powerup = Some(p);
        obs
    }

    const ROOF: LaneView = LaneView { near: Obstacle::TrainRoof, mid: Obstacle::TrainRoof, far: Obstacle::TrainRoof, coins: false, powerup: false };

    #[test]
    fn roof_train_body_is_a_wall() {
        let r = reflex();
        let obs = on_roof(lanes(ROOF, LaneView { near: Obstacle::TrainBody, ..ROOF }, ROOF));
        let mask = r.safety_mask(&obs);
        assert!(!mask.contains(Action::Stay) && mask.contains(Action::Left) && mask.contains(Action::Right));
    }

    #[test]
    fn roof_to_roof_step_is_safe() {
        let r = reflex();
        let obs = on_roof(lanes(ROOF, LaneView { mid: Obstacle::TrainBody, ..ROOF }, FREE));
        assert!(r.safety_mask(&obs).contains(Action::Left));
    }

    #[test]
    fn roof_end_is_a_safe_drop() {
        let r = reflex();
        let obs = on_roof(lanes(FREE, LaneView { near: Obstacle::RoofEnd, ..FREE }, FREE));
        assert!(r.safety_mask(&obs).contains(Action::Stay));
    }

    #[test]
    fn dropping_off_the_roofs_is_allowed_but_staying_up_is_preferred() {
        let r = reflex();
        let obs = on_roof(lanes(FREE, ROOF, ROOF));
        let mask = r.safety_mask(&obs);
        assert!(mask.contains(Action::Left), "a drop is safe");
        // Left drops to the track, right steps to the next roof: right wins.
        let wall = LaneView { near: Obstacle::TrainBody, ..ROOF };
        let obs = on_roof(lanes(FREE, wall, ROOF));
        assert_eq!(r.default_action(&obs, r.safety_mask(&obs)), Action::Right);
    }

    #[test]
    fn roof_seen_from_the_ground_is_a_train() {
        let r = reflex();
        let obs = Observation::running(1, Lane::C, lanes(FREE, near(Obstacle::TrainRoof), FREE));
        assert!(!r.safety_mask(&obs).contains(Action::Stay));
    }

    #[test]
    fn ramp_beside_a_blocking_train_is_taken_on_purpose() {
        let r = reflex();
        let ramp = LaneView { near: Obstacle::TrainRamp, mid: Obstacle::TrainBody, ..FREE };
        let obs = Observation::running(1, Lane::C, lanes(FREE, mid(Obstacle::TrainBody), ramp));
        assert_eq!(r.default_action(&obs, r.safety_mask(&obs)), Action::Right);
    }

    #[test]
    fn jetpack_ignores_the_track_and_follows_coins() {
        let r = reflex();
        let wall = LaneView { near: Obstacle::TrainBody, ..FREE };
        let obs = with_powerup(Observation::running(1, Lane::C, lanes(wall, wall, LaneView { coins: true, ..wall })), PowerUp::Jetpack);
        let mask = r.safety_mask(&obs);
        assert!(mask.contains(Action::Stay) && mask.contains(Action::Left) && mask.contains(Action::Right));
        assert!(!r.emergency(&obs));
        assert_eq!(r.default_action(&obs, mask), Action::Right);
        let pogo = with_powerup(Observation::running(1, Lane::C, lanes(wall, wall, wall)), PowerUp::PogoStick);
        assert!(r.safety_mask(&pogo).contains(Action::Stay));
    }

    #[test]
    fn landing_after_a_jetpack_holds_the_lane() {
        let mut r = reflex();
        let free = Observation::running(1, Lane::C, lanes(FREE, FREE, FREE));
        r.update(&with_powerup(free.clone(), PowerUp::Jetpack), 1000.0);
        r.update(&free, 1100.0);
        let mask = r.safety_mask(&free);
        assert!(!mask.contains(Action::Left) && !mask.contains(Action::Right) && mask.contains(Action::Stay));
        // A second later the window is over.
        r.update(&free, 2200.0);
        assert!(r.safety_mask(&free).contains(Action::Left));
        // Even while landing, a lane change stays allowed when it is the only way out.
        let mut r = reflex();
        r.update(&with_powerup(free.clone(), PowerUp::Jetpack), 1000.0);
        let trapped = Observation::running(1, Lane::C, lanes(FREE, near(Obstacle::TrainBody), FREE));
        r.update(&trapped, 1100.0);
        assert!(r.safety_mask(&trapped).contains(Action::Left));
    }

    #[test]
    fn super_sneakers_jump_onto_a_train() {
        let r = reflex();
        let obs = Observation::running(1, Lane::L, lanes(near(Obstacle::TrainBody), near(Obstacle::TrainBody), FREE));
        assert!(!r.safety_mask(&obs).contains(Action::Jump));
        assert!(r.safety_mask(&with_powerup(obs, PowerUp::SuperSneakers)).contains(Action::Jump));
    }

    #[test]
    fn magnet_multiplier_and_hoverboard_change_no_rule() {
        let r = reflex();
        let obs = Observation::running(1, Lane::C, lanes(near(Obstacle::TrainBody), near(Obstacle::LowBarrier), FREE));
        let plain: Vec<Action> = r.safety_mask(&obs).iter().collect();
        for p in [PowerUp::Magnet, PowerUp::Multiplier, PowerUp::Hoverboard] {
            assert_eq!(r.safety_mask(&with_powerup(obs.clone(), p)).iter().collect::<Vec<_>>(), plain, "{p:?}");
        }
    }

    #[test]
    fn power_up_items_outweigh_coins() {
        let r = reflex();
        let obs = Observation::running(1, Lane::C, lanes(LaneView { coins: true, ..FREE }, mid(Obstacle::TrainBody), LaneView { powerup: true, ..FREE }));
        assert_eq!(r.default_action(&obs, r.safety_mask(&obs)), Action::Right);
    }
}

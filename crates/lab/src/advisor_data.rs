//! Training data for OpenJev's decision head (`sidecar/train_head.py`). Each record is a facts
//! premise, the option sentences the advisor would send, and the correct option(s).
//!
//! - train: random situations; the correct move is the reflex policy's choice (its safety
//!   rules are unit-tested), so the head learns safe, coin-seeking moves from correct facts.
//! - test_bench: the hand-written benchmark cases (several answers may be acceptable).
//! - test_real: situations from labelled gameplay frames (`labels/*.jsonl`), with the
//!   correct move computed from the *labelled* scene, not from perception.
//! - test_roof_powerup: the same, from labelled frames on a roof or with a power-up running
//!   (SPEC-roofs-and-powerups §4.5), kept apart so the new cases are scored on their own.
//!
//! Any training situation whose facts string appears in a test set is dropped.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use ssbot_engine::config::{PolicyConfig, Wording};
use ssbot_engine::facts::facts;
use ssbot_engine::perception::{GameState, Lane, LaneView, Level, Obstacle, Observation, PowerUp};
use ssbot_engine::policy::Action;
use ssbot_engine::policy::advisor::options_with;
use ssbot_engine::policy::reflex::Reflex;

use crate::advisor_bench::cases;
use crate::label::FrameLabel;

#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub premise: String,
    pub options: BTreeMap<String, String>,
    pub best: Vec<String>,
    pub source: String,
}

/// Small deterministic PRNG (xorshift), so datasets are reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<T: Copy>(&mut self, weighted: &[(T, f64)]) -> T {
        let total: f64 = weighted.iter().map(|(_, w)| w).sum();
        let mut r = self.next() * total;
        for (v, w) in weighted {
            if r < *w {
                return *v;
            }
            r -= w;
        }
        weighted[weighted.len() - 1].0
    }
}

fn random_view(rng: &mut Rng) -> LaneView {
    use Obstacle::*;
    let dist = [(Free, 0.48), (TrainBody, 0.22), (TrainRamp, 0.06), (LowBarrier, 0.08), (HighBarrier, 0.07), (OverheadBar, 0.07), (Unknown, 0.02)];
    let near = rng.pick(&dist);
    // Trains are long: a train close ahead usually continues into the mid band.
    let mid = if (near == TrainBody && rng.next() < 0.7) || near == TrainRamp { TrainBody } else { rng.pick(&dist) };
    let far = rng.pick(&dist);
    LaneView { near, mid, far, coins: rng.next() < 0.3, powerup: rng.next() < 0.05 }
}

/// A lane seen from a roof: mostly roofs, roofs that end (and the drop after), the odd wall.
fn random_roof_view(rng: &mut Rng) -> LaneView {
    use Obstacle::*;
    let dist = [(TrainRoof, 0.52), (RoofEnd, 0.12), (Free, 0.2), (TrainBody, 0.06), (OverheadBar, 0.05), (Unknown, 0.05)];
    let mut bands = [rng.pick(&dist), rng.pick(&dist), rng.pick(&dist)];
    // Past the end of a roof the lane is down on the track.
    if let Some(end) = bands.iter().position(|o| *o == RoofEnd) {
        for b in &mut bands[end + 1..] {
            *b = Free;
        }
    }
    LaneView { near: bands[0], mid: bands[1], far: bands[2], coins: rng.next() < 0.3, powerup: rng.next() < 0.05 }
}

const POWERUPS: [PowerUp; 6] = [PowerUp::Jetpack, PowerUp::SuperSneakers, PowerUp::PogoStick, PowerUp::Magnet, PowerUp::Multiplier, PowerUp::Hoverboard];

fn record(obs: &Observation, best: Vec<Action>, wording: Wording, source: &str) -> Option<Record> {
    let options = options_with(obs, wording);
    let best: Vec<String> = best.into_iter().map(|a| a.key().to_string()).filter(|k| options.contains_key(k)).collect();
    if best.is_empty() || options.len() < 2 {
        return None;
    }
    Some(Record { premise: facts(obs)?, options, best, source: source.into() })
}

pub fn obs_from_label(id: u64, l: &FrameLabel) -> Option<Observation> {
    if l.game_state != GameState::Running {
        return None;
    }
    let lane = match l.player_lane.as_str() {
        "L" => Lane::L,
        "C" => Lane::C,
        "R" => Lane::R,
        _ => return None,
    };
    let lanes = [0, 1, 2].map(|i| {
        let z = |b| l.zone(i, b).map(|z| z.obstacle).unwrap_or(Obstacle::Unknown);
        let coins = (0..3).any(|b| l.zone(i, b).is_some_and(|z| z.coins));
        let powerup = (0..3).any(|b| l.zone(i, b).is_some_and(|z| z.powerup));
        LaneView { near: z(0), mid: z(1), far: z(2), coins, powerup }
    });
    let mut obs = Observation::running(id, lane, lanes);
    obs.airborne = l.player_action == "jumping" || l.level() == Some(Level::Air);
    obs.rolling = l.player_action == "rolling";
    obs.level = l.level().unwrap_or(if obs.airborne { Level::Air } else { Level::Ground });
    obs.active_powerup = l.powerup().flatten();
    Some(obs)
}

pub struct Datasets {
    pub train: Vec<Record>,
    pub test_bench: Vec<Record>,
    pub test_real: Vec<Record>,
    /// Random situations with an off-screen lane (runner in an outer lane), drawn with a
    /// different seed and kept out of training.
    pub test_offscreen: Vec<Record>,
    pub test_roof_powerup: Vec<Record>,
}

pub fn build(n_train: usize, seed: u64, wording: Wording, labelled: &[(u64, FrameLabel)]) -> Datasets {
    let reflex = Reflex::new(PolicyConfig::default());
    let test_bench: Vec<Record> = cases().iter().filter_map(|c| record(&c.obs, c.best.clone(), wording, &c.name)).collect();
    let mut seen_real = HashSet::new();
    let (mut test_real, mut test_roof_powerup) = (Vec::new(), Vec::new());
    for obs in labelled.iter().filter_map(|(id, l)| obs_from_label(*id, l)) {
        let action = reflex.evaluate(&obs).action;
        let Some(r) = record(&obs, vec![action], wording, &format!("frame {}", obs.frame_id)) else { continue };
        if !seen_real.insert(r.premise.clone()) {
            continue;
        }
        if obs.on_roof() || obs.active_powerup.is_some() { test_roof_powerup.push(r) } else { test_real.push(r) }
    }
    let held: HashSet<String> = test_bench.iter().chain(&test_real).chain(&test_roof_powerup).map(|r| r.premise.clone()).collect();

    let offscreen_test = random_records(400, seed.wrapping_mul(7919).wrapping_add(99), wording, &held)
        .into_iter()
        .filter(|r| r.premise.contains("not visible"))
        .collect::<Vec<_>>();
    let mut held = held;
    held.extend(offscreen_test.iter().map(|r| r.premise.clone()));
    let train = random_records(n_train, seed, wording, &held);
    Datasets { train, test_bench, test_real, test_offscreen: offscreen_test, test_roof_powerup }
}

fn random_records(n_train: usize, seed: u64, wording: Wording, held: &HashSet<String>) -> Vec<Record> {
    let reflex = Reflex::new(PolicyConfig::default());
    let mut rng = Rng(seed.max(1));
    let mut train = Vec::new();
    let mut seen = HashSet::new();
    let mut tries = 0;
    while train.len() < n_train && tries < n_train * 50 {
        tries += 1;
        let lane = Lane::ALL[(rng.next() * 3.0) as usize % 3];
        let level = rng.pick(&[(Level::Ground, 0.70), (Level::Roof, 0.25), (Level::Air, 0.05)]);
        let mut views = if level == Level::Roof {
            let mut v = [random_roof_view(&mut rng), random_roof_view(&mut rng), random_roof_view(&mut rng)];
            // The runner stands on a roof: its own lane starts with roof most of the time.
            if rng.next() < 0.8 {
                v[lane.index()].near = Obstacle::TrainRoof;
            }
            v
        } else {
            [random_view(&mut rng), random_view(&mut rng), random_view(&mut rng)]
        };
        // Live, the lane two away from an outer-lane runner is off-screen (all Unknown).
        if lane != Lane::C && rng.next() < 0.8 {
            let far = if lane == Lane::L { 2 } else { 0 };
            views[far] = LaneView { near: Obstacle::Unknown, mid: Obstacle::Unknown, far: Obstacle::Unknown, coins: false, powerup: false };
        }
        let mut obs = Observation::running(0, lane, views);
        obs.level = level;
        obs.airborne = level == Level::Air || (level == Level::Ground && rng.next() < 0.1);
        obs.rolling = !obs.airborne && rng.next() < 0.1;
        if rng.next() < 0.2 {
            obs.active_powerup = Some(POWERUPS[(rng.next() * POWERUPS.len() as f64) as usize % POWERUPS.len()]);
        }
        let out = reflex.evaluate(&obs);
        if out.action == Action::Hoverboard {
            continue; // no safe ordinary move: nothing for the advisor to learn
        }
        let Some(r) = record(&obs, vec![out.action], wording, "synthetic") else { continue };
        if held.contains(&r.premise) || !seen.insert(r.premise.clone()) {
            continue;
        }
        train.push(r);
    }
    train
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datasets_are_disjoint_and_teacher_moves_are_safe() {
        let d = build(300, 7, Wording::default(), &[]);
        assert!(d.test_offscreen.len() > 20);
        let off: HashSet<_> = d.test_offscreen.iter().map(|r| &r.premise).collect();
        assert!(d.train.iter().all(|r| !off.contains(&r.premise)));
        assert_eq!(d.train.len(), 300);
        let held: HashSet<_> = d.test_bench.iter().map(|r| &r.premise).collect();
        assert!(d.train.iter().all(|r| !held.contains(&r.premise)));
        assert!(d.train.iter().all(|r| r.best.len() == 1 && r.options.contains_key(&r.best[0])));
        // Not everything is "stay": the head must learn to move.
        let moves = d.train.iter().filter(|r| r.best[0] != "stay").count();
        assert!(moves > 50, "{moves}");
        let off_screen = d.train.iter().filter(|r| r.premise.contains("not visible")).count();
        assert!(off_screen > 50, "{off_screen} situations with an off-screen lane");
        // Roofs and power-ups are in the training data (SPEC-roofs-and-powerups §4.5).
        let roof = d.train.iter().filter(|r| r.premise.contains("on a train roof")).count();
        let powered = d.train.iter().filter(|r| r.premise.contains("Power-up:")).count();
        assert!(roof > 40 && powered > 30, "{roof} roof, {powered} power-up situations");
    }
}

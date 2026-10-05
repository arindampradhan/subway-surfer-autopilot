//! Two-layer policy (SPEC §4.5): Rust reflexes that always run, an OpenJev advisor that
//! runs alongside, and an arbiter that only lets advice through the reflex safety mask.

pub mod advisor;
pub mod arbiter;
pub mod bench;
pub mod dataset;
pub mod reflex;

use serde::{Deserialize, Serialize};

use crate::perception::{Lane, Observation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Stay,
    Left,
    Right,
    Jump,
    Roll,
    Hoverboard,
}

impl Action {
    pub const ALL: [Action; 6] = [Self::Stay, Self::Left, Self::Right, Self::Jump, Self::Roll, Self::Hoverboard];

    pub fn key(self) -> &'static str {
        match self {
            Self::Stay => "stay",
            Self::Left => "left",
            Self::Right => "right",
            Self::Jump => "jump",
            Self::Roll => "roll",
            Self::Hoverboard => "hoverboard",
        }
    }

    pub fn from_key(key: &str) -> Option<Action> {
        Self::ALL.into_iter().find(|a| a.key() == key)
    }

    /// The keyboard key this action presses, if any (SPEC §2 controls).
    pub fn dom_key(self) -> Option<&'static str> {
        match self {
            Self::Stay => None,
            Self::Left => Some("ArrowLeft"),
            Self::Right => Some("ArrowRight"),
            Self::Jump => Some("ArrowUp"),
            Self::Roll => Some("ArrowDown"),
            Self::Hoverboard => Some("Space"),
        }
    }

    /// Lane after the action, or `None` if it would leave the track.
    pub fn target_lane(self, lane: Lane) -> Option<Lane> {
        match self {
            Self::Left => lane.index().checked_sub(1).and_then(Lane::from_index),
            Self::Right => Lane::from_index(lane.index() + 1),
            _ => Some(lane),
        }
    }
}

/// A small set of actions (bit per `Action`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSet(u8);

impl ActionSet {
    pub fn insert(&mut self, a: Action) {
        self.0 |= 1 << a as u8;
    }
    pub fn contains(self, a: Action) -> bool {
        self.0 & (1 << a as u8) != 0
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn iter(self) -> impl Iterator<Item = Action> {
        Action::ALL.into_iter().filter(move |a| self.contains(*a))
    }
}

impl FromIterator<Action> for ActionSet {
    fn from_iter<I: IntoIterator<Item = Action>>(iter: I) -> Self {
        let mut s = ActionSet::default();
        iter.into_iter().for_each(|a| s.insert(a));
        s
    }
}

/// Actions that are possible at all: no `Left` from the left lane, no second jump mid-air.
/// Hoverboard is kept out of normal play; the reflex layer uses it only as a last resort.
pub fn possible_actions(obs: &Observation) -> ActionSet {
    let Some(lane) = obs.player_lane else { return [Action::Stay].into_iter().collect() };
    Action::ALL
        .into_iter()
        .filter(|a| match a {
            Action::Left | Action::Right => a.target_lane(lane).is_some(),
            Action::Jump => !obs.airborne,
            Action::Hoverboard => false,
            _ => true,
        })
        .collect()
}

/// The runner's lane, tracked from the bot's own lane changes. The camera follows the runner
/// sideways, so its screen position doesn't tell the lane (SPEC §10.2); runs start in the
/// centre lane and every Left/Right the bot sends moves it one lane.
#[derive(Debug, Clone, Copy)]
pub struct LaneTracker {
    lane: Lane,
}

impl Default for LaneTracker {
    fn default() -> Self {
        LaneTracker { lane: Lane::C }
    }
}

impl LaneTracker {
    pub fn lane(&self) -> Lane {
        self.lane
    }
    pub fn reset(&mut self) {
        self.lane = Lane::C;
    }
    pub fn apply(&mut self, action: Action) {
        if let Some(l) = action.target_lane(self.lane) {
            self.lane = l;
        }
    }
}

#[cfg(test)]
mod lane_tests {
    use super::*;

    #[test]
    fn tracks_and_clamps() {
        let mut t = LaneTracker::default();
        t.apply(Action::Left);
        assert_eq!(t.lane(), Lane::L);
        t.apply(Action::Left);
        assert_eq!(t.lane(), Lane::L);
        t.apply(Action::Right);
        t.apply(Action::Right);
        t.apply(Action::Jump);
        assert_eq!(t.lane(), Lane::R);
        t.reset();
        assert_eq!(t.lane(), Lane::C);
    }
}

//! Observation → short plain-fact text for the advisor (SPEC §4.4). Closed vocabulary: the
//! same situation always gives the same string, which is what makes the advisor cache work.
//! Facts never say which move is best.

use crate::perception::{Lane, LaneView, Obstacle, Observation};

pub fn obstacle_phrase(o: Obstacle) -> &'static str {
    match o {
        Obstacle::Free => "nothing",
        Obstacle::TrainBody => "train",
        Obstacle::TrainRamp => "train ramp",
        Obstacle::LowBarrier => "low barrier",
        Obstacle::HighBarrier => "high barrier",
        Obstacle::OverheadBar => "overhead bar",
        Obstacle::Unknown => "unclear object",
    }
}

pub const BAND_PHRASES: [&str; 3] = ["close ahead", "at mid distance", "far ahead"];

/// The first non-free band in a lane, as `(band, obstacle)`.
pub fn first_obstacle(view: &LaneView) -> Option<(usize, Obstacle)> {
    view.bands().into_iter().enumerate().find(|(_, o)| *o != Obstacle::Free)
}

pub fn lane_phrase(view: &LaneView) -> String {
    // A lane two away from the runner is off-screen (the camera follows the runner).
    if view.bands().iter().all(|o| *o == Obstacle::Unknown) && !view.coins && !view.powerup {
        return "not visible".into();
    }
    let mut s = match first_obstacle(view) {
        None => "free".to_string(),
        Some((band, o)) => format!("{} {}", obstacle_phrase(o), BAND_PHRASES[band]),
    };
    if view.coins {
        s.push_str(", coins ahead");
    }
    if view.powerup {
        s.push_str(", power-up");
    }
    s
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// `None` when the player's lane is unknown: the advisor is only asked about clear situations.
pub fn facts(obs: &Observation) -> Option<String> {
    let lane = obs.player_lane?;
    let posture = if obs.airborne {
        "in the air"
    } else if obs.rolling {
        "rolling"
    } else {
        "on the ground"
    };
    let mut s = format!("Subway run. You are in the {lane} lane, {posture}.\n");
    let parts: Vec<String> =
        Lane::ALL.iter().map(|l| format!("{} lane: {}.", capitalise(l.name()), lane_phrase(&obs.lanes[l.index()]))).collect();
    s.push_str(&parts.join(" "));
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::LaneView;

    #[test]
    fn spec_example() {
        let lanes = [
            LaneView { near: Obstacle::TrainBody, ..LaneView::FREE },
            LaneView { mid: Obstacle::LowBarrier, ..LaneView::FREE },
            LaneView { coins: true, ..LaneView::FREE },
        ];
        let obs = Observation::running(1, Lane::C, lanes);
        assert_eq!(
            facts(&obs).unwrap(),
            "Subway run. You are in the center lane, on the ground.\n\
             Left lane: train close ahead. Center lane: low barrier at mid distance. Right lane: free, coins ahead."
        );
    }

    #[test]
    fn off_screen_lane_is_not_visible() {
        let unknown = LaneView { near: Obstacle::Unknown, mid: Obstacle::Unknown, far: Obstacle::Unknown, coins: false, powerup: false };
        let obs = Observation::running(1, Lane::R, [unknown, LaneView::FREE, LaneView::FREE]);
        assert!(facts(&obs).unwrap().contains("Left lane: not visible."));
    }

    #[test]
    fn deterministic_and_short() {
        // The longest phrases in every lane. Measured with the OpenJev 0.8B tokenizer: this
        // string is 247 characters and 60 tokens, the SPEC §4.4 limit.
        let lanes = [
            LaneView { near: Obstacle::Unknown, coins: true, powerup: true, ..LaneView::FREE },
            LaneView { mid: Obstacle::OverheadBar, coins: true, powerup: true, ..LaneView::FREE },
            LaneView { mid: Obstacle::HighBarrier, coins: true, powerup: true, ..LaneView::FREE },
        ];
        let obs = Observation::running(7, Lane::C, lanes);
        let a = facts(&obs).unwrap();
        assert_eq!(a, facts(&Observation { frame_id: 99, ..obs.clone() }).unwrap());
        assert!(a.len() <= 247, "{} chars: {a}", a.len());
        assert!(facts(&Observation { player_lane: None, ..obs }).is_none());
    }
}

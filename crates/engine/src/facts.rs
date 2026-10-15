//! Observation → short plain-fact text for the advisor (SPEC §4.4). Closed vocabulary: the
//! same situation always gives the same string, which is what makes the advisor cache work.
//! Facts never say which move is best.

use crate::perception::{Lane, LaneView, Level, Obstacle, Observation};

pub fn obstacle_phrase(o: Obstacle) -> &'static str {
    match o {
        Obstacle::Free => "nothing",
        Obstacle::TrainBody => "train",
        Obstacle::TrainRamp => "train ramp",
        Obstacle::LowBarrier => "low barrier",
        Obstacle::HighBarrier => "high barrier",
        Obstacle::OverheadBar => "overhead bar",
        Obstacle::Unknown => "unclear object",
        Obstacle::TrainRoof => "train roof",
        Obstacle::RoofEnd => "end of roof",
    }
}

/// `obstacle_phrase`, except that on a roof a lane with no train has no roof: a drop to the
/// track, not "nothing". (Shorter than "drop to the track": the worst roof facts string with a
/// power-up then stays within 60 OpenJev tokens.)
pub fn obstacle_phrase_at(o: Obstacle, on_roof: bool) -> &'static str {
    if on_roof && o == Obstacle::Free { "no roof" } else { obstacle_phrase(o) }
}

pub const BAND_PHRASES: [&str; 3] = ["close ahead", "at mid distance", "far ahead"];

/// The first band in a lane the runner has to deal with, as `(band, obstacle)`: not free on the
/// ground; on a roof, not the roof itself.
pub fn first_obstacle(view: &LaneView, on_roof: bool) -> Option<(usize, Obstacle)> {
    let walkable = |o: Obstacle| if on_roof { o == Obstacle::TrainRoof } else { o == Obstacle::Free };
    view.bands().into_iter().enumerate().find(|(_, o)| !walkable(*o))
}

pub fn lane_phrase(view: &LaneView, on_roof: bool) -> String {
    // A lane two away from the runner is off-screen (the camera follows the runner).
    if view.bands().iter().all(|o| *o == Obstacle::Unknown) && !view.coins && !view.powerup {
        return "not visible".into();
    }
    let mut s = match first_obstacle(view, on_roof) {
        None if on_roof => "train roof".to_string(),
        None => "free".to_string(),
        Some((band, o)) => format!("{} {}", obstacle_phrase_at(o, on_roof), BAND_PHRASES[band]),
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
    let posture = if obs.airborne || obs.level == Level::Air {
        "in the air"
    } else if obs.rolling {
        "rolling"
    } else if obs.on_roof() {
        "on a train roof"
    } else {
        "on the ground"
    };
    let mut s = format!("Subway run. You are in the {lane} lane, {posture}.\n");
    let parts: Vec<String> = Lane::ALL
        .iter()
        .map(|l| format!("{} lane: {}.", capitalise(l.name()), lane_phrase(&obs.lanes[l.index()], obs.on_roof())))
        .collect();
    s.push_str(&parts.join(" "));
    if let Some(p) = obs.active_powerup {
        s.push_str(&format!(" Power-up: {}.", p.name()));
    }
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
    fn roof_and_powerup_facts() {
        let lanes = [
            LaneView { near: Obstacle::TrainRoof, mid: Obstacle::RoofEnd, ..LaneView::FREE },
            LaneView { near: Obstacle::TrainRoof, mid: Obstacle::TrainRoof, far: Obstacle::TrainRoof, ..LaneView::FREE },
            LaneView::FREE,
        ];
        let obs = Observation { level: Level::Roof, active_powerup: Some(crate::perception::PowerUp::Jetpack), ..Observation::running(1, Lane::C, lanes) };
        assert_eq!(
            facts(&obs).unwrap(),
            "Subway run. You are in the center lane, on a train roof.\n\
             Left lane: end of roof at mid distance. Center lane: train roof. Right lane: no roof close ahead. \
             Power-up: jetpack."
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

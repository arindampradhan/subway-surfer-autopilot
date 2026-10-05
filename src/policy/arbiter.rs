//! Arbiter (SPEC §4.5): handles the between-runs flow, then picks reflex, advisor or default.
//! Restart flow (SPEC §10.1): Space on the new-high-score banner → click PLAY on the score
//! screen → wait out ad breaks. It never clicks anything but PLAY-style markers and the canvas.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::Action;
use super::advisor::AdvisorPick;
use super::reflex::ReflexOut;
use crate::perception::{GameState, Observation};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Command {
    Wait,
    Act(Action),
    PressSpace,
    /// Click at a normalised canvas position.
    Click([f32; 2]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Flow,
    Cooldown,
    Reflex,
    Advisor,
    Default,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Chosen {
    pub command: Command,
    pub source: Source,
}

pub struct Arbiter {
    cooldown: Duration,
    flow_interval: Duration,
    blocked_until: Option<Instant>,
    last_flow: Option<Instant>,
    canvas_center: [f32; 2],
    last_hoverboard: Option<Instant>,
    hoverboard_cooldown: Duration,
    hoverboard_enabled: bool,
    /// Last lane change and when, to stop immediate reversals.
    last_lane_change: Option<(Action, Instant)>,
}

impl Arbiter {
    pub fn new(cooldown_ms: u64, canvas_center: [f32; 2]) -> Self {
        Self {
            cooldown: Duration::from_millis(cooldown_ms),
            flow_interval: Duration::from_millis(1500),
            blocked_until: None,
            last_flow: None,
            canvas_center,
            last_hoverboard: None,
            // A hoverboard lasts several seconds and the supply is limited: one at a time.
            hoverboard_cooldown: Duration::from_secs(15),
            hoverboard_enabled: true,
            last_lane_change: None,
        }
    }

    /// Whether the hoverboard key may be pressed at all (see `PolicyConfig::use_hoverboard`).
    pub fn with_hoverboard(mut self, enabled: bool) -> Self {
        self.hoverboard_enabled = enabled;
        self
    }

    /// Non-running screens: what to press, rate-limited so a slow screen isn't spammed.
    fn flow(&mut self, state: GameState, marker_click: Option<[f32; 2]>, now: Instant) -> Chosen {
        let command = match state {
            GameState::Menu => Command::Click(marker_click.unwrap_or(self.canvas_center)),
            GameState::NewHighScore => Command::PressSpace,
            // Only a calibrated PLAY marker may be clicked here; never "Free Double Up".
            GameState::ScoreScreen | GameState::RevivePrompt | GameState::Paused => {
                marker_click.map(Command::Click).unwrap_or(Command::Wait)
            }
            // Ads, loading and crash animations: wait, never click.
            _ => Command::Wait,
        };
        let ready = self.last_flow.is_none_or(|t| now.duration_since(t) >= self.flow_interval);
        if command == Command::Wait || !ready {
            return Chosen { command: Command::Wait, source: Source::Flow };
        }
        self.last_flow = Some(now);
        Chosen { command, source: Source::Flow }
    }

    pub fn decide(
        &mut self,
        obs: &Observation,
        marker_click: Option<[f32; 2]>,
        reflex: &ReflexOut,
        advice: Option<&AdvisorPick>,
        now: Instant,
    ) -> Chosen {
        if obs.state != GameState::Running {
            return self.flow(obs.state, marker_click, now);
        }
        if self.blocked_until.is_some_and(|t| now < t) {
            return Chosen { command: Command::Wait, source: Source::Cooldown };
        }
        let (mut action, source) = if reflex.emergency {
            (reflex.action, Source::Reflex)
        } else if let Some(pick) = advice.filter(|p| reflex.mask.contains(&p.action)) {
            (pick.action, Source::Advisor)
        } else {
            (reflex.action, Source::Default)
        };
        // Whatever chose it: no hoverboard when disabled or used recently. With none in stock,
        // Space opens a shop dialog that pauses the game (seen live twice).
        let recent = self.last_hoverboard.is_some_and(|t| now.duration_since(t) < self.hoverboard_cooldown);
        if action == Action::Hoverboard && (recent || !self.hoverboard_enabled) {
            action = reflex.fallback;
        }
        // Undoing a lane change right after making it is dithering, unless staying would crash.
        let reverses = |a: Action, b: Action| matches!((a, b), (Action::Left, Action::Right) | (Action::Right, Action::Left));
        if !reflex.emergency
            && let Some((last, at)) = self.last_lane_change
            && reverses(last, action)
            && now.duration_since(at) < Duration::from_millis(600)
        {
            action = Action::Stay;
        }
        if matches!(action, Action::Left | Action::Right) {
            self.last_lane_change = Some((action, now));
        }
        if action != Action::Stay {
            self.blocked_until = Some(now + self.cooldown);
        }
        if action == Action::Hoverboard {
            self.last_hoverboard = Some(now);
        }
        Chosen { command: if action == Action::Stay { Command::Wait } else { Command::Act(action) }, source }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::{Lane, LaneView};

    fn reflex(mask: &[Action], emergency: bool, action: Action) -> ReflexOut {
        ReflexOut { mask: mask.to_vec(), emergency, action, tti_ms: None, speed: 3.0, fallback: Action::Jump }
    }

    #[test]
    fn hoverboard_at_most_once_per_cooldown() {
        let t0 = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        let r = reflex(&[Action::Hoverboard], true, Action::Hoverboard);
        let picks: Vec<Command> =
            (0..20).map(|i| a.decide(&running(), None, &r, None, t0 + Duration::from_millis(200 * i)).command).collect();
        let boards = picks.iter().filter(|c| **c == Command::Act(Action::Hoverboard)).count();
        assert_eq!(boards, 1, "{picks:?}");
        assert!(picks.contains(&Command::Act(Action::Jump)), "uses the least-bad move instead");
        let later = a.decide(&running(), None, &r, None, t0 + Duration::from_secs(16)).command;
        assert_eq!(later, Command::Act(Action::Hoverboard));
    }

    #[test]
    fn no_quick_reversal_unless_emergency() {
        let t0 = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        let right = reflex(&[Action::Stay, Action::Left, Action::Right], false, Action::Right);
        let left = reflex(&[Action::Stay, Action::Left, Action::Right], false, Action::Left);
        assert_eq!(a.decide(&running(), None, &right, None, t0).command, Command::Act(Action::Right));
        assert_eq!(a.decide(&running(), None, &left, None, t0 + Duration::from_millis(300)).command, Command::Wait);
        assert_eq!(a.decide(&running(), None, &left, None, t0 + Duration::from_millis(700)).command, Command::Act(Action::Left));
        // An emergency may reverse straight away.
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        a.decide(&running(), None, &right, None, t0);
        let urgent = reflex(&[Action::Left], true, Action::Left);
        assert_eq!(a.decide(&running(), None, &urgent, None, t0 + Duration::from_millis(300)).command, Command::Act(Action::Left));
    }

    #[test]
    fn hoverboard_off_also_blocks_the_default_path() {
        let t0 = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]).with_hoverboard(false);
        // Not an emergency: the reflex default picked the hoverboard (mask had nothing else).
        let r = reflex(&[Action::Hoverboard], false, Action::Hoverboard);
        let c = a.decide(&running(), None, &r, None, t0).command;
        assert_eq!(c, Command::Act(Action::Jump));
    }

    #[test]
    fn hoverboard_off_never_presses_space() {
        let t0 = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]).with_hoverboard(false);
        let r = reflex(&[Action::Hoverboard], true, Action::Hoverboard);
        for i in 0..50 {
            let c = a.decide(&running(), None, &r, None, t0 + Duration::from_millis(200 * i)).command;
            assert_ne!(c, Command::Act(Action::Hoverboard));
        }
    }

    fn running() -> Observation {
        Observation::running(1, Lane::C, [LaneView::FREE; 3])
    }

    #[test]
    fn emergency_beats_advice() {
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        let pick = AdvisorPick { action: Action::Left, cached: true };
        let c = a.decide(&running(), None, &reflex(&[Action::Left, Action::Right], true, Action::Right), Some(&pick), Instant::now());
        assert_eq!(c, Chosen { command: Command::Act(Action::Right), source: Source::Reflex });
    }

    #[test]
    fn advice_must_pass_mask() {
        let now = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        let unsafe_pick = AdvisorPick { action: Action::Left, cached: false };
        let c = a.decide(&running(), None, &reflex(&[Action::Stay, Action::Right], false, Action::Stay), Some(&unsafe_pick), now);
        assert_eq!(c.source, Source::Default);
        let ok_pick = AdvisorPick { action: Action::Right, cached: false };
        let c = a.decide(&running(), None, &reflex(&[Action::Stay, Action::Right], false, Action::Stay), Some(&ok_pick), now);
        assert_eq!(c, Chosen { command: Command::Act(Action::Right), source: Source::Advisor });
        // Cooldown after a move.
        let c = a.decide(&running(), None, &reflex(&[Action::Left], true, Action::Left), None, now + Duration::from_millis(50));
        assert_eq!(c.source, Source::Cooldown);
        let c = a.decide(&running(), None, &reflex(&[Action::Left], true, Action::Left), None, now + Duration::from_millis(200));
        assert_eq!(c.command, Command::Act(Action::Left));
    }

    #[test]
    fn restart_flow_never_clicks_unknown_buttons() {
        let now = Instant::now();
        let mut a = Arbiter::new(180, [0.5, 0.5]);
        let r = reflex(&[Action::Stay], false, Action::Stay);
        let mut obs = running();
        obs.state = GameState::NewHighScore;
        assert_eq!(a.decide(&obs, None, &r, None, now).command, Command::PressSpace);
        obs.state = GameState::ScoreScreen;
        // Rate-limited, then no marker → wait instead of guessing where PLAY is.
        assert_eq!(a.decide(&obs, None, &r, None, now + Duration::from_secs(2)).command, Command::Wait);
        assert_eq!(a.decide(&obs, Some([0.8, 0.85]), &r, None, now + Duration::from_secs(4)).command, Command::Click([0.8, 0.85]));
        for s in [GameState::AdBreak, GameState::Ad, GameState::Loading, GameState::Crashed] {
            obs.state = s;
            assert_eq!(a.decide(&obs, Some([0.1, 0.1]), &r, None, now + Duration::from_secs(10)).command, Command::Wait);
        }
    }
}

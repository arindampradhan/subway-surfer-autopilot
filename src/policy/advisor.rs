//! Advisor layer (SPEC §4.5): asks OpenJev to pick between moves described by their computed
//! effects, asynchronously, and caches answers by facts string.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::error::TryRecvError;

use super::{Action, possible_actions};
use crate::facts::{BAND_PHRASES, first_obstacle, obstacle_phrase};
use crate::perception::{LaneView, Obstacle, Observation};
use crate::sidecar::{Reply, Request, SidecarLink, WarmItem};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub action: Action,
    pub probs: BTreeMap<String, f32>,
    pub ms: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvisorPick {
    pub action: Action,
    pub cached: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AdvisorStats {
    pub requests: u64,
    pub replies: u64,
    /// Replies that arrived within `advisor_max_age_ms` of their request.
    pub fresh: u64,
    pub errors: u64,
    pub lookups: u64,
    pub cache_hits: u64,
    pub model_ms_total: f64,
}

impl AdvisorStats {
    pub fn freshness_rate(&self) -> f64 {
        if self.replies == 0 { 0.0 } else { self.fresh as f64 / self.replies as f64 }
    }
    pub fn cache_hit_rate(&self) -> f64 {
        if self.lookups == 0 { 0.0 } else { self.cache_hits as f64 / self.lookups as f64 }
    }
}

struct InFlight {
    id: u64,
    facts: String,
    sent: Instant,
}

pub struct Advisor {
    max_age: Duration,
    wording: Wording,
    link: Option<SidecarLink>,
    cache: HashMap<String, Decision>,
    in_flight: Option<InFlight>,
    next_id: u64,
    pub stats: AdvisorStats,
}

fn lane_description(view: &LaneView) -> String {
    let mut s = match first_obstacle(view) {
        None => "a free lane".to_string(),
        Some((band, o)) => format!("a lane with {} {}", article(obstacle_phrase(o)), BAND_PHRASES[band]),
    };
    if view.coins {
        s.push_str(if first_obstacle(view).is_none() { " with coins ahead" } else { " and coins ahead" });
    }
    s
}

fn article(noun: &str) -> String {
    if noun.starts_with(|c: char| "aeiou".contains(c)) { format!("an {noun}") } else { format!("a {noun}") }
}

/// How effects are phrased: where the action takes you, or what it then runs into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum EffectStyle {
    /// "moves into a lane with a train close ahead", "jumps toward the low barrier ..."
    Position,
    /// Adds the computed consequence: "jumps over the low barrier ...", "keeps you running
    /// toward the train ...". Still facts from geometry, like Appendix A's "blocks X".
    Consequence,
}

/// Option wording (SPEC §4.5, §9): an effect style and a sentence template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Wording {
    pub style: EffectStyle,
    /// 0: "<Action> is the best move because it <effect>." (Appendix A's best)
    /// 1: "The best move is <action>, because it <effect>."
    pub template: usize,
}

/// Default from `ssbot advisor-bench` on OpenJev 0.8B (2026-10-05): Consequence/1 scored
/// 14/27 raw vs 8–10/27 for the position wordings.
impl Default for Wording {
    fn default() -> Self {
        Wording { style: EffectStyle::Consequence, template: 1 }
    }
}

impl Wording {
    pub const ALL: [Wording; 4] = [
        Wording { style: EffectStyle::Position, template: 0 },
        Wording { style: EffectStyle::Position, template: 1 },
        Wording { style: EffectStyle::Consequence, template: 0 },
        Wording { style: EffectStyle::Consequence, template: 1 },
    ];

    pub fn sentence(&self, action: Action, effect: &str) -> String {
        match self.template {
            1 => format!("The best move is {}, because it {effect}.", action.key()),
            _ => format!("{} is the best move because it {effect}.", capitalise(action.key())),
        }
    }
}

/// What an action does, as a computed fact (SPEC §4.5 option wording). Never says which is best.
pub fn effect(obs: &Observation, action: Action) -> String {
    effect_with(obs, action, EffectStyle::Position)
}

pub fn effect_with(obs: &Observation, action: Action, style: EffectStyle) -> String {
    let Some(lane) = obs.player_lane else { return "has no known effect".into() };
    let here = &obs.lanes[lane.index()];
    let next_hazard = here.bands().into_iter().enumerate().find(|(_, o)| *o != Obstacle::Free);
    let at = |band: usize, o: Obstacle| format!("the {} {}", obstacle_phrase(o), BAND_PHRASES[band]);
    match (style, action) {
        (_, Action::Hoverboard) => "uses a hoverboard".into(),
        (_, Action::Left | Action::Right) => match action.target_lane(lane) {
            Some(t) => format!("moves into {}", lane_description(&obs.lanes[t.index()])),
            None => "runs into the wall".into(),
        },
        (EffectStyle::Position, Action::Stay) => format!("keeps you in {}", lane_description(here)),
        (EffectStyle::Position, Action::Jump) => match next_hazard {
            Some((band, o)) => format!("jumps toward {}", at(band, o)),
            None => "jumps while the lane ahead is free".into(),
        },
        (EffectStyle::Position, Action::Roll) => match next_hazard {
            Some((band, o)) => format!("rolls toward {}", at(band, o)),
            None => "rolls while the lane ahead is free".into(),
        },
        (EffectStyle::Consequence, Action::Stay) => match next_hazard {
            Some((band, Obstacle::TrainRamp)) => format!("keeps you running up {}", at(band, Obstacle::TrainRamp)),
            Some((band, o)) => format!("keeps you running toward {}", at(band, o)),
            None => format!("keeps you in {}", lane_description(here)),
        },
        (EffectStyle::Consequence, Action::Jump) => match next_hazard {
            Some((band, o @ Obstacle::LowBarrier)) => format!("jumps over {}", at(band, o)),
            Some((band, o @ Obstacle::TrainRamp)) => format!("jumps onto {}", at(band, o)),
            Some((band, o)) => format!("jumps but cannot clear {}", at(band, o)),
            None => "jumps while the lane ahead is free".into(),
        },
        (EffectStyle::Consequence, Action::Roll) => match next_hazard {
            Some((band, o @ (Obstacle::LowBarrier | Obstacle::HighBarrier | Obstacle::OverheadBar))) => {
                format!("rolls under {}", at(band, o))
            }
            Some((band, o @ Obstacle::TrainRamp)) => format!("rolls up {}", at(band, o)),
            Some((band, o)) => format!("rolls but cannot pass {}", at(band, o)),
            None => "rolls while the lane ahead is free".into(),
        },
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// One hypothesis per possible action, in the Appendix A position wording.
pub fn options(obs: &Observation) -> BTreeMap<String, String> {
    options_with(obs, Wording { style: EffectStyle::Position, template: 0 })
}

pub fn options_with(obs: &Observation, wording: Wording) -> BTreeMap<String, String> {
    possible_actions(obs)
        .iter()
        .map(|a| (a.key().to_string(), wording.sentence(a, &effect_with(obs, a, wording.style))))
        .collect()
}

pub fn best_action(probs: &BTreeMap<String, f32>) -> Option<Action> {
    probs.iter().max_by(|a, b| a.1.total_cmp(b.1)).and_then(|(k, _)| Action::from_key(k))
}

impl Advisor {
    pub fn new(max_age_ms: u64, link: Option<SidecarLink>) -> Self {
        Self::with_wording(max_age_ms, link, Wording { style: EffectStyle::Position, template: 0 })
    }

    pub fn with_wording(max_age_ms: u64, link: Option<SidecarLink>, wording: Wording) -> Self {
        Self {
            max_age: Duration::from_millis(max_age_ms),
            wording,
            link,
            cache: HashMap::new(),
            in_flight: None,
            next_id: 1,
            stats: AdvisorStats::default(),
        }
    }

    pub fn connected(&self) -> bool {
        self.link.is_some()
    }

    /// Optional pre-warm: the sidecar precomputes these so their first lookup is fast.
    pub fn warm(&mut self, items: Vec<WarmItem>) {
        if let Some(link) = &self.link
            && !items.is_empty()
        {
            let _ = link.requests.send(Request::Warm { items });
        }
    }

    fn drain(&mut self, now: Instant) {
        let Some(link) = &mut self.link else { return };
        loop {
            match link.replies.try_recv() {
                Ok(Reply::Decision { id, probs, ms }) => {
                    let Some(flight) = self.in_flight.take_if(|f| f.id == id) else { continue };
                    self.stats.replies += 1;
                    self.stats.model_ms_total += ms as f64;
                    if now.duration_since(flight.sent) <= self.max_age {
                        self.stats.fresh += 1;
                    }
                    // A decision depends only on the facts string, so it stays valid in the
                    // cache; staleness only matters for whether it applies to the current frame.
                    if let Some(action) = best_action(&probs) {
                        self.cache.insert(flight.facts, Decision { action, probs, ms });
                    }
                }
                Ok(Reply::Error { id, message }) => {
                    tracing::warn!("advisor error: {message}");
                    self.stats.errors += 1;
                    if id.is_none() || self.in_flight.as_ref().map(|f| f.id) == id {
                        self.in_flight = None;
                    }
                }
                Ok(Reply::Ready { .. }) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    tracing::error!("advisor link closed; reflex-only from here");
                    self.link = None;
                    self.in_flight = None;
                    break;
                }
            }
        }
        // A request that never got an answer shouldn't block the next one forever.
        if self.in_flight.as_ref().is_some_and(|f| now.duration_since(f.sent) > self.max_age * 10) {
            self.in_flight = None;
        }
    }

    /// Called once per running frame. Returns advice that applies to *these* facts, if any;
    /// otherwise maybe sends a request for them.
    pub fn tick(&mut self, obs: &Observation, facts: Option<&str>, now: Instant) -> Option<AdvisorPick> {
        let fresh_before = self.stats.replies;
        self.drain(now);
        let facts = facts?;
        self.stats.lookups += 1;
        if let Some(d) = self.cache.get(facts) {
            // A reply that landed this tick is a fresh answer, not a cache hit.
            let cached = self.stats.replies == fresh_before;
            if cached {
                self.stats.cache_hits += 1;
            }
            return Some(AdvisorPick { action: d.action, cached });
        }
        if self.in_flight.is_none()
            && let Some(link) = &self.link
        {
            let options = options_with(obs, self.wording);
            if options.len() >= 2 {
                let id = self.next_id;
                self.next_id += 1;
                let req = Request::Decide { id, premise: facts.to_string(), options };
                if link.requests.send(req).is_ok() {
                    self.stats.requests += 1;
                    self.in_flight = Some(InFlight { id, facts: facts.to_string(), sent: now });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::facts;
    use crate::perception::Lane;

    fn obs() -> Observation {
        Observation::running(
            1,
            Lane::C,
            [
                LaneView { near: Obstacle::TrainBody, ..LaneView::FREE },
                LaneView { mid: Obstacle::LowBarrier, ..LaneView::FREE },
                LaneView { coins: true, ..LaneView::FREE },
            ],
        )
    }

    #[test]
    fn option_wording() {
        let o = options(&obs());
        assert_eq!(o["right"], "Right is the best move because it moves into a free lane with coins ahead.");
        assert_eq!(o["stay"], "Stay is the best move because it keeps you in a lane with a low barrier at mid distance.");
        assert_eq!(o["left"], "Left is the best move because it moves into a lane with a train close ahead.");
        assert!(!o.contains_key("hoverboard"));
    }

    #[test]
    fn consequence_wording() {
        let o = obs();
        let w = Wording { style: EffectStyle::Consequence, template: 0 };
        let opts = options_with(&o, w);
        assert_eq!(opts["stay"], "Stay is the best move because it keeps you running toward the low barrier at mid distance.");
        assert_eq!(opts["jump"], "Jump is the best move because it jumps over the low barrier at mid distance.");
        assert_eq!(opts["roll"], "Roll is the best move because it rolls under the low barrier at mid distance.");
        let t1 = Wording { style: EffectStyle::Consequence, template: 1 };
        assert!(options_with(&o, t1)["left"].starts_with("The best move is left, because it moves into"));
    }

    #[test]
    fn request_reply_cache() {
        let (link, mut requests, replies) = SidecarLink::channels();
        let mut adv = Advisor::new(400, Some(link));
        let obs = obs();
        let f = facts(&obs).unwrap();
        let t0 = Instant::now();
        assert_eq!(adv.tick(&obs, Some(&f), t0), None);
        let Request::Decide { id, premise, options } = requests.try_recv().unwrap() else { panic!() };
        assert_eq!(premise, f);
        assert!(options.contains_key("right"));
        // Only one request in flight at a time.
        assert_eq!(adv.tick(&obs, Some(&f), t0), None);
        assert!(requests.try_recv().is_err());

        replies.send(Reply::Decision { id, probs: BTreeMap::from([("right".into(), 0.7), ("stay".into(), 0.3)]), ms: 230.0 }).unwrap();
        let pick = adv.tick(&obs, Some(&f), t0 + Duration::from_millis(250)).unwrap();
        assert_eq!(pick, AdvisorPick { action: Action::Right, cached: false });
        let pick = adv.tick(&obs, Some(&f), t0 + Duration::from_millis(300)).unwrap();
        assert!(pick.cached);
        assert_eq!(adv.stats.fresh, 1);
        assert_eq!(adv.stats.cache_hits, 1);
    }

    #[test]
    fn stale_reply_is_not_fresh_and_other_facts_get_nothing() {
        let (link, mut requests, replies) = SidecarLink::channels();
        let mut adv = Advisor::new(400, Some(link));
        let obs = obs();
        let f = facts(&obs).unwrap();
        let t0 = Instant::now();
        adv.tick(&obs, Some(&f), t0);
        let Request::Decide { id, .. } = requests.try_recv().unwrap() else { panic!() };
        replies.send(Reply::Decision { id, probs: BTreeMap::from([("left".into(), 0.9)]), ms: 900.0 }).unwrap();
        // The facts changed in the meantime: the answer is cached for its own facts only.
        let other = "Subway run. You are in the left lane, on the ground.";
        assert_eq!(adv.tick(&obs, Some(other), t0 + Duration::from_millis(900)), None);
        assert_eq!(adv.stats.fresh, 0);
        assert_eq!(adv.stats.replies, 1);
    }

    #[test]
    fn disconnect_falls_back() {
        let (link, requests, replies) = SidecarLink::channels();
        drop(requests);
        drop(replies);
        let mut adv = Advisor::new(400, Some(link));
        let obs = obs();
        assert_eq!(adv.tick(&obs, facts(&obs).as_deref(), Instant::now()), None);
        assert!(!adv.connected());
    }
}

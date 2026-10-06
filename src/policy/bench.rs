//! Offline advisor benchmark (SPEC §9), like `../laya_tictactoe/benchmark.py`: situations with
//! a known best action, scored per option wording. The wording is chosen by score; wordings
//! that name the best move outright are never used.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::Action;
use super::advisor::{best_action, options_with};
use super::reflex::Reflex;
use crate::config::{PolicyConfig, Wording};
use crate::facts::facts;
use crate::perception::{Lane, LaneView, Obstacle, Observation};
use crate::sidecar::{Reply, Request, SidecarLink};

pub struct Case {
    pub name: String,
    pub obs: Observation,
    /// Any of these counts as correct.
    pub best: Vec<Action>,
}

fn view(near: Obstacle, mid: Obstacle, coins: bool) -> LaneView {
    LaneView { near, mid, far: Obstacle::Free, coins, powerup: false }
}

fn mirror_action(a: Action) -> Action {
    match a {
        Action::Left => Action::Right,
        Action::Right => Action::Left,
        other => other,
    }
}

/// Hand-written situations, plus their left–right mirror images.
pub fn cases() -> Vec<Case> {
    use Obstacle::*;
    let free = view(Free, Free, false);
    let coins = view(Free, Free, true);
    let train_near = view(TrainBody, Free, false);
    let train_mid = view(Free, TrainBody, false);
    let train_both = view(TrainBody, TrainBody, false);
    let base: Vec<(&str, Lane, [LaneView; 3], Vec<Action>)> = vec![
        ("train ahead, left free", Lane::C, [free, train_near, train_both], vec![Action::Left]),
        ("train ahead, right has coins", Lane::C, [train_both, train_near, coins], vec![Action::Right]),
        ("train at mid, right free", Lane::C, [train_both, train_mid, free], vec![Action::Right]),
        ("train ahead from left lane", Lane::L, [train_near, free, train_both], vec![Action::Right]),
        ("low barrier, sides blocked", Lane::C, [train_both, view(LowBarrier, Free, false), train_both], vec![Action::Jump, Action::Roll]),
        ("high barrier, sides blocked", Lane::C, [train_both, view(HighBarrier, Free, false), train_both], vec![Action::Roll]),
        ("overhead bar, sides blocked", Lane::C, [train_both, view(OverheadBar, Free, false), train_both], vec![Action::Roll]),
        ("low barrier in left lane", Lane::L, [view(LowBarrier, Free, false), train_both, train_both], vec![Action::Jump, Action::Roll]),
        ("high barrier in left lane", Lane::L, [view(HighBarrier, Free, false), train_near, free], vec![Action::Roll]),
        ("all free, coins here", Lane::C, [free, coins, free], vec![Action::Stay]),
        ("all free, coins to the right", Lane::C, [free, free, coins], vec![Action::Right]),
        ("ramp ahead is fine", Lane::C, [train_both, view(TrainRamp, TrainBody, false), train_both], vec![Action::Stay]),
        ("train in two lanes, stay in free", Lane::R, [train_both, train_both, coins], vec![Action::Stay]),
        ("coins in left, centre blocked at mid", Lane::C, [coins, train_mid, train_both], vec![Action::Left]),
        ("barrier at mid, left free", Lane::C, [free, view(Free, HighBarrier, false), train_both], vec![Action::Left, Action::Roll]),
        ("low barrier ahead, right free with coins", Lane::C, [train_both, view(LowBarrier, Free, false), coins], vec![Action::Right, Action::Jump, Action::Roll]),
    ];
    let mut out = Vec::new();
    for (name, lane, lanes, best) in base {
        out.push(Case { name: name.into(), obs: Observation::running(0, lane, lanes), best: best.clone() });
        let mirrored_lane = Lane::from_index(2 - lane.index()).unwrap();
        let mirrored = [lanes[2], lanes[1], lanes[0]];
        if mirrored != lanes || mirrored_lane != lane {
            out.push(Case {
                name: format!("{name} (mirrored)"),
                obs: Observation::running(0, mirrored_lane, mirrored),
                best: best.iter().map(|a| mirror_action(*a)).collect(),
            });
        }
    }
    out
}

#[derive(Debug, Default, Serialize)]
pub struct WordingScore {
    pub wording: String,
    pub correct: usize,
    /// Correct after the reflex safety mask vetoes unsafe picks (what the arbiter would do).
    pub correct_after_mask: usize,
    pub total: usize,
    pub mean_ms: f64,
    pub misses: Vec<String>,
}

/// Reference point: the reflex default alone, no advisor.
pub fn rules_only() -> WordingScore {
    let cases = cases();
    let reflex = Reflex::new(PolicyConfig::default());
    let mut score = WordingScore { wording: "rules only".into(), total: cases.len(), ..Default::default() };
    for case in &cases {
        let out = reflex.evaluate(&case.obs);
        let ok = case.best.contains(&out.action);
        score.correct += ok as usize;
        score.correct_after_mask += ok as usize;
        if !ok {
            score.misses.push(format!("{}: picked {:?}", case.name, out.action));
        }
    }
    score
}

async fn ask(link: &mut SidecarLink, id: u64, premise: String, options: BTreeMap<String, String>) -> Result<(BTreeMap<String, f32>, f32)> {
    link.requests.send(Request::Decide { id, premise, options }).context("sidecar gone")?;
    loop {
        match tokio::time::timeout(Duration::from_secs(60), link.replies.recv()).await? {
            Some(Reply::Decision { id: rid, probs, ms }) if rid == id => return Ok((probs, ms)),
            Some(Reply::Error { message, .. }) => bail!("sidecar error: {message}"),
            Some(_) => continue,
            None => bail!("sidecar exited"),
        }
    }
}

pub async fn run(link: &mut SidecarLink, wordings: &[Wording]) -> Result<Vec<WordingScore>> {
    let cases = cases();
    let reflex = Reflex::new(PolicyConfig::default());
    let mut scores = vec![rules_only()];
    let mut id = 0;
    for w in wordings {
        let mut score = WordingScore { wording: format!("{:?}/{}", w.style, w.template), total: cases.len(), ..Default::default() };
        let start = Instant::now();
        let mut model_ms = 0.0;
        for case in &cases {
            id += 1;
            let premise = facts(&case.obs).context("facts")?;
            let (probs, ms) = ask(link, id, premise, options_with(&case.obs, *w)).await?;
            model_ms += ms as f64;
            let pick = best_action(&probs);
            let mask = reflex.safety_mask(&case.obs);
            // The arbiter's fallback when advice is masked out is the reflex default.
            let masked = pick.filter(|a| mask.contains(*a)).unwrap_or_else(|| reflex.default_action(&case.obs, mask));
            let ok = pick.is_some_and(|a| case.best.contains(&a));
            score.correct += ok as usize;
            score.correct_after_mask += case.best.contains(&masked) as usize;
            if !ok {
                score.misses.push(format!("{}: picked {:?}", case.name, pick));
            }
        }
        score.mean_ms = model_ms / cases.len() as f64;
        tracing::info!("{}: {}/{} in {:.1} s", score.wording, score.correct, score.total, start.elapsed().as_secs_f64());
        scores.push(score);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_answers_pass_the_reflex_mask() {
        // Every case's expected answer must be something the reflex layer allows, or the
        // benchmark would reward advice the arbiter throws away.
        let reflex = Reflex::new(PolicyConfig::default());
        for case in cases() {
            let mask = reflex.safety_mask(&case.obs);
            assert!(case.best.iter().any(|a| mask.contains(*a)), "{}: mask {:?}", case.name, mask.iter().collect::<Vec<_>>());
        }
    }

    #[test]
    fn mirrored_cases_exist() {
        let c = cases();
        assert!(c.len() > 25);
        assert!(c.iter().any(|c| c.name.ends_with("(mirrored)")));
    }
}

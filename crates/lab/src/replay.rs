//! `ssbot replay runs/<ts>` (SPEC §4.7): re-runs perception and policy on a run's saved frames
//! with no browser, for tuning thresholds and catching regressions. The advisor isn't called;
//! unless `--no-advisor`, the recorded advice is reused wherever the facts still match.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use ssbot_engine::config::Config;
use ssbot_engine::facts::facts;
use ssbot_engine::perception::zones::Calibration;
use ssbot_engine::perception::{GameState, Perceiver};
use ssbot_engine::policy::Action;
use ssbot_engine::policy::arbiter::{Arbiter, Command};
use ssbot_engine::policy::reflex::Reflex;
use ssbot_runtime::recorder::{EventLine, read_events};

use crate::label::select::list_frames;

#[derive(Debug, Default, Serialize)]
pub struct ReplayReport {
    pub frames: usize,
    pub state_changed: usize,
    pub lane_view_changed: usize,
    pub player_lane_changed: usize,
    pub action_changed: usize,
    pub emergencies: usize,
    /// Running frames where nothing but the hoverboard was safe: a crash the policy predicts
    /// it can't avoid. Regression gate: this must not go up (SPEC §9).
    pub predicted_crashes: usize,
    /// Actions the replayed policy chose, by kind.
    pub actions: std::collections::BTreeMap<String, usize>,
    pub perceive_p95_ms: f64,
    /// How many frames perception put in each game state.
    pub states: std::collections::BTreeMap<String, usize>,
}

pub fn replay(run_dir: &Path, cfg: &Config, calib_path: &Path, use_advice: bool) -> Result<ReplayReport> {
    let events: HashMap<u64, EventLine> = read_events(run_dir)?.into_iter().map(|e| (e.frame_id, e)).collect();
    let frames = list_frames(&run_dir.join("frames"))?;
    let calib = Calibration::load_or_default(calib_path);
    let base = calib_path.parent().unwrap_or(Path::new("."));
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let mut perceiver = Perceiver::new(calib.clone(), base, work);
    let mut reflex = Reflex::new(cfg.policy.clone());
    let mut arbiter = Arbiter::new(cfg.policy.cooldown_ms, calib.canvas_center).with_hoverboard(cfg.policy.use_hoverboard);
    let mut report = ReplayReport::default();
    let mut lanes = ssbot_engine::policy::LaneTracker::default();
    let mut was_running = false;
    let mut perceive_ms = Vec::new();
    let t0 = Instant::now();

    for (id, path) in frames {
        let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let img = image::imageops::resize(&img, work.0, work.1, image::imageops::FilterType::Triangle);
        let start = Instant::now();
        let m = perceiver.measure(id, &img);
        perceive_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        let mut obs = m.obs;
        if obs.state == GameState::Running {
            if !was_running {
                lanes.reset();
            }
            obs.player_lane = Some(lanes.lane());
            obs.lanes = ssbot_engine::perception::to_absolute(obs.lanes, lanes.lane());
        }
        was_running = obs.state == GameState::Running;
        let recorded = events.get(&id);
        let t = recorded.map(|e| e.t).unwrap_or(id as f64 * 33.0);
        reflex.update(&obs, t);
        let out = reflex.evaluate(&obs);
        let advice = if use_advice {
            recorded.and_then(|e| e.advisor.clone().filter(|_| e.facts.is_some() && e.facts == facts(&obs)))
        } else {
            None
        };
        // Replay time: the arbiter's cooldown runs on the recorded clock.
        let now = t0 + Duration::from_secs_f64(t / 1000.0);
        let chosen = arbiter.decide(&obs, m.marker_click, &out, advice.as_ref(), now);
        if let Command::Act(a) = chosen.command {
            lanes.apply(a);
            *report.actions.entry(format!("{a:?}")).or_default() += 1;
        }

        report.frames += 1;
        *report.states.entry(format!("{:?}", obs.state)).or_default() += 1;
        if obs.state == GameState::Running {
            report.emergencies += out.emergency as usize;
            report.predicted_crashes += (out.mask == vec![Action::Hoverboard]) as usize;
        }
        if let Some(e) = recorded {
            report.state_changed += (e.obs.state != obs.state) as usize;
            report.lane_view_changed += (e.obs.lanes != obs.lanes) as usize;
            report.player_lane_changed += (e.obs.player_lane != obs.player_lane) as usize;
            let act = |c: Command| matches!(c, Command::Act(_)).then_some(c);
            report.action_changed += (act(e.chosen.command) != act(chosen.command)) as usize;
        }
    }
    report.perceive_p95_ms = ssbot_runtime::recorder::percentile(&mut perceive_ms, 0.95);
    Ok(report)
}

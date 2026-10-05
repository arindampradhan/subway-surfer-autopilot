//! Recorder (SPEC §4.7): `runs/<ts>/frames/<id>.jpg`, `events.jsonl` and `summary.json`.
//! Frames are sampled every n-th frame, plus the last `keep_pre_crash_ms` before every crash.
//! JPEG encoding runs on its own thread so it stays off the reflex path.

use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::perception::{GameState, Observation};
use crate::policy::Action;
use crate::policy::advisor::{AdvisorPick, AdvisorStats};
use crate::policy::arbiter::{Chosen, Command};
use crate::policy::reflex::ReflexOut;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    /// Browser → us (screencast metadata), when known.
    pub capture_ms: Option<f32>,
    pub perceive_ms: f32,
    pub policy_ms: f32,
    pub dispatch_ms: f32,
    /// Capture delay + everything up to the key event being sent: the reflex path (SPEC §5).
    pub total_ms: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLine {
    pub frame_id: u64,
    /// Milliseconds since the run started.
    pub t: f64,
    pub obs: Observation,
    pub facts: Option<String>,
    pub reflex: ReflexOut,
    pub advisor: Option<AdvisorPick>,
    pub chosen: Chosen,
    pub latency_ms: Latency,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Summary {
    pub frames: u64,
    pub survival_s: f64,
    pub downtime_s: f64,
    pub crashed: bool,
    pub crash_cause: Option<String>,
    pub actions: u64,
    pub actions_per_minute: f64,
    pub by_source: std::collections::BTreeMap<String, u64>,
    pub advisor: Option<AdvisorStats>,
    pub advisor_freshness_rate: Option<f64>,
    pub advisor_cache_hit_rate: Option<f64>,
    pub reflex_p50_ms: f64,
    pub reflex_p95_ms: f64,
    pub capture_p95_ms: Option<f64>,
    pub fps: f64,
}

pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    values[idx]
}

/// Crash cause from the last 2 s of events: what was in the player's lane just before.
pub fn crash_cause(events: &[EventLine], crash_t: f64) -> Option<String> {
    let window: Vec<&EventLine> =
        events.iter().filter(|e| e.t >= crash_t - 2000.0 && e.t < crash_t && e.obs.state == GameState::Running).collect();
    let last = window.iter().rev().find(|e| e.obs.player_lane.is_some())?;
    let lane = last.obs.player_lane?;
    let view = last.obs.lanes[lane.index()];
    let hazard = view.bands().into_iter().zip(["near", "mid", "far"]).find(|(o, _)| *o != crate::perception::Obstacle::Free);
    let seen = match hazard {
        Some((o, band)) => format!("{o:?} at {band} in {lane} lane"),
        None => format!("nothing seen in {lane} lane (perception miss?)"),
    };
    let last_act = window.iter().rev().find_map(|e| match e.chosen.command {
        Command::Act(a) => Some((a, e.chosen.source)),
        _ => None,
    });
    let act = match last_act {
        Some((a, s)) => format!("; last action {a:?} by {s:?}"),
        None => "; no action in the last 2 s".into(),
    };
    let no_escape = window.iter().any(|e| e.reflex.mask == vec![Action::Hoverboard]);
    Some(format!("{seen}{act}{}", if no_escape { "; reflex saw no safe action" } else { "" }))
}

/// Metrics from a run's events (SPEC §9). Survival counts only `Running` time, so ads are
/// downtime.
pub fn summarise(events: &[EventLine], advisor: Option<AdvisorStats>) -> Summary {
    let mut s = Summary { frames: events.len() as u64, advisor, ..Default::default() };
    let mut running_ms = 0.0;
    for w in events.windows(2) {
        let dt = w[1].t - w[0].t;
        if w[0].obs.state == GameState::Running {
            running_ms += dt;
        } else {
            s.downtime_s += dt / 1000.0;
        }
    }
    s.survival_s = running_ms / 1000.0;
    if let Some(crash) = events.iter().find(|e| e.obs.state == GameState::Crashed) {
        s.crashed = true;
        s.crash_cause = crash_cause(events, crash.t);
    }
    for e in events {
        if let Command::Act(_) = e.chosen.command {
            s.actions += 1;
            *s.by_source.entry(format!("{:?}", e.chosen.source)).or_default() += 1;
        }
    }
    s.actions_per_minute = if running_ms > 0.0 { s.actions as f64 / (running_ms / 60_000.0) } else { 0.0 };
    let mut lat: Vec<f64> = events.iter().filter(|e| e.obs.state == GameState::Running).map(|e| e.latency_ms.total_ms as f64).collect();
    s.reflex_p50_ms = percentile(&mut lat, 0.5);
    s.reflex_p95_ms = percentile(&mut lat, 0.95);
    let mut cap: Vec<f64> = events.iter().filter_map(|e| e.latency_ms.capture_ms.map(|c| c as f64)).collect();
    s.capture_p95_ms = (!cap.is_empty()).then(|| percentile(&mut cap, 0.95));
    if let (Some(a), Some(b)) = (events.first(), events.last())
        && b.t > a.t
    {
        s.fps = (events.len() - 1) as f64 / ((b.t - a.t) / 1000.0);
    }
    s.advisor_freshness_rate = advisor.map(|a| a.freshness_rate());
    s.advisor_cache_hit_rate = advisor.map(|a| a.cache_hit_rate());
    s
}

pub fn read_events(dir: &Path) -> Result<Vec<EventLine>> {
    let path = dir.join("events.jsonl");
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<std::io::Result<_>>()?;
    let n = lines.len();
    let mut out = Vec::with_capacity(n);
    for (i, line) in lines.into_iter().enumerate() {
        match serde_json::from_str(&line) {
            Ok(ev) => out.push(ev),
            // A run killed mid-write leaves a truncated last line; keep everything before it.
            Err(_) if i + 1 == n => tracing::warn!("{}: ignoring truncated last line", path.display()),
            Err(err) => return Err(err).with_context(|| format!("{}:{}", path.display(), i + 1)),
        }
    }
    Ok(out)
}

pub struct Recorder {
    pub dir: PathBuf,
    events: BufWriter<File>,
    lines: Vec<EventLine>,
    sample_every: u64,
    keep_pre_crash_ms: f64,
    ring: VecDeque<(u64, f64, Arc<RgbImage>)>,
    saved: HashSet<u64>,
    writer: Option<(mpsc::Sender<(PathBuf, Arc<RgbImage>)>, JoinHandle<()>)>,
    was_crashed: bool,
}

impl Recorder {
    pub fn create(runs_dir: &Path, sample_every: u64, keep_pre_crash_ms: u64) -> Result<Recorder> {
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S%.3f").to_string();
        let dir = runs_dir.join(ts);
        std::fs::create_dir_all(dir.join("frames")).with_context(|| format!("creating {}", dir.display()))?;
        let events = BufWriter::new(File::create(dir.join("events.jsonl"))?);
        let (tx, rx) = mpsc::channel::<(PathBuf, Arc<RgbImage>)>();
        let handle = std::thread::spawn(move || {
            for (path, img) in rx {
                if let Err(err) = img.save_with_format(&path, image::ImageFormat::Jpeg) {
                    tracing::warn!("saving {}: {err}", path.display());
                }
            }
        });
        Ok(Recorder {
            dir,
            events,
            lines: Vec::new(),
            sample_every: sample_every.max(1),
            keep_pre_crash_ms: keep_pre_crash_ms as f64,
            ring: VecDeque::new(),
            saved: HashSet::new(),
            writer: Some((tx, handle)),
            was_crashed: false,
        })
    }

    fn save_frame(&mut self, id: u64, img: Arc<RgbImage>) {
        if self.saved.insert(id)
            && let Some((tx, _)) = &self.writer
        {
            let _ = tx.send((self.dir.join("frames").join(format!("{id}.jpg")), img));
        }
    }

    pub fn record(&mut self, canvas: Arc<RgbImage>, ev: EventLine) -> Result<()> {
        let (id, t) = (ev.frame_id, ev.t);
        if id % self.sample_every == 0 {
            self.save_frame(id, canvas.clone());
        }
        self.ring.push_back((id, t, canvas));
        while self.ring.front().is_some_and(|(_, ft, _)| t - ft > self.keep_pre_crash_ms) {
            self.ring.pop_front();
        }
        let crashed = ev.obs.state == GameState::Crashed;
        if crashed && !self.was_crashed {
            // Always keep the frames leading up to a crash.
            let ring: Vec<_> = self.ring.drain(..).collect();
            for (fid, _, img) in ring {
                self.save_frame(fid, img);
            }
        }
        self.was_crashed = crashed;
        serde_json::to_writer(&mut self.events, &ev)?;
        self.events.write_all(b"\n")?;
        self.lines.push(ev);
        Ok(())
    }

    pub fn events(&self) -> &[EventLine] {
        &self.lines
    }

    pub fn finish(mut self, advisor: Option<AdvisorStats>) -> Result<Summary> {
        self.events.flush()?;
        if let Some((tx, handle)) = self.writer.take() {
            drop(tx);
            let _ = handle.join();
        }
        let summary = summarise(&self.lines, advisor);
        std::fs::write(self.dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::{Lane, LaneView, Obstacle};
    use crate::policy::arbiter::Source;

    fn ev(id: u64, t: f64, state: GameState, act: Option<Action>) -> EventLine {
        let mut obs = Observation::running(id, Lane::C, [LaneView::FREE, LaneView { near: Obstacle::TrainBody, ..LaneView::FREE }, LaneView::FREE]);
        obs.state = state;
        EventLine {
            frame_id: id,
            t,
            obs,
            facts: None,
            reflex: ReflexOut { mask: vec![Action::Left], emergency: true, action: Action::Left, tti_ms: Some(160.0), speed: 3.0, fallback: Action::Left },
            advisor: None,
            chosen: Chosen {
                command: act.map(Command::Act).unwrap_or(Command::Wait),
                source: if act.is_some() { Source::Reflex } else { Source::Flow },
            },
            latency_ms: Latency { total_ms: 30.0 + id as f32, ..Default::default() },
        }
    }

    #[test]
    fn summary_metrics() {
        let mut evs: Vec<EventLine> = (0..100).map(|i| ev(i, i as f64 * 50.0, GameState::Running, (i % 10 == 0).then_some(Action::Left))).collect();
        evs.push(ev(100, 5000.0, GameState::Crashed, None));
        evs.push(ev(101, 9000.0, GameState::AdBreak, None));
        let s = summarise(&evs, None);
        assert!((s.survival_s - 5.0).abs() < 1e-9);
        assert!((s.downtime_s - 4.0).abs() < 1e-9);
        assert!(s.crashed);
        assert_eq!(s.actions, 10);
        assert!((s.actions_per_minute - 120.0).abs() < 1e-6);
        let cause = s.crash_cause.unwrap();
        assert!(cause.starts_with("TrainBody at near in center lane"), "{cause}");
        assert!(s.reflex_p95_ms >= s.reflex_p50_ms);
    }

    #[test]
    fn records_pre_crash_frames() {
        let tmp = std::env::temp_dir().join(format!("ssbot-rec-{}", std::process::id()));
        let mut r = Recorder::create(&tmp, 1000, 2000).unwrap();
        let img = Arc::new(RgbImage::new(8, 8));
        for i in 1..=60u64 {
            let state = if i == 60 { GameState::Crashed } else { GameState::Running };
            r.record(img.clone(), ev(i, i as f64 * 100.0, state, None)).unwrap();
        }
        let dir = r.dir.clone();
        r.finish(None).unwrap();
        let n = std::fs::read_dir(dir.join("frames")).unwrap().count();
        // 2 s at 100 ms per frame ≈ 21 frames kept before the crash; no regular samples.
        assert!((20..=22).contains(&n), "{n} frames");
        assert_eq!(read_events(&dir).unwrap().len(), 60);
        assert!(dir.join("summary.json").exists());
        let _ = std::fs::remove_dir_all(tmp);
    }
}

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
    /// HUD score read from the last saved running frames (needs `tesseract`; `None` without it).
    pub score: Option<u64>,
    /// The run folder these numbers came from.
    pub run_dir: Option<String>,
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

/// States that end a run. The crash detector only reads `Crashed` when the screen freezes, so
/// the revive prompt and the score screens count as the run ending too.
pub fn ends_run(state: GameState) -> bool {
    matches!(state, GameState::Crashed | GameState::RevivePrompt | GameState::NewHighScore | GameState::ScoreScreen)
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
    let first_run = events.iter().position(|e| e.obs.state == GameState::Running);
    if let Some(end) = first_run.and_then(|i| events[i..].iter().find(|e| ends_run(e.obs.state))) {
        s.crashed = true;
        s.crash_cause = crash_cause(events, end.t);
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

/// Reads the score off the top-right HUD with `tesseract`: the zero-padded six-digit number, with
/// the crop starting to the right of the "x2" multiplier badge. Takes the last six digits read
/// in case a stray character gets in.
pub fn hud_score(frame: &Path, scratch: &Path) -> Option<u64> {
    let img = image::open(frame).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    let (x, y) = ((w as f32 * 0.885) as u32, (h as f32 * 0.055) as u32);
    let crop = image::imageops::crop_imm(&img, x, y, w - x - (w as f32 * 0.01) as u32, (h as f32 * 0.095) as u32).to_image();
    let big = image::imageops::resize(&crop, crop.width() * 4, crop.height() * 4, image::imageops::FilterType::Lanczos3);
    // White digits on a dark plate -> black digits on white, which tesseract reads best.
    let bin = image::GrayImage::from_fn(big.width(), big.height(), |px, py| {
        let p = big.get_pixel(px, py).0;
        let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8;
        image::Luma([if lum > 170 { 0 } else { 255 }])
    });
    bin.save(scratch).ok()?;
    let out = std::process::Command::new("tesseract")
        .arg(scratch)
        .args(["stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789"])
        .output()
        .ok()?;
    let digits: String = String::from_utf8_lossy(&out.stdout).chars().filter(char::is_ascii_digit).collect();
    (digits.len() >= 6).then(|| digits[digits.len() - 6..].parse().ok()).flatten()
}

/// The HUD score from the last saved running frames of a run folder. The score never goes down
/// within a run, so the best of the last three readable frames is the final one.
pub fn run_score(dir: &Path, events: &[EventLine]) -> Option<u64> {
    let frames = dir.join("frames");
    // A misread can't be higher than the run could plausibly have scored: ~40 points a second
    // in the runs seen, so 300 a second is a very generous ceiling.
    let running_s: f64 = events.windows(2).filter(|w| w[0].obs.state == GameState::Running).map(|w| (w[1].t - w[0].t) / 1000.0).sum();
    let ceiling = (300.0 * running_s + 300.0) as u64;
    events
        .iter()
        .rev()
        .filter(|e| e.obs.state == GameState::Running)
        .map(|e| frames.join(format!("{}.jpg", e.frame_id)))
        .filter(|p| p.exists())
        .take(3)
        .filter_map(|p| hud_score(&p, &dir.join("score_ocr.png")))
        .filter(|s| *s <= ceiling)
        .max()
}

/// Saved frames from the last `window_ms` of running before a run ended, thinned to at most
/// `count` evenly spaced ones: the moments the bot misjudged. Empty if the run didn't end.
pub fn crash_window(dir: &Path, events: &[EventLine], window_ms: f64, count: usize) -> Vec<u64> {
    let Some(first_run) = events.iter().position(|e| e.obs.state == GameState::Running) else { return Vec::new() };
    let Some(end) = events[first_run..].iter().find(|e| ends_run(e.obs.state)) else { return Vec::new() };
    let frames = dir.join("frames");
    let ids: Vec<u64> = events
        .iter()
        .filter(|e| e.obs.state == GameState::Running && e.t >= end.t - window_ms && e.t < end.t)
        .map(|e| e.frame_id)
        .filter(|id| frames.join(format!("{id}.jpg")).exists())
        .collect();
    if ids.len() <= count {
        return ids;
    }
    (0..count).map(|i| ids[i * (ids.len() - 1) / (count - 1).max(1)]).collect()
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
    in_run: bool,
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
            in_run: false,
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
        if ev.obs.state == GameState::Running {
            self.in_run = true;
        } else if self.in_run && ends_run(ev.obs.state) {
            // Always keep the frames leading up to the end of a run.
            self.in_run = false;
            let ring: Vec<_> = self.ring.drain(..).collect();
            for (fid, _, img) in ring {
                self.save_frame(fid, img);
            }
        }
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
        let mut summary = summarise(&self.lines, advisor);
        summary.score = run_score(&self.dir, &self.lines);
        summary.run_dir = Some(self.dir.display().to_string());
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
    fn revive_prompt_counts_as_a_crash() {
        let mut evs: Vec<EventLine> = (0..40).map(|i| ev(i, i as f64 * 50.0, GameState::Running, None)).collect();
        evs.push(ev(40, 2100.0, GameState::Unknown, None));
        evs.push(ev(41, 2900.0, GameState::RevivePrompt, None));
        let s = summarise(&evs, None);
        assert!(s.crashed);
        assert!(s.crash_cause.unwrap().starts_with("TrainBody at near in center lane"));
        // A run that never started running isn't a crash.
        let idle: Vec<EventLine> = (0..5).map(|i| ev(i, i as f64 * 50.0, GameState::ScoreScreen, None)).collect();
        assert!(!summarise(&idle, None).crashed);
    }

    #[test]
    fn crash_window_picks_frames_before_the_end() {
        let dir = std::env::temp_dir().join(format!("ssbot_cw_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("frames")).unwrap();
        let mut evs: Vec<EventLine> = (0..100).map(|i| ev(i, i as f64 * 50.0, GameState::Running, None)).collect();
        evs.push(ev(100, 5000.0, GameState::RevivePrompt, None));
        for e in &evs {
            std::fs::write(dir.join("frames").join(format!("{}.jpg", e.frame_id)), b"x").unwrap();
        }
        let ids = crash_window(&dir, &evs, 2000.0, 5);
        assert_eq!(ids.len(), 5);
        assert!(ids.iter().all(|&i| (60..100).contains(&i)), "{ids:?}");
        assert_eq!(*ids.last().unwrap(), 99);
        assert!(crash_window(&dir, &evs[..50], 2000.0, 5).is_empty(), "no run end, no window");
        std::fs::remove_dir_all(&dir).ok();
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

//! The live loop (SPEC §3): capture → perception → facts → reflex → (advisor) → arbiter →
//! key press, recording every frame. Also the M0 spike that answers SPEC §10 questions 1, 2, 5.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::browser::Game;
use crate::capture::{Crop, FrameSource, LatestFrame, screencast::Screencast};
use crate::config::{CaptureBackend, Config};
use crate::facts::facts;
use crate::perception::zones::Calibration;
use crate::perception::{GameState, Perceiver};
use crate::policy::advisor::{Advisor, options_with};
use crate::policy::arbiter::{Arbiter, Command};
use crate::policy::reflex::Reflex;
use crate::recorder::{EventLine, Latency, Recorder, Summary, percentile, read_events};
use crate::sidecar::{Sidecar, WarmItem};

pub struct RunOpts {
    pub runs: usize,
    pub advisor: bool,
    pub capture: CaptureBackend,
    /// A human plays; the bot only records (M1).
    pub human: bool,
    pub calibration: std::path::PathBuf,
}

struct Capture {
    source: LatestFrame,
    screencast: Option<Screencast>,
}

async fn start_capture(game: &Game, cfg: &Config, backend: CaptureBackend) -> Result<Capture> {
    let [vw, vh] = game.viewport();
    let c = game.canvas;
    let crop = Crop { x: c.x / vw as f64, y: c.y / vh as f64, w: c.w / vw as f64, h: c.h / vh as f64 };
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    match backend {
        CaptureBackend::Screencast => {
            let (sc, source) = Screencast::start(&game.page, crop, cfg.capture.jpeg_quality, cfg.capture.max_width, work).await?;
            Ok(Capture { source, screencast: Some(sc) })
        }
        #[cfg(feature = "xcap")]
        CaptureBackend::Native => {
            let source = crate::capture::native::start(game.viewport(), crop, cfg.capture.max_width, work, 60)?;
            Ok(Capture { source, screencast: None })
        }
        #[cfg(not(feature = "xcap"))]
        CaptureBackend::Native => anyhow::bail!("native capture needs `cargo build --features xcap`"),
    }
}

async fn dispatch(game: &Game, cmd: Command, hold: Duration) -> Result<()> {
    match cmd {
        Command::Wait => Ok(()),
        Command::Act(a) => match a.dom_key() {
            Some(k) => game.press(k, hold).await,
            None => Ok(()),
        },
        Command::PressSpace => game.press("Space", hold).await,
        Command::Click(p) => game.click(p).await,
    }
}

/// Most frequent facts strings in earlier recordings, with their options, for the sidecar's
/// optional pre-warm (SPEC §4.5).
fn prewarm_items(dirs: &[String], top: usize, wording: crate::policy::advisor::Wording) -> Vec<WarmItem> {
    let mut counts: std::collections::HashMap<String, (usize, crate::perception::Observation)> = Default::default();
    for dir in dirs {
        let Ok(events) = read_events(Path::new(dir)) else { continue };
        for e in events {
            if let Some(f) = e.facts {
                counts.entry(f).or_insert((0, e.obs)).0 += 1;
            }
        }
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    v.into_iter().take(top).map(|(premise, (_, obs))| WarmItem { premise, options: options_with(&obs, wording) }).collect()
}

pub async fn run(cfg: &Config, opts: &RunOpts) -> Result<Vec<Summary>> {
    let calib = Calibration::load_or_default(&opts.calibration);
    let base = opts.calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let mut game = Game::launch(&cfg.browser).await?;
    game.open(&cfg.browser).await?;
    game.focus().await?;
    let canvas_center = calib.canvas_center;

    let mut sidecar = None;
    let mut link = None;
    if opts.advisor && cfg.advisor.enabled && !opts.human {
        match Sidecar::spawn(&cfg.advisor).await {
            Ok((s, l)) => {
                tracing::info!("advisor ready: {}", s.model);
                sidecar = Some(s);
                link = Some(l);
            }
            Err(err) => tracing::error!("advisor unavailable ({err:#}); playing reflex-only"),
        }
    }
    let mut advisor = Advisor::with_wording(cfg.policy.advisor_max_age_ms, link, cfg.advisor.wording);
    advisor.warm(prewarm_items(&cfg.advisor.prewarm_from, cfg.advisor.prewarm_top, cfg.advisor.wording));

    let mut capture = start_capture(&game, cfg, opts.capture).await?;
    let mut perceiver = Perceiver::new(calib, &base, work);
    let mut reflex = Reflex::new(cfg.policy.clone());
    let mut arbiter = Arbiter::new(cfg.policy.cooldown_ms, canvas_center).with_hoverboard(cfg.policy.use_hoverboard);
    let hold = Duration::from_millis(cfg.browser.key_hold_ms);
    let runs_dir = Path::new(&cfg.recorder.runs_dir).to_path_buf();
    let mut summaries = Vec::new();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    if opts.human {
        eprintln!("Recording you play. Close the browser window (or press Ctrl-C) when you're done.");
    }
    'runs: for run_no in 1..=opts.runs {
        let mut rec = Recorder::create(&runs_dir, cfg.recorder.frame_sample_every, cfg.recorder.keep_pre_crash_ms)?;
        tracing::info!("run {run_no}/{}: recording to {}", opts.runs, rec.dir.display());
        reflex.reset();
        let mut lanes = crate::policy::LaneTracker::default();
        let started = Instant::now();
        let mut ended = false; // saw the run end (crash or score screens)
        let mut ran = false;
        let stats_before = advisor.stats;
        loop {
            let frame = tokio::select! {
                f = tokio::time::timeout(Duration::from_secs(15), capture.source.next()) => match f {
                    Ok(Ok(f)) => f,
                    Err(_) => {
                        // No frames for 15 s: the Mac slept or the window was hidden. Don't hang.
                        tracing::warn!("no frames for 15 s (window hidden or Mac asleep?); saving and stopping");
                        summaries.push(rec.finish(advisor.connected().then_some(advisor.stats))?);
                        break 'runs;
                    }
                    Ok(Err(err)) => {
                        // The browser window was closed (or Chrome quit): keep what was recorded.
                        tracing::info!("capture ended ({err:#}); was the browser window closed? saving and stopping");
                        summaries.push(rec.finish(advisor.connected().then_some(advisor.stats))?);
                        break 'runs;
                    }
                },
                _ = &mut ctrl_c => {
                    tracing::info!("interrupted");
                    summaries.push(rec.finish(advisor.connected().then_some(advisor.stats))?);
                    break 'runs;
                }
            };
            let t_frame = frame.t_capture;
            let m = perceiver.measure(frame.id, &frame.rgb);
            let perceived = Instant::now();
            let mut obs = m.obs;
            if obs.state == GameState::Running {
                if !ran {
                    lanes.reset(); // a run starts in the centre lane
                }
                obs.player_lane = Some(lanes.lane());
            obs.lanes = crate::perception::to_absolute(obs.lanes, lanes.lane());
            }
            let t = t_frame.duration_since(started).as_secs_f64() * 1000.0;
            reflex.update(&obs, t);
            let reflex_out = reflex.evaluate(&obs);
            let fact_text = if obs.state == GameState::Running { facts(&obs) } else { None };
            let advice = if obs.state == GameState::Running { advisor.tick(&obs, fact_text.as_deref(), Instant::now()) } else { None };
            let mut chosen = arbiter.decide(&obs, m.marker_click, &reflex_out, advice.as_ref(), Instant::now());
            if opts.human {
                chosen.command = Command::Wait;
            }
            let decided = Instant::now();
            match dispatch(&game, chosen.command, hold).await {
                Ok(()) => {
                    if let Command::Act(a) = chosen.command {
                        lanes.apply(a);
                    }
                }
                Err(err) => tracing::warn!("input failed: {err:#}"),
            }
            let done = Instant::now();
            let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f32() * 1000.0;
            let latency = Latency {
                capture_ms: frame.delay_ms,
                perceive_ms: ms(t_frame, perceived),
                policy_ms: ms(perceived, decided),
                dispatch_ms: ms(decided, done),
                total_ms: frame.delay_ms.unwrap_or(0.0).max(0.0) + ms(t_frame, done),
            };
            rec.record(
                frame.canvas.clone(),
                EventLine { frame_id: frame.id, t, obs: obs.clone(), facts: fact_text, reflex: reflex_out, advisor: advice, chosen, latency_ms: latency },
            )?;
            // A human session records until the window is closed: no run boundaries.
            match obs.state {
                _ if opts.human => {}
                GameState::Running if ended => break, // the next run has started
                GameState::Running => ran = true,
                GameState::Crashed | GameState::NewHighScore | GameState::ScoreScreen if ran => ended = true,
                _ => {}
            }
        }
        let mut stats = advisor.stats;
        stats.requests -= stats_before.requests;
        stats.replies -= stats_before.replies;
        stats.fresh -= stats_before.fresh;
        stats.errors -= stats_before.errors;
        stats.lookups -= stats_before.lookups;
        stats.cache_hits -= stats_before.cache_hits;
        stats.model_ms_total -= stats_before.model_ms_total;
        let summary = rec.finish(advisor.connected().then_some(stats))?;
        tracing::info!(
            "run {run_no}: survived {:.1} s, cause: {}",
            summary.survival_s,
            summary.crash_cause.as_deref().unwrap_or("-")
        );
        summaries.push(summary);
    }
    if let Some(sc) = capture.screencast.take() {
        // Bounded: the browser may already be gone.
        let _ = tokio::time::timeout(Duration::from_secs(2), sc.stop()).await;
    }
    if let Some(s) = sidecar {
        s.shutdown().await;
    }
    let _ = tokio::time::timeout(Duration::from_secs(5), game.close()).await;
    Ok(summaries)
}

#[derive(Debug, Serialize)]
pub struct SpikeReport {
    pub iframe_origin: Option<String>,
    pub iframe_src: Option<String>,
    pub canvas: crate::browser::CanvasRect,
    pub frames: usize,
    pub fps: f64,
    pub delay_p50_ms: Option<f64>,
    pub delay_p95_ms: Option<f64>,
    /// Per key: mean frame change in the 500 ms after the press, divided by the change in the
    /// 500 ms before. Well above 1 means the runner reacted.
    pub key_response: Vec<(String, f64)>,
    pub keys_reach_game: bool,
}

fn frame_change(a: &image::RgbImage, b: &image::RgbImage) -> f64 {
    a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / a.as_raw().len().max(1) as f64
}

/// Mean frame-to-frame change over `dur`.
async fn change_window(source: &mut LatestFrame, dur: Duration) -> f64 {
    let start = Instant::now();
    let mut prev: Option<image::RgbImage> = None;
    let (mut sum, mut n) = (0.0, 0usize);
    while start.elapsed() < dur {
        let Ok(Ok(f)) = tokio::time::timeout(Duration::from_secs(2), source.next()).await else { break };
        if let Some(p) = &prev {
            sum += frame_change(p, &f.rgb);
            n += 1;
        }
        prev = Some(f.rgb);
    }
    sum / n.max(1) as f64
}

/// M0 spike: open the page, read the iframe origin, measure screencast fps and delay, and
/// send ← → ↑ ↓ to see whether the runner reacts. Start a run by hand when prompted.
pub async fn spike(cfg: &Config, measure_s: u64) -> Result<SpikeReport> {
    let mut game = Game::launch(&cfg.browser).await?;
    game.open(&cfg.browser).await?;
    game.focus().await?;
    let mut capture = start_capture(&game, cfg, CaptureBackend::Screencast).await?;

    let shots = Path::new(&cfg.recorder.runs_dir).join("spike");
    std::fs::create_dir_all(&shots)?;
    eprintln!("Measuring for {measure_s} s; clicking the canvas every 3 s to get past the menu. Snapshots in {}", shots.display());
    let t0 = Instant::now();
    let mut frames = 0usize;
    let mut delays = Vec::new();
    let mut next_click = 3.0;
    while t0.elapsed() < Duration::from_secs(measure_s) {
        let f = tokio::time::timeout(Duration::from_secs(5), capture.source.next()).await.context("no screencast frames")??;
        frames += 1;
        if let Some(d) = f.delay_ms {
            delays.push(d as f64);
        }
        let secs = t0.elapsed().as_secs_f64();
        if secs >= next_click {
            f.canvas.save(shots.join(format!("t{:02}.jpg", secs as u32)))?;
            game.click([0.5, 0.5]).await?;
            next_click += 3.0;
        }
    }
    let fps = frames as f64 / t0.elapsed().as_secs_f64();

    game.focus().await?;
    let mut key_response = Vec::new();
    for key in ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"] {
        if let Ok(Ok(f)) = tokio::time::timeout(Duration::from_secs(2), capture.source.next()).await {
            f.canvas.save(shots.join(format!("before_{key}.jpg")))?;
        }
        let before = change_window(&mut capture.source, Duration::from_millis(500)).await;
        game.press(key, Duration::from_millis(cfg.browser.key_hold_ms)).await?;
        tokio::time::sleep(Duration::from_millis(250)).await;
        if let Ok(Ok(f)) = tokio::time::timeout(Duration::from_secs(2), capture.source.next()).await {
            f.canvas.save(shots.join(format!("after_{key}.jpg")))?;
        }
        let after = change_window(&mut capture.source, Duration::from_millis(250)).await;
        key_response.push((key.to_string(), after / before.max(1e-6)));
        tokio::time::sleep(Duration::from_millis(700)).await;
    }
    // The first-run tutorial rewinds until the requested key is pressed at the right moment,
    // so cycle all four keys for a while and snapshot: progress past the tutorial means
    // keys arrive.
    let keys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];
    let cycle_start = Instant::now();
    let mut k = 0;
    let mut next_shot = 0.0;
    while cycle_start.elapsed() < Duration::from_secs(12) {
        game.press(keys[k % 4], Duration::from_millis(cfg.browser.key_hold_ms)).await?;
        k += 1;
        tokio::time::sleep(Duration::from_millis(400)).await;
        if cycle_start.elapsed().as_secs_f64() >= next_shot {
            if let Ok(Ok(f)) = tokio::time::timeout(Duration::from_secs(2), capture.source.next()).await {
                f.canvas.save(shots.join(format!("cycle_{:02}.jpg", next_shot as u32)))?;
            }
            next_shot += 2.0;
        }
    }
    let keys_reach_game = key_response.iter().filter(|(_, r)| *r > 1.3).count() >= 2;
    let report = SpikeReport {
        iframe_origin: game.iframe.as_ref().map(|i| i.origin.clone()),
        iframe_src: game.iframe.as_ref().map(|i| i.src.clone()),
        canvas: game.canvas,
        frames,
        fps,
        delay_p50_ms: (!delays.is_empty()).then(|| percentile(&mut delays.clone(), 0.5)),
        delay_p95_ms: (!delays.is_empty()).then(|| percentile(&mut delays, 0.95)),
        key_response,
        keys_reach_game,
    };
    if let Some(sc) = capture.screencast.take() {
        sc.stop().await;
    }
    game.close().await;
    Ok(report)
}

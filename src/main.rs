//! `ssbot` CLI (SPEC §7).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use ssbot::config::{CaptureBackend, Config};
use ssbot::label::cmd::LabelOpts;
use ssbot::label::{load_final_labels, run_name};
use ssbot::perception::fit::{self, LabelledFrame};
use ssbot::perception::zones::Calibration;
use ssbot::{bot, calibrate, frames, label, replay};

#[derive(Parser)]
#[command(name = "ssbot", about = "Subway Surfers bot: Rust perception and reflexes, OpenJev advisor")]
struct Cli {
    /// Settings file (SPEC §11).
    #[arg(long, global = true, default_value = "ssbot.toml")]
    config: PathBuf,
    /// Zones, markers and thresholds (SPEC §4.3).
    #[arg(long, global = true, default_value = "calibration.toml")]
    calibration: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play (or, with --human, record a human playing).
    Run {
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        no_advisor: bool,
        #[arg(long, value_enum)]
        capture: Option<CaptureBackend>,
        #[arg(long, default_value_t = 1)]
        runs: usize,
        /// Don't send any input; record while a human plays (M1).
        #[arg(long)]
        human: bool,
    },
    /// M0 spike: iframe origin, screencast fps/delay, whether keys reach the game.
    Spike {
        #[arg(long, default_value_t = 10)]
        seconds: u64,
    },
    /// Capture reference frames per screen, or preview zones on frames.
    Calibrate {
        /// Draw zones from calibration.toml onto frames in this directory instead.
        #[arg(long)]
        preview: Option<PathBuf>,
        #[arg(long, default_value = "calibration/preview")]
        out: PathBuf,
        /// Write the default (uncalibrated) calibration.toml and exit.
        #[arg(long)]
        init: bool,
    },
    /// Re-run perception and policy on a recorded run, no browser.
    Replay {
        run: PathBuf,
        #[arg(long)]
        no_advisor: bool,
    },
    /// ffmpeg → canvas-cropped frames at 10 fps, 640 px wide.
    Frames {
        video: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Canvas crop W:H:X:Y in source pixels; detected from black borders if omitted.
        #[arg(long)]
        crop: Option<frames::CropPx>,
        #[arg(long)]
        no_crop: bool,
        #[arg(long, default_value_t = 10)]
        fps: u32,
    },
    /// Label frames with Claude (Batches API by default).
    Label {
        frames_dir: PathBuf,
        #[arg(long, default_value = "claude-opus-5-5")]
        model: String,
        #[arg(long, default_value = "low")]
        effort: String,
        #[arg(long, default_value_t = 600)]
        max: usize,
        /// Send through POST /v1/messages instead of a batch (quick checks).
        #[arg(long)]
        sync: bool,
        /// Select frames, write overlays and estimate cost without calling the API.
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        #[arg(long, default_value = "prompts/label_guide.md")]
        guide: PathBuf,
        /// Label every n-th frame twice (10% by default).
        #[arg(long, default_value_t = 10)]
        repeat_every: usize,
    },
    /// Static HTML gallery for spot-checks and fixes.
    Review {
        frames_dir: PathBuf,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        /// Label by hand from scratch (no Claude labels needed).
        #[arg(long)]
        manual: bool,
    },
    /// Copy a varied subset of frames (no near-duplicates) into a new directory for labelling.
    Select {
        frames_dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 300)]
        max: usize,
    },
    /// Import bounding boxes from CVAT / Label Studio / makesense.ai / Roboflow as zone labels.
    Import {
        frames_dir: PathBuf,
        /// COCO JSON file, or a directory of YOLO .txt files.
        annotations: PathBuf,
        #[arg(long, value_enum)]
        format: Option<label::cmd::BoxFormat>,
        /// YOLO class names in id order, if there's no classes.txt / data.yaml.
        #[arg(long, value_delimiter = ',')]
        classes: Option<Vec<String>>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        /// Fraction of a zone a box must cover to label it (less, down to 5%, marks it unsure).
        #[arg(long, default_value_t = 0.4)]
        min_overlap: f32,
        #[arg(long)]
        force: bool,
    },
    /// Recording of a person playing → frames → selection → labels → zone classifier →
    /// OpenJev head. Rerunnable: each step reuses what earlier runs produced.
    TrainByHuman {
        /// The screen recording (.mov/.mp4) of you playing, game in fullscreen, or a run
        /// recorded with `ssbot run --human` (its `runs/<timestamp>` directory).
        video: PathBuf,
        /// Dataset name (defaults to the video's file name).
        #[arg(long)]
        name: Option<String>,
        /// How many frames to label.
        #[arg(long, default_value_t = 300)]
        max_frames: usize,
        /// auto (Claude if credentials exist, else the manual gallery), claude or manual.
        #[arg(long, default_value = "auto")]
        labeller: String,
        /// Use boxes exported from CVAT / Label Studio / makesense.ai (COCO JSON or YOLO dir).
        #[arg(long)]
        import: Option<PathBuf>,
        /// Synthetic situations for OpenJev's training set.
        #[arg(long, default_value_t = 2000)]
        situations: usize,
        /// Stop after the zone classifier.
        #[arg(long)]
        no_head: bool,
    },
    /// Train the learned zone classifier on labels; reports cross-validated accuracy.
    TrainZones {
        frames_dirs: Vec<PathBuf>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        #[arg(long, default_value = "zone_model.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 3000)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-3)]
        l2: f32,
        /// Save the model and point calibration.toml at it.
        #[arg(long)]
        write: bool,
    },
    /// Fit classifier thresholds to Claude's labels and report held-out accuracy (M2).
    Fit {
        frames_dirs: Vec<PathBuf>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        /// Save the fitted thresholds into calibration.toml.
        #[arg(long)]
        write: bool,
    },
    /// Score OpenJev option wordings on situations with a known best action (SPEC §9).
    AdvisorBench {
        #[arg(long)]
        model: Option<String>,
    },
    /// Write OpenJev decision-head training and test sets (JSON lines) to a directory.
    AdvisorData {
        #[arg(long, default_value = "data/advisor")]
        out: PathBuf,
        #[arg(long, default_value_t = 2000)]
        n: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Labelled frame directories whose running frames become the real test set.
        #[arg(long)]
        frames: Vec<PathBuf>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
    },
    /// Print final labels (Claude + human fixes) as JSON lines.
    LabelDump {
        frames_dir: PathBuf,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "ssbot=info".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let mut cfg = Config::load(&cli.config)?;
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);

    match cli.cmd {
        Cmd::Run { model, no_advisor, capture, runs, human } => {
            if let Some(m) = model {
                cfg.advisor.model = m;
            }
            let opts = bot::RunOpts {
                runs,
                advisor: !no_advisor,
                capture: capture.unwrap_or(cfg.capture.backend),
                human,
                calibration: cli.calibration.clone(),
            };
            let summaries = bot::run(&cfg, &opts).await?;
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        Cmd::Spike { seconds } => {
            let report = bot::spike(&cfg, seconds).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Calibrate { init: true, .. } => {
            if cli.calibration.exists() {
                bail!("{} exists; not overwriting", cli.calibration.display());
            }
            Calibration::default().save(&cli.calibration)?;
            eprintln!("wrote {}", cli.calibration.display());
        }
        Cmd::Calibrate { preview: Some(dir), out, .. } => {
            let written = calibrate::preview(&dir, &cli.calibration, &out, 20)?;
            eprintln!("wrote {} previews to {}", written.len(), out.display());
        }
        Cmd::Calibrate { preview: None, .. } => calibrate::capture(&cfg, &cli.calibration).await?,
        Cmd::Replay { run, no_advisor } => {
            let report = replay::replay(&run, &cfg, &cli.calibration, !no_advisor)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Frames { video, out, crop, no_crop, fps } => {
            let out = out.unwrap_or_else(|| frames::default_out_dir(&video));
            let crop = match (no_crop, crop) {
                (true, _) => None,
                (false, Some(c)) => Some(c),
                (false, None) => {
                    let c = frames::detect_crop(&video)?;
                    match c {
                        Some(c) => eprintln!("detected canvas crop {}:{}:{}:{}", c.w, c.h, c.x, c.y),
                        None => eprintln!("warning: couldn't detect the canvas; pass --crop W:H:X:Y"),
                    }
                    c
                }
            };
            let n = frames::extract(&video, &out, crop, fps, 640)?;
            eprintln!("{n} frames in {}", out.display());
        }
        Cmd::Label { frames_dir, model, effort, max, sync, dry_run, labels, guide, repeat_every } => {
            let opts = LabelOpts {
                frames_dir,
                model,
                effort,
                max,
                sync,
                dry_run,
                labels_dir: labels,
                calibration: cli.calibration.clone(),
                guide,
                repeat_every,
            };
            label::cmd::label(&opts).await?;
        }
        Cmd::Review { frames_dir, labels, manual } => {
            let out = label::cmd::review(&frames_dir, &labels, &cli.calibration, manual)?;
            eprintln!("open {}", out.display());
        }
        Cmd::Select { frames_dir, out, max } => {
            let n = label::cmd::select_frames(&frames_dir, &out, max, &cli.calibration)?;
            eprintln!("selected {n} frames into {}", out.display());
        }
        Cmd::Import { frames_dir, annotations, format, classes, labels, min_overlap, force } => {
            let (n, total) =
                label::cmd::import(&frames_dir, &annotations, format, classes, &labels, &cli.calibration, min_overlap, force)?;
            eprintln!("imported {n} labelled frames (of {total} in {}) into {}", frames_dir.display(), labels.join(format!("{}.jsonl", run_name(&frames_dir))).display());
            eprintln!("check them with `ssbot review {} --manual`, then `ssbot fit`", frames_dir.display());
        }
        Cmd::TrainZones { frames_dirs, labels, out, epochs, l2, write } => {
            let r = ssbot::train::train_zones(&cfg, &cli.calibration, &frames_dirs, &labels, &out, epochs, l2, write)?;
            for m in &r.facts_misses {
                eprintln!("  {m}");
            }
            println!("{}", serde_json::to_string_pretty(&r)?);
            eprintln!(
                "{}-fold accuracy {:.3} (near {:.3}); facts exact {}/{}, same move {}/{}; SPEC §9 zone targets {}",
                r.folds, r.accuracy, r.near, r.facts_exact.0, r.facts_exact.1, r.same_move.0, r.same_move.1,
                if r.accuracy >= 0.95 && r.near >= 0.98 { "met" } else { "NOT met" }
            );
            if write {
                eprintln!("saved {} and pointed {} at it", out.display(), cli.calibration.display());
            }
        }
        Cmd::TrainByHuman { video, name, max_frames, labeller, import, situations, no_head } => {
            let labeller = match (import, labeller.as_str()) {
                (Some(p), _) => ssbot::train::Labeller::Import(p),
                (None, "claude") => ssbot::train::Labeller::Claude,
                (None, "manual") => ssbot::train::Labeller::Manual,
                (None, "auto") => ssbot::train::Labeller::Auto,
                (None, other) => bail!("--labeller is auto, claude or manual, not {other}"),
            };
            let opts = ssbot::train::HumanOpts { video, name, max_frames, labeller, situations, train_head: !no_head };
            ssbot::train::train_by_human(&cfg, &cli.calibration, &opts).await?;
        }
        Cmd::Fit { frames_dirs, labels, write } => {
            if frames_dirs.is_empty() {
                bail!("give at least one labelled frames directory");
            }
            let mut calib = Calibration::load_or_default(&cli.calibration);
            let base = cli.calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
            let frames = ssbot::train::load_labelled(&frames_dirs, &labels, work)?;
            let (test, train): (Vec<&LabelledFrame>, Vec<&LabelledFrame>) = frames.iter().partition(|f| fit::is_test(f.id));
            eprintln!("{} labelled frames: {} train, {} held out", frames.len(), train.len(), test.len());

            // Background thresholds change the features, so they're searched in an outer loop.
            let mut best = (calib.thresholds, f64::MIN);
            for bg in [0.2f32, 0.25, 0.3, 0.35, 0.4] {
                let mut c = calib.clone();
                c.thresholds.bg_sat_max = bg;
                let samples = fit::zone_samples(&train, &c, work);
                let (th, score) = fit::fit_classifier(&samples, c.thresholds);
                if score > best.1 {
                    best = (th, score);
                }
            }
            calib.thresholds = best.0;
            let mut acc = fit::zone_accuracy(&fit::zone_samples(&test, &calib, work), &calib.thresholds);
            let (state, lane) = fit::sequence_accuracy(&test, &calib, &base, work);
            acc.game_state = Some(state);
            acc.player_lane = lane;
            println!("{}", serde_json::to_string_pretty(&acc)?);
            eprintln!("SPEC §9 targets {}", if acc.meets_targets() { "met" } else { "NOT met" });
            if write {
                calib.save(&cli.calibration)?;
                eprintln!("saved thresholds to {}", cli.calibration.display());
            }
        }
        Cmd::AdvisorBench { model } => {
            if let Some(m) = model {
                cfg.advisor.model = m;
            }
            let (sidecar, mut link) = ssbot::sidecar::Sidecar::spawn(&cfg.advisor).await?;
            eprintln!("{} · {} cases per wording", sidecar.model, ssbot::policy::bench::cases().len());
            let scores = ssbot::policy::bench::run(&mut link, &ssbot::policy::advisor::Wording::ALL).await?;
            println!("{}", serde_json::to_string_pretty(&scores)?);
            drop(link);
            sidecar.shutdown().await;
        }
        Cmd::AdvisorData { out, n, seed, frames, labels } => {
            let mut labelled = Vec::new();
            for (d, dir) in frames.iter().enumerate() {
                for (id, l) in load_final_labels(&labels, &run_name(dir))? {
                    labelled.push((d as u64 * 10_000_000 + id, l));
                }
            }
            labelled.sort_by_key(|(id, _)| *id);
            let ds = ssbot::policy::dataset::build(n, seed, cfg.advisor.wording, &labelled);
            std::fs::create_dir_all(&out)?;
            for (name, recs) in [("train", &ds.train), ("test_bench", &ds.test_bench), ("test_real", &ds.test_real), ("test_offscreen", &ds.test_offscreen)] {
                let text: String = recs.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
                std::fs::write(out.join(format!("{name}.jsonl")), text)?;
                eprintln!("{name}: {} records", recs.len());
            }
        }
        Cmd::LabelDump { frames_dir, labels } => {
            let run = run_name(&frames_dir);
            let mut all: Vec<_> = load_final_labels(&labels, &run)?.into_iter().collect();
            all.sort_by_key(|(id, _)| *id);
            for (id, l) in all {
                println!("{}", serde_json::json!({"run": run, "frame_id": id, "label": l}));
            }
        }
    }
    Ok(())
}

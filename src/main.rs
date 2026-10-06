//! `ssbot` CLI (SPEC §7).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
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
        /// Drive with an imitation model trained by `sidecar/il_cnn.py`.
        #[arg(long)]
        il: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ssbot::policy::arbiter::IlMode::Guarded)]
        il_mode: ssbot::policy::arbiter::IlMode,
        /// Least probability for the model's action to be used.
        #[arg(long, default_value_t = 0.5)]
        il_threshold: f32,
    },
    /// Play N runs and print median survival and score, to compare two configurations. Saves
    /// `runs/bench_<tag>.json`; compare two with `ssbot bench-compare a.json b.json`.
    Bench {
        #[arg(long, default_value_t = 20)]
        runs: usize,
        /// Name for this configuration (used in the output file name).
        #[arg(long, default_value = "bench")]
        tag: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        no_advisor: bool,
        #[arg(long, value_enum)]
        capture: Option<CaptureBackend>,
        /// Drive with an imitation model (see `run --il`).
        #[arg(long)]
        il: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ssbot::policy::arbiter::IlMode::Guarded)]
        il_mode: ssbot::policy::arbiter::IlMode,
        #[arg(long, default_value_t = 0.5)]
        il_threshold: f32,
    },
    /// Copy frames from just before each run ended (from run folders, or a saved bench report)
    /// into one frames folder, ready for `ssbot label` / manual labelling.
    CrashFrames {
        /// Run folders, or a `runs/bench_<tag>.json`.
        sources: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        per_run: usize,
        #[arg(long, default_value_t = 2000.0)]
        window_ms: f64,
    },
    /// Print two saved bench reports side by side.
    BenchCompare {
        a: PathBuf,
        b: PathBuf,
    },
    /// Recompute `summary.json` (crash, survival, HUD score) for recorded runs.
    Summarize {
        runs: Vec<PathBuf>,
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
    /// Score a saved zone model on labelled frames (confusion matrix, missed hazards, false alarms).
    EvalZones {
        frames_dirs: Vec<PathBuf>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        /// Model to score; defaults to the one calibration.toml points at.
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Frames from human-recorded runs with the action pressed shortly after each, for
    /// `sidecar/il_cnn.py` (imitation learning).
    IlData {
        runs: Vec<PathBuf>,
        #[arg(long, default_value = "data/il")]
        out: PathBuf,
        /// A key press labels the frames up to this many ms before it.
        #[arg(long, default_value_t = 350.0)]
        lead_ms: f64,
        /// Keep every n-th saved frame.
        #[arg(long, default_value_t = 2)]
        stride: usize,
        /// Also count the bot's own actions as labels (to test the pipeline).
        #[arg(long)]
        any_source: bool,
    },
    /// Score an imitation model on data built by `il-data` (accuracy and time per frame).
    IlEval {
        model: PathBuf,
        #[arg(long, default_value = "data/il")]
        data: PathBuf,
    },
    /// Draw what the bot's perception reports on a run's saved frames (a grid image), zones
    /// coloured by predicted class: F free, T train, R ramp, L low barrier, H high barrier.
    See {
        frames_dir: PathBuf,
        #[arg(long, default_value = "see.jpg")]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        n: usize,
        /// Start this fraction of the way into the run's running frames.
        #[arg(long)]
        from: Option<f64>,
    },
    /// Copy frames where the eyes report a barrier (from run folders or a bench report) into a
    /// folder for labelling: barriers are the scarce class.
    Mine {
        /// Run folders, or a `runs/bench_<tag>.json`.
        sources: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 6)]
        per_run: usize,
        /// Least time between two picked frames of one run.
        #[arg(long, default_value_t = 400.0)]
        gap_ms: f64,
    },
    /// Write labelled zone crops (the CNN's training data) for `sidecar/zone_cnn.py`.
    ZoneCrops {
        frames_dirs: Vec<PathBuf>,
        #[arg(long, default_value = "labels")]
        labels: PathBuf,
        #[arg(long, default_value = "data/zone_crops")]
        out: PathBuf,
        /// Cut crops from the frame at its recorded resolution instead of the 320×180 work image.
        #[arg(long)]
        native: bool,
        /// Crop side in pixels (a multiple of 8).
        #[arg(long, default_value_t = 32)]
        crop: usize,
        /// Grow each zone's box by this fraction on every side and keep the surroundings.
        #[arg(long, default_value_t = 0.0)]
        ctx: f32,
        /// Stack the masked zone crop and the context crop as two views (six channels).
        #[arg(long)]
        dual: bool,
        /// Context for far zones, if different from --ctx.
        #[arg(long)]
        ctx_far: Option<f32>,
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
        Cmd::Run { model, no_advisor, capture, runs, human, il, il_mode, il_threshold } => {
            if let Some(m) = model {
                cfg.advisor.model = m;
            }
            let opts = bot::RunOpts {
                runs,
                advisor: !no_advisor,
                capture: capture.unwrap_or(cfg.capture.backend),
                human,
                calibration: cli.calibration.clone(),
                il: il.map(|p| (p, il_mode, il_threshold)),
            };
            let summaries = bot::run(&cfg, &opts).await?;
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        Cmd::Bench { runs, tag, model, no_advisor, capture, il, il_mode, il_threshold } => {
            if let Some(m) = model {
                cfg.advisor.model = m;
            }
            let opts = bot::RunOpts {
                runs,
                advisor: !no_advisor,
                capture: capture.unwrap_or(cfg.capture.backend),
                human: false,
                calibration: cli.calibration.clone(),
                il: il.map(|p| (p, il_mode, il_threshold)),
            };
            let summaries = bot::run(&cfg, &opts).await?;
            let report = ssbot::bench::report(&tag, !no_advisor, &summaries);
            let path = Path::new(&cfg.recorder.runs_dir).join(format!("bench_{tag}.json"));
            std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
            println!("{report}\nsaved {}", path.display());
        }
        Cmd::CrashFrames { sources, out, per_run, window_ms } => {
            let mut dirs: Vec<PathBuf> = Vec::new();
            for s in sources {
                if s.extension().is_some_and(|e| e == "json") {
                    let r: ssbot::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&s)?)?;
                    dirs.extend(r.run_dirs.into_iter().map(PathBuf::from));
                } else {
                    dirs.push(s);
                }
            }
            std::fs::create_dir_all(&out)?;
            let mut copied = 0;
            for (k, dir) in dirs.iter().enumerate() {
                let events = ssbot::recorder::read_events(dir)?;
                let ids = ssbot::recorder::crash_window(dir, &events, window_ms, per_run);
                // Frame ids restart every run, so give each run its own block of ids.
                for id in &ids {
                    let name = format!("{}.jpg", (k as u64 + 1) * 10_000_000 + id);
                    std::fs::copy(dir.join("frames").join(format!("{id}.jpg")), out.join(name))?;
                }
                copied += ids.len();
                eprintln!("{}: {} frames", dir.display(), ids.len());
            }
            println!("{copied} frames from {} runs in {}", dirs.len(), out.display());
        }
        Cmd::BenchCompare { a, b } => {
            for p in [a, b] {
                let r: ssbot::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
                println!("{r}");
            }
        }
        Cmd::Summarize { runs } => {
            for dir in runs {
                let events = ssbot::recorder::read_events(&dir)?;
                let mut summary = ssbot::recorder::summarise(&events, None);
                summary.score = ssbot::recorder::run_score(&dir, &events);
                // The advisor's counters aren't in the events, so keep the ones already saved.
                if let Ok(old) = std::fs::read_to_string(dir.join("summary.json"))
                    && let Ok(old) = serde_json::from_str::<ssbot::recorder::Summary>(&old)
                {
                    summary.advisor = old.advisor;
                    summary.advisor_freshness_rate = old.advisor_freshness_rate;
                    summary.advisor_cache_hit_rate = old.advisor_cache_hit_rate;
                }
                std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
                println!(
                    "{}: survived {:.1} s, crashed {}, score {}",
                    dir.display(),
                    summary.survival_s,
                    summary.crashed,
                    summary.score.map_or("?".into(), |s| s.to_string())
                );
            }
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
        Cmd::IlData { runs, out, lead_ms, stride, any_source } => {
            let opts = ssbot::il::IlOpts { lead_ms, stride, size: (128, 72), human_only: !any_source };
            let (n, counts) = ssbot::il::build(&runs, &out, &opts)?;
            println!("{n} frames in {} (stay {}, left {}, right {}, jump {}, roll {})", out.display(), counts[0], counts[1], counts[2], counts[3], counts[4]);
        }
        Cmd::Mine { sources, out, per_run, gap_ms } => {
            let mut dirs: Vec<PathBuf> = Vec::new();
            for s in sources {
                if s.extension().is_some_and(|e| e == "json") {
                    let r: ssbot::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&s)?)?;
                    dirs.extend(r.run_dirs.into_iter().map(PathBuf::from));
                } else {
                    dirs.push(s);
                }
            }
            let n = ssbot::see::mine_barriers(&cfg, &cli.calibration, &dirs, &out, per_run, gap_ms)?;
            println!("{n} barrier frames from {} runs in {}", dirs.len(), out.display());
        }
        Cmd::See { frames_dir, out, n, from } => {
            let k = ssbot::see::sheet(&cfg, &cli.calibration, &frames_dir, &out, n, from)?;
            println!("{k} frames drawn in {}", out.display());
        }
        Cmd::IlEval { model, data } => {
            let net = ssbot::il::IlNet::load(&model)?;
            let raw = std::fs::read(data.join("frames.bin"))?;
            let labels: Vec<usize> = std::fs::read_to_string(data.join("meta.jsonl"))?
                .lines()
                .map(|l| serde_json::from_str::<serde_json::Value>(l).map(|v| v["label"].as_u64().unwrap_or(0) as usize))
                .collect::<Result<_, _>>()?;
            let (w, h) = (net.width, net.height);
            let size = (w * h * 3) as usize;
            let started = std::time::Instant::now();
            let mut hits = 0;
            for (i, label) in labels.iter().enumerate() {
                let img = image::RgbImage::from_raw(w, h, raw[i * size..(i + 1) * size].to_vec()).context("frame size")?;
                hits += (ssbot::il::class_of(net.best(&img).0) == Some(*label)) as usize;
            }
            println!(
                "{} frames: accuracy {:.3}, {:.1} ms per frame",
                labels.len(),
                hits as f64 / labels.len() as f64,
                started.elapsed().as_secs_f64() * 1000.0 / labels.len() as f64
            );
        }
        Cmd::ZoneCrops { frames_dirs, labels, out, native, crop, ctx, dual, ctx_far } => {
            let n = ssbot::train::dump_zone_crops(&cfg, &cli.calibration, &frames_dirs, &labels, &out, native, crop, ctx, dual, ctx_far)?;
            println!("{n} zone crops in {}", out.display());
        }
        Cmd::EvalZones { frames_dirs, labels, model } => {
            let calib = Calibration::load_or_default(&cli.calibration);
            let base = cli.calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
            let model = match (model, calib.zone_model) {
                (Some(m), _) => m,
                (None, Some(m)) => base.join(m),
                (None, None) => bail!("calibration has no zone_model; pass --model"),
            };
            let r = ssbot::train::eval_zones(&cfg, &cli.calibration, &model, &frames_dirs, &labels)?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            println!(
                "{} sure zones on {} frames: accuracy {:.3}; missed hazards {}/{} ({:.1}%); false alarms {}/{} ({:.1}%); barrier recall {}/{} ({:.1}%)",
                r.zones,
                r.frames,
                r.accuracy,
                r.missed_hazards.0,
                r.missed_hazards.1,
                100.0 * r.missed_hazards.0 as f64 / r.missed_hazards.1.max(1) as f64,
                r.false_alarms.0,
                r.false_alarms.1,
                100.0 * r.false_alarms.0 as f64 / r.false_alarms.1.max(1) as f64,
                r.barrier_recall.0,
                r.barrier_recall.1,
                100.0 * r.barrier_recall.0 as f64 / r.barrier_recall.1.max(1) as f64,
            );
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

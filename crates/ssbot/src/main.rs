//! `ssbot` CLI (SPEC §7).

mod cli;
mod cmd;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

use cli::{Cli, Cmd};
use cmd::{advisor, label, play};
use ssbot_core::config::Config;
use ssbot_core::perception::zones::Calibration;
use ssbot_tools::fit::{self, LabelledFrame};
use ssbot_tools::{calibrate, frames, replay};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "ssbot=info,ssbot_core=info,ssbot_live=info,ssbot_tools=info".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let mut cfg = Config::load(&cli.config)?;
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);

    match cli.cmd {
        Cmd::Run(args) => play::run(&mut cfg, &cli.calibration, args).await?,
        Cmd::Bench(args) => play::bench(&mut cfg, &cli.calibration, args).await?,
        Cmd::CrashFrames(args) => play::crash_frames(args)?,
        Cmd::BenchCompare(args) => play::bench_compare(args)?,
        Cmd::Summarize(args) => play::summarize(args)?,
        Cmd::Spike(args) => play::spike(&cfg, args).await?,
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
        Cmd::Label(args) => label::label(&cli.calibration, args).await?,
        Cmd::Review(args) => label::review(&cli.calibration, args)?,
        Cmd::Select(args) => label::select(&cli.calibration, args)?,
        Cmd::Import(args) => label::import(&cli.calibration, args)?,
        Cmd::IlData { runs, out, lead_ms, stride, any_source } => {
            let opts = ssbot_tools::il_data::IlOpts { lead_ms, stride, size: (128, 72), human_only: !any_source };
            let (n, counts) = ssbot_tools::il_data::build(&runs, &out, &opts)?;
            println!("{n} frames in {} (stay {}, left {}, right {}, jump {}, roll {})", out.display(), counts[0], counts[1], counts[2], counts[3], counts[4]);
        }
        Cmd::Mine { sources, out, per_run, gap_ms } => {
            let mut dirs: Vec<PathBuf> = Vec::new();
            for s in sources {
                if s.extension().is_some_and(|e| e == "json") {
                    let r: ssbot_tools::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&s)?)?;
                    dirs.extend(r.run_dirs.into_iter().map(PathBuf::from));
                } else {
                    dirs.push(s);
                }
            }
            let n = ssbot_tools::see::mine_barriers(&cfg, &cli.calibration, &dirs, &out, per_run, gap_ms)?;
            println!("{n} barrier frames from {} runs in {}", dirs.len(), out.display());
        }
        Cmd::See { frames_dir, out, n, from } => {
            let k = ssbot_tools::see::sheet(&cfg, &cli.calibration, &frames_dir, &out, n, from)?;
            println!("{k} frames drawn in {}", out.display());
        }
        Cmd::IlEval { model, data } => {
            let net = ssbot_core::il::IlNet::load(&model)?;
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
                hits += (ssbot_core::il::class_of(net.best(&img).0) == Some(*label)) as usize;
            }
            println!(
                "{} frames: accuracy {:.3}, {:.1} ms per frame",
                labels.len(),
                hits as f64 / labels.len() as f64,
                started.elapsed().as_secs_f64() * 1000.0 / labels.len() as f64
            );
        }
        Cmd::ZoneCrops { frames_dirs, labels, out, native, crop, ctx, dual, ctx_far } => {
            let opts = ssbot_tools::train::ZoneCropsOpts { frames_dirs: &frames_dirs, labels_dir: &labels, out: &out, native, crop, ctx, dual, ctx_far };
            let n = ssbot_tools::train::dump_zone_crops(&cfg, &cli.calibration, opts)?;
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
            let r = ssbot_tools::train::eval_zones(&cfg, &cli.calibration, &model, &frames_dirs, &labels)?;
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
            let opts = ssbot_tools::train::TrainZonesOpts { frames_dirs: &frames_dirs, labels_dir: &labels, out: &out, epochs, l2, write };
            let r = ssbot_tools::train::train_zones(&cfg, &cli.calibration, opts)?;
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
                (Some(p), _) => ssbot_tools::train::Labeller::Import(p),
                (None, "claude") => ssbot_tools::train::Labeller::Claude,
                (None, "manual") => ssbot_tools::train::Labeller::Manual,
                (None, "auto") => ssbot_tools::train::Labeller::Auto,
                (None, other) => bail!("--labeller is auto, claude or manual, not {other}"),
            };
            let opts = ssbot_tools::train::HumanOpts { video, name, max_frames, labeller, situations, train_head: !no_head };
            ssbot_tools::train::train_by_human(&cfg, &cli.calibration, &opts).await?;
        }
        Cmd::Fit { frames_dirs, labels, write } => {
            if frames_dirs.is_empty() {
                bail!("give at least one labelled frames directory");
            }
            let mut calib = Calibration::load_or_default(&cli.calibration);
            let base = cli.calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
            let frames = ssbot_tools::train::load_labelled(&frames_dirs, &labels, work)?;
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
        Cmd::AdvisorBench(args) => advisor::advisor_bench(&mut cfg, args).await?,
        Cmd::AdvisorData(args) => advisor::advisor_data(&cfg, args)?,
        Cmd::LabelDump(args) => label::label_dump(args)?,
    }
    Ok(())
}

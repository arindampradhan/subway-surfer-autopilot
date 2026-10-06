//! `ssbot` CLI (SPEC §7).

mod cli;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

use cli::{Cli, Cmd};
use ssbot_core::config::Config;
use ssbot_core::perception::zones::Calibration;
use ssbot_live::bot;
use ssbot_tools::fit::{self, LabelledFrame};
use ssbot_tools::label::cmd::LabelOpts;
use ssbot_tools::label::{load_final_labels, run_name};
use ssbot_tools::{calibrate, frames, label, replay};

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
            let report = ssbot_tools::bench::report(&tag, !no_advisor, &summaries);
            let path = Path::new(&cfg.recorder.runs_dir).join(format!("bench_{tag}.json"));
            std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
            println!("{report}\nsaved {}", path.display());
        }
        Cmd::CrashFrames { sources, out, per_run, window_ms } => {
            let mut dirs: Vec<PathBuf> = Vec::new();
            for s in sources {
                if s.extension().is_some_and(|e| e == "json") {
                    let r: ssbot_tools::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&s)?)?;
                    dirs.extend(r.run_dirs.into_iter().map(PathBuf::from));
                } else {
                    dirs.push(s);
                }
            }
            std::fs::create_dir_all(&out)?;
            let mut copied = 0;
            for (k, dir) in dirs.iter().enumerate() {
                let events = ssbot_live::recorder::read_events(dir)?;
                let ids = ssbot_live::recorder::crash_window(dir, &events, window_ms, per_run);
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
                let r: ssbot_tools::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
                println!("{r}");
            }
        }
        Cmd::Summarize { runs } => {
            for dir in runs {
                let events = ssbot_live::recorder::read_events(&dir)?;
                let mut summary = ssbot_live::recorder::summarise(&events, None);
                summary.score = ssbot_live::recorder::run_score(&dir, &events);
                // The advisor's counters aren't in the events, so keep the ones already saved.
                if let Ok(old) = std::fs::read_to_string(dir.join("summary.json"))
                    && let Ok(old) = serde_json::from_str::<ssbot_live::recorder::Summary>(&old)
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
            let (n, total) = label::cmd::import(label::cmd::ImportOpts {
                frames_dir: &frames_dir,
                annotations: &annotations,
                format,
                classes,
                labels_dir: &labels,
                calibration: &cli.calibration,
                min_overlap,
                force,
            })?;
            eprintln!("imported {n} labelled frames (of {total} in {}) into {}", frames_dir.display(), labels.join(format!("{}.jsonl", run_name(&frames_dir))).display());
            eprintln!("check them with `ssbot review {} --manual`, then `ssbot fit`", frames_dir.display());
        }
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
        Cmd::AdvisorBench { model } => {
            if let Some(m) = model {
                cfg.advisor.model = m;
            }
            let (sidecar, mut link) = ssbot_live::sidecar::Sidecar::spawn(&cfg.advisor).await?;
            eprintln!("{} · {} cases per wording", sidecar.model, ssbot_tools::advisor_bench::cases().len());
            let scores = ssbot_tools::advisor_bench::run(&mut link, &ssbot_core::config::Wording::ALL).await?;
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
            let ds = ssbot_tools::advisor_data::build(n, seed, cfg.advisor.wording, &labelled);
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

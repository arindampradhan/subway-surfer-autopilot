//! Training and scoring models: `train-by-human`, `train-zones`, `eval-zones`, `zone-crops`, `fit`,
//! `il-data`, `il-eval`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;

use ssbot_engine::config::Config;
use ssbot_engine::perception::zones::Calibration;
use ssbot_lab::fit::{self, LabelledFrame};

#[derive(Args)]
pub struct TrainByHumanArgs {
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
}

#[derive(Args)]
pub struct TrainZonesArgs {
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
}

#[derive(Args)]
pub struct EvalZonesArgs {
    frames_dirs: Vec<PathBuf>,
    #[arg(long, default_value = "labels")]
    labels: PathBuf,
    /// Model to score; defaults to the one calibration.toml points at.
    #[arg(long)]
    model: Option<PathBuf>,
}

#[derive(Args)]
pub struct IlDataArgs {
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
}

#[derive(Args)]
pub struct IlEvalArgs {
    model: PathBuf,
    #[arg(long, default_value = "data/il")]
    data: PathBuf,
}

#[derive(Args)]
pub struct ZoneCropsArgs {
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
}

#[derive(Args)]
pub struct FitArgs {
    frames_dirs: Vec<PathBuf>,
    #[arg(long, default_value = "labels")]
    labels: PathBuf,
    /// Save the fitted thresholds into calibration.toml.
    #[arg(long)]
    write: bool,
}

pub async fn train_by_human(cfg: &Config, calibration: &Path, args: TrainByHumanArgs) -> Result<()> {
    let TrainByHumanArgs { video, name, max_frames, labeller, import, situations, no_head } = args;
    let labeller = match (import, labeller.as_str()) {
        (Some(p), _) => ssbot_lab::train::Labeller::Import(p),
        (None, "claude") => ssbot_lab::train::Labeller::Claude,
        (None, "manual") => ssbot_lab::train::Labeller::Manual,
        (None, "auto") => ssbot_lab::train::Labeller::Auto,
        (None, other) => bail!("--labeller is auto, claude or manual, not {other}"),
    };
    let opts = ssbot_lab::train::HumanOpts { video, name, max_frames, labeller, situations, train_head: !no_head };
    ssbot_lab::train::train_by_human(cfg, calibration, &opts).await?;
    Ok(())
}

pub fn train_zones(cfg: &Config, calibration: &Path, args: TrainZonesArgs) -> Result<()> {
    let TrainZonesArgs { frames_dirs, labels, out, epochs, l2, write } = args;
    let opts = ssbot_lab::train::TrainZonesOpts { frames_dirs: &frames_dirs, labels_dir: &labels, out: &out, epochs, l2, write };
    let r = ssbot_lab::train::train_zones(cfg, calibration, opts)?;
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
        eprintln!("saved {} and pointed {} at it", out.display(), calibration.display());
    }
    Ok(())
}

pub fn eval_zones(cfg: &Config, calibration: &Path, args: EvalZonesArgs) -> Result<()> {
    let EvalZonesArgs { frames_dirs, labels, model } = args;
    let calib = Calibration::load_or_default(calibration);
    let base = calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
    let model = match (model, calib.zone_model) {
        (Some(m), _) => m,
        (None, Some(m)) => base.join(m),
        (None, None) => bail!("calibration has no zone_model; pass --model"),
    };
    let r = ssbot_lab::train::eval_zones(cfg, calibration, &model, &frames_dirs, &labels)?;
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
    Ok(())
}

pub fn il_data(args: IlDataArgs) -> Result<()> {
    let IlDataArgs { runs, out, lead_ms, stride, any_source } = args;
    let opts = ssbot_lab::il_data::IlOpts { lead_ms, stride, size: (128, 72), human_only: !any_source };
    let (n, counts) = ssbot_lab::il_data::build(&runs, &out, &opts)?;
    println!("{n} frames in {} (stay {}, left {}, right {}, jump {}, roll {})", out.display(), counts[0], counts[1], counts[2], counts[3], counts[4]);
    Ok(())
}

pub fn il_eval(args: IlEvalArgs) -> Result<()> {
    let IlEvalArgs { model, data } = args;
    let net = ssbot_engine::il::IlNet::load(&model)?;
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
        hits += (ssbot_engine::il::class_of(net.best(&img).0) == Some(*label)) as usize;
    }
    println!(
        "{} frames: accuracy {:.3}, {:.1} ms per frame",
        labels.len(),
        hits as f64 / labels.len() as f64,
        started.elapsed().as_secs_f64() * 1000.0 / labels.len() as f64
    );
    Ok(())
}

pub fn zone_crops(cfg: &Config, calibration: &Path, args: ZoneCropsArgs) -> Result<()> {
    let ZoneCropsArgs { frames_dirs, labels, out, native, crop, ctx, dual, ctx_far } = args;
    let opts = ssbot_lab::train::ZoneCropsOpts { frames_dirs: &frames_dirs, labels_dir: &labels, out: &out, native, crop, ctx, dual, ctx_far };
    let n = ssbot_lab::train::dump_zone_crops(cfg, calibration, opts)?;
    println!("{n} zone crops in {}", out.display());
    Ok(())
}

pub fn fit(cfg: &Config, calibration: &Path, args: FitArgs) -> Result<()> {
    let FitArgs { frames_dirs, labels, write } = args;
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    if frames_dirs.is_empty() {
        bail!("give at least one labelled frames directory");
    }
    let mut calib = Calibration::load_or_default(calibration);
    let base = calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
    let frames = ssbot_lab::train::load_labelled(&frames_dirs, &labels, work)?;
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
        calib.save(calibration)?;
        eprintln!("saved thresholds to {}", calibration.display());
    }
    Ok(())
}

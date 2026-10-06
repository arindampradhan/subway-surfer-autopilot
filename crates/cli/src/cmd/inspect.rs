//! Calibration and looking at frames: `calibrate`, `replay`, `frames`, `see`, `mine`.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Args;

use ssbot_engine::config::Config;
use ssbot_engine::perception::zones::Calibration;
use ssbot_lab::{calibrate, frames, replay};

#[derive(Args)]
pub struct CalibrateArgs {
    /// Draw zones from calibration.toml onto frames in this directory instead.
    #[arg(long)]
    preview: Option<PathBuf>,
    #[arg(long, default_value = "calibration/preview")]
    out: PathBuf,
    /// Write the default (uncalibrated) calibration.toml and exit.
    #[arg(long)]
    init: bool,
}

#[derive(Args)]
pub struct ReplayArgs {
    run: PathBuf,
    #[arg(long)]
    no_advisor: bool,
}

#[derive(Args)]
pub struct FramesArgs {
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
}

#[derive(Args)]
pub struct SeeArgs {
    frames_dir: PathBuf,
    #[arg(long, default_value = "see.jpg")]
    out: PathBuf,
    #[arg(long, default_value_t = 8)]
    n: usize,
    /// Start this fraction of the way into the run's running frames.
    #[arg(long)]
    from: Option<f64>,
}

#[derive(Args)]
pub struct MineArgs {
    /// Run folders, or a `runs/bench_<tag>.json`.
    sources: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 6)]
    per_run: usize,
    /// Least time between two picked frames of one run.
    #[arg(long, default_value_t = 400.0)]
    gap_ms: f64,
}

pub async fn calibrate(cfg: &Config, calibration: &Path, args: CalibrateArgs) -> Result<()> {
    match args {
        CalibrateArgs { init: true, .. } => {
            if calibration.exists() {
                bail!("{} exists; not overwriting", calibration.display());
            }
            Calibration::default().save(calibration)?;
            eprintln!("wrote {}", calibration.display());
        }
        CalibrateArgs { preview: Some(dir), out, .. } => {
            let written = calibrate::preview(&dir, calibration, &out, 20)?;
            eprintln!("wrote {} previews to {}", written.len(), out.display());
        }
        CalibrateArgs { preview: None, .. } => calibrate::capture(cfg, calibration).await?,
    }
    Ok(())
}

pub fn replay(cfg: &Config, calibration: &Path, args: ReplayArgs) -> Result<()> {
    let ReplayArgs { run, no_advisor } = args;
    let report = replay::replay(&run, cfg, calibration, !no_advisor)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn frames(args: FramesArgs) -> Result<()> {
    let FramesArgs { video, out, crop, no_crop, fps } = args;
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
    Ok(())
}

pub fn see(cfg: &Config, calibration: &Path, args: SeeArgs) -> Result<()> {
    let SeeArgs { frames_dir, out, n, from } = args;
    let k = ssbot_lab::see::sheet(cfg, calibration, &frames_dir, &out, n, from)?;
    println!("{k} frames drawn in {}", out.display());
    Ok(())
}

pub fn mine(cfg: &Config, calibration: &Path, args: MineArgs) -> Result<()> {
    let MineArgs { sources, out, per_run, gap_ms } = args;
    let mut dirs: Vec<PathBuf> = Vec::new();
    for s in sources {
        if s.extension().is_some_and(|e| e == "json") {
            let r: ssbot_lab::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&s)?)?;
            dirs.extend(r.run_dirs.into_iter().map(PathBuf::from));
        } else {
            dirs.push(s);
        }
    }
    let n = ssbot_lab::see::mine_barriers(cfg, calibration, &dirs, &out, per_run, gap_ms)?;
    println!("{n} barrier frames from {} runs in {}", dirs.len(), out.display());
    Ok(())
}

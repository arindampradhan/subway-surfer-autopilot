//! Playing and its results: `run`, `bench`, `crash-frames`, `bench-compare`, `summarize`, `spike`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;

use ssbot_core::config::{CaptureBackend, Config};
use ssbot_live::bot;

#[derive(Args)]
pub struct RunArgs {
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
    #[arg(long, value_enum, default_value_t = ssbot_core::policy::arbiter::IlMode::Guarded)]
    il_mode: ssbot_core::policy::arbiter::IlMode,
    /// Least probability for the model's action to be used.
    #[arg(long, default_value_t = 0.5)]
    il_threshold: f32,
}

#[derive(Args)]
pub struct BenchArgs {
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
    #[arg(long, value_enum, default_value_t = ssbot_core::policy::arbiter::IlMode::Guarded)]
    il_mode: ssbot_core::policy::arbiter::IlMode,
    #[arg(long, default_value_t = 0.5)]
    il_threshold: f32,
}

#[derive(Args)]
pub struct CrashFramesArgs {
    /// Run folders, or a `runs/bench_<tag>.json`.
    sources: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 8)]
    per_run: usize,
    #[arg(long, default_value_t = 2000.0)]
    window_ms: f64,
}

#[derive(Args)]
pub struct BenchCompareArgs {
    a: PathBuf,
    b: PathBuf,
}

#[derive(Args)]
pub struct SummarizeArgs {
    runs: Vec<PathBuf>,
}

#[derive(Args)]
pub struct SpikeArgs {
    #[arg(long, default_value_t = 10)]
    seconds: u64,
}

pub async fn run(cfg: &mut Config, calibration: &Path, args: RunArgs) -> Result<()> {
    let RunArgs { model, no_advisor, capture, runs, human, il, il_mode, il_threshold } = args;
    if let Some(m) = model {
        cfg.advisor.model = m;
    }
    let opts = bot::RunOpts {
        runs,
        advisor: !no_advisor,
        capture: capture.unwrap_or(cfg.capture.backend),
        human,
        calibration: calibration.to_path_buf(),
        il: il.map(|p| (p, il_mode, il_threshold)),
    };
    let summaries = bot::run(cfg, &opts).await?;
    println!("{}", serde_json::to_string_pretty(&summaries)?);
    Ok(())
}

pub async fn bench(cfg: &mut Config, calibration: &Path, args: BenchArgs) -> Result<()> {
    let BenchArgs { runs, tag, model, no_advisor, capture, il, il_mode, il_threshold } = args;
    if let Some(m) = model {
        cfg.advisor.model = m;
    }
    let opts = bot::RunOpts {
        runs,
        advisor: !no_advisor,
        capture: capture.unwrap_or(cfg.capture.backend),
        human: false,
        calibration: calibration.to_path_buf(),
        il: il.map(|p| (p, il_mode, il_threshold)),
    };
    let summaries = bot::run(cfg, &opts).await?;
    let report = ssbot_tools::bench::report(&tag, !no_advisor, &summaries);
    let path = Path::new(&cfg.recorder.runs_dir).join(format!("bench_{tag}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    println!("{report}\nsaved {}", path.display());
    Ok(())
}

pub fn crash_frames(args: CrashFramesArgs) -> Result<()> {
    let CrashFramesArgs { sources, out, per_run, window_ms } = args;
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
    Ok(())
}

pub fn bench_compare(args: BenchCompareArgs) -> Result<()> {
    let BenchCompareArgs { a, b } = args;
    for p in [a, b] {
        let r: ssbot_tools::bench::BenchReport = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        println!("{r}");
    }
    Ok(())
}

pub fn summarize(args: SummarizeArgs) -> Result<()> {
    let SummarizeArgs { runs } = args;
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
    Ok(())
}

pub async fn spike(cfg: &Config, args: SpikeArgs) -> Result<()> {
    let SpikeArgs { seconds } = args;
    let report = bot::spike(cfg, seconds).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

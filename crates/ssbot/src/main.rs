//! `ssbot` CLI (SPEC §7).

mod cli;
mod cmd;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Cmd};
use cmd::{advisor, inspect, label, play, train};
use ssbot_core::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "ssbot=info,ssbot_core=info,ssbot_live=info,ssbot_tools=info".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let mut cfg = Config::load(&cli.config)?;

    match cli.cmd {
        Cmd::Run(args) => play::run(&mut cfg, &cli.calibration, args).await?,
        Cmd::Bench(args) => play::bench(&mut cfg, &cli.calibration, args).await?,
        Cmd::CrashFrames(args) => play::crash_frames(args)?,
        Cmd::BenchCompare(args) => play::bench_compare(args)?,
        Cmd::Summarize(args) => play::summarize(args)?,
        Cmd::Spike(args) => play::spike(&cfg, args).await?,
        Cmd::Calibrate(args) => inspect::calibrate(&cfg, &cli.calibration, args).await?,
        Cmd::Replay(args) => inspect::replay(&cfg, &cli.calibration, args)?,
        Cmd::Frames(args) => inspect::frames(args)?,
        Cmd::Label(args) => label::label(&cli.calibration, args).await?,
        Cmd::Review(args) => label::review(&cli.calibration, args)?,
        Cmd::Select(args) => label::select(&cli.calibration, args)?,
        Cmd::Import(args) => label::import(&cli.calibration, args)?,
        Cmd::TrainByHuman(args) => train::train_by_human(&cfg, &cli.calibration, args).await?,
        Cmd::TrainZones(args) => train::train_zones(&cfg, &cli.calibration, args)?,
        Cmd::EvalZones(args) => train::eval_zones(&cfg, &cli.calibration, args)?,
        Cmd::IlData(args) => train::il_data(args)?,
        Cmd::IlEval(args) => train::il_eval(args)?,
        Cmd::See(args) => inspect::see(&cfg, &cli.calibration, args)?,
        Cmd::Mine(args) => inspect::mine(&cfg, &cli.calibration, args)?,
        Cmd::ZoneCrops(args) => train::zone_crops(&cfg, &cli.calibration, args)?,
        Cmd::Fit(args) => train::fit(&cfg, &cli.calibration, args)?,
        Cmd::AdvisorBench(args) => advisor::advisor_bench(&mut cfg, args).await?,
        Cmd::AdvisorData(args) => advisor::advisor_data(&cfg, args)?,
        Cmd::LabelDump(args) => label::label_dump(args)?,
    }
    Ok(())
}

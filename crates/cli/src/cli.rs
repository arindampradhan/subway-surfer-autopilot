//! Command-line interface: the `Cli` arguments and the `Cmd` subcommands. Doc comments are the `--help` text.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::cmd::{advisor, inspect, label, play, train};

#[derive(Parser)]
#[command(name = "ssbot", about = "Subway Surfers bot: Rust perception and reflexes, OpenJev advisor")]
pub struct Cli {
    /// Settings file (SPEC §11).
    #[arg(long, global = true, default_value = "ssbot.toml")]
    pub config: PathBuf,
    /// Zones, markers and thresholds (SPEC §4.3).
    #[arg(long, global = true, default_value = "calibration.toml")]
    pub calibration: PathBuf,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Play (or, with --human, record a human playing).
    Run(play::RunArgs),
    /// Play N runs and print median survival and score, to compare two configurations. Saves
    /// `runs/bench_<tag>.json`; compare two with `ssbot bench-compare a.json b.json`.
    Bench(play::BenchArgs),
    /// Copy frames from just before each run ended (from run folders, or a saved bench report)
    /// into one frames folder, ready for `ssbot label` / manual labelling.
    CrashFrames(play::CrashFramesArgs),
    /// Print two saved bench reports side by side.
    BenchCompare(play::BenchCompareArgs),
    /// Recompute `summary.json` (crash, survival, HUD score) for recorded runs.
    Summarize(play::SummarizeArgs),
    /// M0 spike: iframe origin, screencast fps/delay, whether keys reach the game.
    Spike(play::SpikeArgs),
    /// Capture reference frames per screen, or preview zones on frames.
    Calibrate(inspect::CalibrateArgs),
    /// Re-run perception and policy on a recorded run, no browser.
    Replay(inspect::ReplayArgs),
    /// ffmpeg → canvas-cropped frames at 10 fps, 640 px wide.
    Frames(inspect::FramesArgs),
    /// Label frames with Claude (Batches API by default).
    Label(label::LabelArgs),
    /// Static HTML gallery for spot-checks and fixes.
    Review(label::ReviewArgs),
    /// Copy a varied subset of frames (no near-duplicates) into a new directory for labelling.
    Select(label::SelectArgs),
    /// Import bounding boxes from CVAT / Label Studio / makesense.ai / Roboflow as zone labels.
    Import(label::ImportArgs),
    /// Recording of a person playing → frames → selection → labels → zone classifier →
    /// OpenJev head. Rerunnable: each step reuses what earlier runs produced.
    TrainByHuman(train::TrainByHumanArgs),
    /// Train the learned zone classifier on labels; reports cross-validated accuracy.
    TrainZones(train::TrainZonesArgs),
    /// Score a saved zone model on labelled frames (confusion matrix, missed hazards, false alarms).
    EvalZones(train::EvalZonesArgs),
    /// Frames from human-recorded runs with the action pressed shortly after each, for
    /// `sidecar/il_cnn.py` (imitation learning).
    IlData(train::IlDataArgs),
    /// Score an imitation model on data built by `il-data` (accuracy and time per frame).
    IlEval(train::IlEvalArgs),
    /// Draw what the bot's perception reports on a run's saved frames (a grid image), zones
    /// coloured by predicted class: F free, T train, R ramp, L low barrier, H high barrier.
    See(inspect::SeeArgs),
    /// Copy frames where the eyes report a barrier (from run folders or a bench report) into a
    /// folder for labelling: barriers are the scarce class.
    Mine(inspect::MineArgs),
    /// Write labelled zone crops (the CNN's training data) for `sidecar/zone_cnn.py`.
    ZoneCrops(train::ZoneCropsArgs),
    /// Fit classifier thresholds to Claude's labels and report held-out accuracy (M2).
    Fit(train::FitArgs),
    /// Score OpenJev option wordings on situations with a known best action (SPEC §9).
    AdvisorBench(advisor::AdvisorBenchArgs),
    /// Write OpenJev decision-head training and test sets (JSON lines) to a directory.
    AdvisorData(advisor::AdvisorDataArgs),
    /// Print final labels (Claude + human fixes) as JSON lines.
    LabelDump(label::LabelDumpArgs),
}

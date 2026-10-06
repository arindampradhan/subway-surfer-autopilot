//! Command-line interface: the `Cli` arguments and the `Cmd` subcommands. Doc comments are the `--help` text.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ssbot_core::config::CaptureBackend;
use ssbot_tools::{frames, label};

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
        #[arg(long, value_enum, default_value_t = ssbot_core::policy::arbiter::IlMode::Guarded)]
        il_mode: ssbot_core::policy::arbiter::IlMode,
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
        #[arg(long, value_enum, default_value_t = ssbot_core::policy::arbiter::IlMode::Guarded)]
        il_mode: ssbot_core::policy::arbiter::IlMode,
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

//! Frame labelling: `label`, `review`, `select`, `import`, `label-dump`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;

use ssbot_lab::label;
use ssbot_lab::label::cmd::LabelOpts;
use ssbot_lab::label::{load_final_labels, run_name};

#[derive(Args)]
pub struct LabelArgs {
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
    #[arg(long, default_value = "labels/label_guide.md")]
    guide: PathBuf,
    /// Label every n-th frame twice (10% by default).
    #[arg(long, default_value_t = 10)]
    repeat_every: usize,
}

#[derive(Args)]
pub struct ReviewArgs {
    frames_dir: PathBuf,
    #[arg(long, default_value = "labels")]
    labels: PathBuf,
    /// Label by hand from scratch (no Claude labels needed).
    #[arg(long)]
    manual: bool,
}

#[derive(Args)]
pub struct SelectArgs {
    frames_dir: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 300)]
    max: usize,
}

#[derive(Args)]
pub struct ImportArgs {
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
}

#[derive(Args)]
pub struct LabelDumpArgs {
    frames_dir: PathBuf,
    #[arg(long, default_value = "labels")]
    labels: PathBuf,
}

pub async fn label(calibration: &Path, args: LabelArgs) -> Result<()> {
    let LabelArgs { frames_dir, model, effort, max, sync, dry_run, labels, guide, repeat_every } = args;
    let opts = LabelOpts {
        frames_dir,
        model,
        effort,
        max,
        sync,
        dry_run,
        labels_dir: labels,
        calibration: calibration.to_path_buf(),
        guide,
        repeat_every,
    };
    label::cmd::label(&opts).await?;
    Ok(())
}

pub fn review(calibration: &Path, args: ReviewArgs) -> Result<()> {
    let ReviewArgs { frames_dir, labels, manual } = args;
    let out = label::cmd::review(&frames_dir, &labels, calibration, manual)?;
    eprintln!("open {}", out.display());
    Ok(())
}

pub fn select(calibration: &Path, args: SelectArgs) -> Result<()> {
    let SelectArgs { frames_dir, out, max } = args;
    let n = label::cmd::select_frames(&frames_dir, &out, max, calibration)?;
    eprintln!("selected {n} frames into {}", out.display());
    Ok(())
}

pub fn import(calibration: &Path, args: ImportArgs) -> Result<()> {
    let ImportArgs { frames_dir, annotations, format, classes, labels, min_overlap, force } = args;
    let (n, total) = label::cmd::import(label::cmd::ImportOpts {
        frames_dir: &frames_dir,
        annotations: &annotations,
        format,
        classes,
        labels_dir: &labels,
        calibration,
        min_overlap,
        force,
    })?;
    eprintln!("imported {n} labelled frames (of {total} in {}) into {}", frames_dir.display(), labels.join(format!("{}.jsonl", run_name(&frames_dir))).display());
    eprintln!("check them with `ssbot review {} --manual`, then `ssbot fit`", frames_dir.display());
    Ok(())
}

pub fn label_dump(args: LabelDumpArgs) -> Result<()> {
    let LabelDumpArgs { frames_dir, labels } = args;
    let run = run_name(&frames_dir);
    let mut all: Vec<_> = load_final_labels(&labels, &run)?.into_iter().collect();
    all.sort_by_key(|(id, _)| *id);
    for (id, l) in all {
        println!("{}", serde_json::json!({"run": run, "frame_id": id, "label": l}));
    }
    Ok(())
}

//! Imitation learning data (`ssbot il-data`): frames from recorded runs, each labelled with the
//! action a person pressed shortly after seeing it. Recorded with `ssbot run --human`, which logs
//! the person's key presses. Training happens in `sidecar/il_cnn.py`; the model runs in Rust.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use ssbot_engine::il::class_of;
use ssbot_engine::perception::GameState;
use ssbot_engine::policy::arbiter::{Command, Source};
use ssbot_runtime::recorder::read_events;

pub struct IlOpts {
    pub lead_ms: f64,
    /// Keep every n-th saved frame.
    pub stride: usize,
    pub size: (u32, u32),
    /// Only count key presses recorded from a human (`false` accepts the bot's own actions too,
    /// for testing the pipeline).
    pub human_only: bool,
}

/// Writes `frames.bin` (N × H × W × 3 bytes) and `meta.jsonl` to `out`. Returns the number of
/// frames and the count per class.
pub fn build(run_dirs: &[PathBuf], out: &Path, opts: &IlOpts) -> Result<(usize, [usize; 5])> {
    std::fs::create_dir_all(out)?;
    let mut bin = std::io::BufWriter::new(std::fs::File::create(out.join("frames.bin"))?);
    let mut meta = std::io::BufWriter::new(std::fs::File::create(out.join("meta.jsonl"))?);
    let (mut n, mut counts) = (0usize, [0usize; 5]);
    for (k, dir) in run_dirs.iter().enumerate() {
        let events = read_events(dir)?;
        let actions: Vec<(f64, usize)> = events
            .iter()
            .filter(|e| !opts.human_only || e.chosen.source == Source::Human)
            .filter_map(|e| match e.chosen.command {
                Command::Act(a) => class_of(a).map(|c| (e.t, c)),
                _ => None,
            })
            .collect();
        let mut kept = 0usize;
        for e in events.iter().filter(|e| e.obs.state == GameState::Running) {
            let path = dir.join("frames").join(format!("{}.jpg", e.frame_id));
            if !path.exists() {
                continue;
            }
            kept += 1;
            if !(kept - 1).is_multiple_of(opts.stride.max(1)) {
                continue;
            }
            let label = actions.iter().find(|(t, _)| *t >= e.t && *t <= e.t + opts.lead_ms).map_or(0, |(_, c)| *c);
            let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
            let img = image::imageops::resize(&img, opts.size.0, opts.size.1, image::imageops::FilterType::Triangle);
            bin.write_all(img.as_raw())?;
            writeln!(meta, "{}", serde_json::json!({"run": k, "id": e.frame_id, "t": e.t, "label": label}))?;
            counts[label] += 1;
            n += 1;
        }
    }
    Ok((n, counts))
}

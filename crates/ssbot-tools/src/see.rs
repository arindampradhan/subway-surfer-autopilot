//! `ssbot see`: what the bot's eyes report. Runs perception on a run's saved frames and draws each
//! zone in the colour of the class it predicts, so misreads are visible at a glance.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

use ssbot_core::config::Config;
use ssbot_core::perception::zones::Calibration;
use ssbot_core::perception::{Obstacle, Perceiver};

use crate::label::overlay::{draw_quad, draw_text};
use crate::label::select::list_frames;

fn colour(o: Obstacle) -> Rgb<u8> {
    match o {
        Obstacle::Free => Rgb([60, 220, 90]),
        Obstacle::TrainBody => Rgb([255, 50, 50]),
        Obstacle::TrainRamp => Rgb([255, 150, 30]),
        Obstacle::LowBarrier => Rgb([255, 240, 0]),
        Obstacle::HighBarrier => Rgb([255, 60, 255]),
        Obstacle::OverheadBar => Rgb([0, 230, 255]),
        Obstacle::Unknown => Rgb([150, 150, 150]),
    }
}

fn letter(o: Obstacle) -> &'static str {
    match o {
        Obstacle::Free => "F",
        Obstacle::TrainBody => "T",
        Obstacle::TrainRamp => "R",
        Obstacle::LowBarrier => "L",
        Obstacle::HighBarrier => "H",
        Obstacle::OverheadBar => "O",
        Obstacle::Unknown => "U",
    }
}

/// Perceives every saved frame in order (the zone vote needs the history), then writes a grid of
/// `n` evenly spaced running frames to `out`: 2 columns, zones coloured by predicted class.
pub fn sheet(cfg: &Config, calib_path: &Path, frames_dir: &Path, out: &Path, n: usize, from_t: Option<f64>) -> Result<usize> {
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let calib = Calibration::load_or_default(calib_path);
    let base = calib_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut p = Perceiver::new(calib.clone(), &base, work);
    let frames: Vec<(u64, PathBuf)> = list_frames(frames_dir)?;
    let mut shown: Vec<(u64, RgbImage, ssbot_core::perception::Observation)> = Vec::new();
    for (id, path) in &frames {
        let canvas = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let small = image::imageops::resize(&canvas, work.0, work.1, image::imageops::FilterType::Triangle);
        let m = p.measure_with(*id, &small, Some(&canvas));
        if m.obs.state == ssbot_core::perception::GameState::Running {
            shown.push((*id, canvas, m.obs));
        }
    }
    let skip = from_t.map_or(0, |f| (shown.len() as f64 * f) as usize);
    let shown = &shown[skip.min(shown.len())..];
    if shown.is_empty() {
        anyhow::bail!("no running frames in {}", frames_dir.display());
    }
    let pick: Vec<usize> = (0..n.min(shown.len())).map(|i| i * shown.len() / n.min(shown.len())).collect();
    let tile_w = 480u32;
    let tile_h = (shown[0].1.height() as f64 * tile_w as f64 / shown[0].1.width() as f64) as u32;
    let rows = pick.len().div_ceil(2) as u32;
    let mut sheet = RgbImage::new(tile_w * 2, tile_h * rows);
    for (k, idx) in pick.iter().enumerate() {
        let (id, canvas, obs) = &shown[*idx];
        let mut tile = image::imageops::resize(canvas, tile_w, tile_h, image::imageops::FilterType::Triangle);
        for (lane, zones) in calib.lanes.iter().enumerate() {
            let classes = [obs.lanes[lane].near, obs.lanes[lane].mid, obs.lanes[lane].far];
            for (band, q) in zones.bands().iter().enumerate() {
                draw_quad(&mut tile, q, colour(classes[band]));
                let (x0, y0, x1, _) = q.bounds();
                let tx = ((x0 + x1) / 2.0 * tile_w as f32) as i32 - 2;
                let ty = (y0 * tile_h as f32) as i32 + 4;
                draw_text(&mut tile, tx, ty, letter(classes[band]), colour(classes[band]), 2);
            }
        }
        draw_text(&mut tile, 6, 6, &format!("{id}"), Rgb([255, 255, 255]), 2);
        let (x, y) = ((k as u32 % 2) * tile_w, (k as u32 / 2) * tile_h);
        image::imageops::replace(&mut sheet, &tile, x as i64, y as i64);
    }
    sheet.save(out)?;
    Ok(pick.len())
}

/// Frames from recorded runs where the eyes report a barrier (low, high or overhead) anywhere,
/// at least `gap_ms` apart, copied into `out` for labelling: barriers are the scarce class, and
/// crash windows alone don't supply enough of them. Ids get a per-run block like `crash-frames`.
pub fn mine_barriers(cfg: &Config, calib_path: &Path, run_dirs: &[PathBuf], out: &Path, per_run: usize, gap_ms: f64) -> Result<usize> {
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let calib = Calibration::load_or_default(calib_path);
    let base = calib_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    std::fs::create_dir_all(out)?;
    let mut copied = 0;
    for (k, dir) in run_dirs.iter().enumerate() {
        let events = ssbot_live::recorder::read_events(dir)?;
        let t_of: std::collections::HashMap<u64, f64> = events.iter().map(|e| (e.frame_id, e.t)).collect();
        let mut p = Perceiver::new(calib.clone(), &base, work);
        let mut last_t = f64::NEG_INFINITY;
        let mut n = 0;
        for (id, path) in list_frames(&dir.join("frames"))? {
            if n >= per_run {
                break;
            }
            let canvas = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
            let small = image::imageops::resize(&canvas, work.0, work.1, image::imageops::FilterType::Triangle);
            let obs = p.measure_with(id, &small, Some(&canvas)).obs;
            let t = t_of.get(&id).copied().unwrap_or(0.0);
            let barrier = obs.state == ssbot_core::perception::GameState::Running
                && obs.lanes.iter().any(|v| {
                    [v.near, v.mid, v.far].iter().any(|o| matches!(o, Obstacle::LowBarrier | Obstacle::HighBarrier | Obstacle::OverheadBar))
                });
            if barrier && t - last_t >= gap_ms {
                std::fs::copy(&path, out.join(format!("{}.jpg", (k as u64 + 1) * 10_000_000 + id)))?;
                last_t = t;
                n += 1;
            }
        }
        copied += n;
    }
    Ok(copied)
}

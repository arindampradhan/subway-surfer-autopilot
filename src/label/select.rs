//! Frame selection to keep labelling cost down (SPEC §4.8 Inputs): always keep state changes
//! and ~1 s before every crash, drop near-duplicates by 64-bit perceptual hash, then sample the
//! rest evenly up to a per-session cap.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image_hasher::{HasherConfig, ImageHash};

use crate::perception::GameState;

#[derive(Debug, Clone)]
pub struct FrameInfo {
    pub id: u64,
    pub path: PathBuf,
    /// Milliseconds from the start of the recording.
    pub t: f64,
    pub state: GameState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    StateChange,
    PreCrash,
    Sampled,
}

/// Frames in a directory sorted by id. Ids come from numeric file stems (`123.jpg`,
/// `f_00123.jpg`), else from sorted order.
pub fn list_frames(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| matches!(x.to_str(), Some("jpg" | "jpeg" | "png"))))
        .filter(|p| !p.file_stem().is_some_and(|s| s.to_string_lossy().starts_with("contact")))
        .collect();
    let num = |p: &Path| -> Option<u64> {
        let stem = p.file_stem()?.to_string_lossy().to_string();
        let digits: String = stem.chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
        digits.parse().ok()
    };
    paths.sort_by_key(|p| (num(p).unwrap_or(u64::MAX), p.clone()));
    let all_numbered = paths.iter().all(|p| num(p).is_some());
    Ok(paths.into_iter().enumerate().map(|(i, p)| (if all_numbered { num(&p).unwrap() } else { i as u64 + 1 }, p)).collect())
}

pub fn phash(path: &Path) -> Result<ImageHash> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(HasherConfig::new().hash_size(8, 8).to_hasher().hash_image(&img))
}

/// Picks frames to label. `hashes[i]` belongs to `frames[i]`.
pub fn select(frames: &[FrameInfo], hashes: &[ImageHash], cap: usize, dup_bits: u32) -> Vec<(usize, Reason)> {
    let crash_times: Vec<f64> = frames
        .windows(2)
        .filter(|w| w[1].state == GameState::Crashed && w[0].state != GameState::Crashed)
        .map(|w| w[1].t)
        .collect();
    let mut must: Vec<(usize, Reason)> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if i > 0 && frames[i - 1].state != f.state {
            must.push((i, Reason::StateChange));
        } else if crash_times.iter().any(|&c| f.t >= c - 1000.0 && f.t <= c) {
            must.push((i, Reason::PreCrash));
        }
    }
    let is_must = |i: usize| must.iter().any(|(m, _)| *m == i);

    // Near-duplicate removal against the last kept frame, in time order.
    let mut candidates = Vec::new();
    let mut last: Option<&ImageHash> = None;
    for i in 0..frames.len() {
        if is_must(i) {
            last = Some(&hashes[i]);
            continue;
        }
        if last.is_some_and(|h| h.dist(&hashes[i]) <= dup_bits) {
            continue;
        }
        last = Some(&hashes[i]);
        candidates.push(i);
    }

    let room = cap.saturating_sub(must.len());
    let mut out = must;
    if room > 0 && !candidates.is_empty() {
        let step = (candidates.len() as f64 / room as f64).max(1.0);
        let mut k = 0.0;
        while (k as usize) < candidates.len() && out.len() < cap {
            out.push((candidates[k as usize], Reason::Sampled));
            k += step;
        }
    }
    out.sort_by_key(|(i, _)| *i);
    out.dedup_by_key(|(i, _)| *i);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};

    fn hash_of(seed: u8) -> ImageHash {
        let img = RgbImage::from_fn(32, 32, |x, y| {
            let v = ((x as u32 * (seed as u32 + 1) * 7 + y as u32 * 13 * (seed as u32 % 5 + 1)) % 256) as u8;
            Rgb([v, v, v])
        });
        HasherConfig::new().hash_size(8, 8).to_hasher().hash_image(&DynamicImage::ImageRgb8(img))
    }

    #[test]
    fn keeps_crash_window_and_state_changes_and_caps() {
        let mut frames = Vec::new();
        let mut hashes = Vec::new();
        for i in 0..200u64 {
            let state = match i {
                0..=9 => GameState::Menu,
                150..=160 => GameState::Crashed,
                _ => GameState::Running,
            };
            frames.push(FrameInfo { id: i, path: PathBuf::new(), t: i as f64 * 100.0, state });
            hashes.push(hash_of((i % 50) as u8));
        }
        let picked = select(&frames, &hashes, 40, 6);
        let ids: Vec<usize> = picked.iter().map(|(i, _)| *i).collect();
        // State changes at 10, 150, 161 and every frame within 1 s before the crash.
        for must in [10, 150, 161, 140, 145, 149] {
            assert!(ids.contains(&must), "missing {must}");
        }
        assert!(picked.len() <= 40, "{}", picked.len());
    }

    #[test]
    fn drops_exact_duplicates() {
        let frames: Vec<FrameInfo> =
            (0..20).map(|i| FrameInfo { id: i, path: PathBuf::new(), t: i as f64 * 100.0, state: GameState::Running }).collect();
        let hashes: Vec<ImageHash> = (0..20).map(|_| hash_of(3)).collect();
        assert_eq!(select(&frames, &hashes, 100, 6).len(), 1);
    }
}

//! Learned zone classifier (between SPEC §4.3's v1 heuristics and the v2 CNN): multinomial
//! logistic regression on a zone's colour histogram and edge features, trained on labels.
//! The v1 colour rules can't tell this game's orange track tiles from train paint; a model
//! fitted to labelled zones can. Saved as JSON next to the calibration (`zone_model`).

use std::path::Path;

use anyhow::{Context, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};

use super::Obstacle;
use super::classify::hsv;
use super::zones::ZoneMask;

const HUE_BINS: usize = 12;
const SAT_BINS: usize = 4;
const VAL_BINS: usize = 4;
/// Features: hue histogram (weighted by saturation), saturation and value histograms, the same
/// three for the top and bottom halves of the zone, edge density, and a bias.
pub const DIM: usize = 3 * (HUE_BINS + SAT_BINS + VAL_BINS) + 1 + 1;

pub fn zone_vector(img: &RgbImage, mask: &ZoneMask) -> Vec<f32> {
    let w = img.width();
    let mut v = vec![0.0f32; DIM];
    let block = HUE_BINS + SAT_BINS + VAL_BINS;
    let (mut n, mut n_top, mut n_bot, mut edges) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(idx, rel_y) in &mask.pixels {
        let (x, y) = (idx % w, idx / w);
        let p = img.get_pixel(x, y);
        let (h, s, val) = hsv(p[0], p[1], p[2]);
        let hb = ((h / 360.0 * HUE_BINS as f32) as usize).min(HUE_BINS - 1);
        let sb = ((s * SAT_BINS as f32) as usize).min(SAT_BINS - 1);
        let vb = ((val * VAL_BINS as f32) as usize).min(VAL_BINS - 1);
        let half = if rel_y < 0.5 { 1 } else { 2 };
        for (offset, weight) in [(0, 1.0), (half * block, 1.0)] {
            v[offset + hb] += s * weight;
            v[offset + HUE_BINS + sb] += weight;
            v[offset + HUE_BINS + SAT_BINS + vb] += weight;
        }
        n += 1.0;
        if half == 1 { n_top += 1.0 } else { n_bot += 1.0 }
        if x + 1 < w && y + 1 < img.height() {
            let l = |q: &image::Rgb<u8>| q[0] as i32 * 299 + q[1] as i32 * 587 + q[2] as i32 * 114;
            let g = (l(img.get_pixel(x + 1, y)) - l(p)).abs() + (l(img.get_pixel(x, y + 1)) - l(p)).abs();
            if g > 48_000 {
                edges += 1.0;
            }
        }
    }
    for (b, count) in [(0, n), (1, n_top), (2, n_bot)] {
        for i in 0..block {
            v[b * block + i] /= count.max(1.0);
        }
    }
    v[DIM - 2] = edges / n.max(1.0);
    v[DIM - 1] = 1.0;
    v
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneModel {
    pub classes: Vec<Obstacle>,
    /// One weight row per class, `DIM` long.
    pub weights: Vec<Vec<f32>>,
    /// Training notes: samples, cross-validated accuracy.
    #[serde(default)]
    pub note: String,
}

fn softmax(z: &mut [f32]) {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in z.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for x in z.iter_mut() {
        *x /= sum;
    }
}

impl ZoneModel {
    /// Full-batch gradient descent with L2, class-balanced so rare obstacles count.
    pub fn train(samples: &[(Vec<f32>, Obstacle)], epochs: usize, l2: f32) -> ZoneModel {
        let mut classes: Vec<Obstacle> = Vec::new();
        for (_, o) in samples {
            if !classes.contains(o) {
                classes.push(*o);
            }
        }
        classes.sort_by_key(|o| Obstacle::ALL.iter().position(|a| a == o));
        let k = classes.len();
        let counts: Vec<f32> = classes.iter().map(|c| samples.iter().filter(|(_, o)| o == c).count() as f32).collect();
        let class_w: Vec<f32> = counts.iter().map(|n| samples.len() as f32 / (k as f32 * n.max(1.0))).collect();
        let mut w = vec![vec![0.0f32; DIM]; k];
        let lr = 0.5;
        for _ in 0..epochs {
            let mut grad = vec![vec![0.0f32; DIM]; k];
            for (x, o) in samples {
                let t = classes.iter().position(|c| c == o).unwrap();
                let mut z: Vec<f32> = w.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect();
                softmax(&mut z);
                for c in 0..k {
                    let err = (z[c] - if c == t { 1.0 } else { 0.0 }) * class_w[t];
                    for (g, xi) in grad[c].iter_mut().zip(x) {
                        *g += err * xi;
                    }
                }
            }
            let n = samples.len().max(1) as f32;
            for c in 0..k {
                for j in 0..DIM {
                    let reg = if j == DIM - 1 { 0.0 } else { l2 * w[c][j] };
                    w[c][j] -= lr * (grad[c][j] / n + reg);
                }
            }
        }
        ZoneModel { classes, weights: w, note: String::new() }
    }

    pub fn predict(&self, x: &[f32]) -> (Obstacle, f32) {
        if self.classes.len() == 1 {
            return (self.classes[0], 1.0);
        }
        let mut z: Vec<f32> = self.weights.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect();
        softmax(&mut z);
        let (i, p) = z.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        (self.classes[i], *p)
    }

    pub fn load(path: &Path) -> Result<ZoneModel> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let m: ZoneModel = serde_json::from_str(&text)?;
        anyhow::ensure!(m.weights.iter().all(|r| r.len() == DIM), "zone model has the wrong feature size");
        Ok(m)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::zones::Quad;

    #[test]
    fn learns_orange_track_vs_white_train() {
        let q = Quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let mask = q.mask(16, 16);
        let orange = RgbImage::from_pixel(16, 16, image::Rgb([240, 160, 40]));
        let white = RgbImage::from_pixel(16, 16, image::Rgb([235, 230, 225]));
        let red = RgbImage::from_pixel(16, 16, image::Rgb([220, 30, 40]));
        let samples = vec![
            (zone_vector(&orange, &mask), Obstacle::Free),
            (zone_vector(&white, &mask), Obstacle::TrainBody),
            (zone_vector(&red, &mask), Obstacle::TrainBody),
        ];
        let m = ZoneModel::train(&samples, 300, 1e-3);
        assert_eq!(m.predict(&samples[0].0).0, Obstacle::Free);
        assert_eq!(m.predict(&samples[1].0).0, Obstacle::TrainBody);
        assert_eq!(m.predict(&samples[2].0).0, Obstacle::TrainBody);
    }
}

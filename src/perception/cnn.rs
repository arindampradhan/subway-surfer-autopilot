//! Zone CNN (SPEC §4.3 v2): a small conv net on a zone's polygon-masked crop, trained offline by
//! `sidecar/zone_cnn.py` and run here in plain Rust. The crop code is shared with
//! `ssbot zone-crops`, which produces the training data, so training and live crops match.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};

use super::Obstacle;
use super::model::ZoneModel;
use super::zones::ZoneMask;

/// Default crop side; a model can ask for another (`crop` in its JSON).
pub const CROP: usize = 32;

/// The zone's bounding box resized to `size`×`size` (row-major RGB bytes). With `ctx` = 0 the
/// pixels outside the zone polygon are set to grey; with `ctx` > 0 the box is grown by that
/// fraction on every side and the real surroundings are kept, so a barrier poking above the zone
/// or a train beside it is still visible.
pub fn zone_crop(img: &RgbImage, mask: &ZoneMask, size: usize, ctx: f32) -> Vec<u8> {
    let w = img.width();
    let Some(&(first, _)) = mask.pixels.first() else { return vec![128; size * size * 3] };
    let (mut x0, mut y0, mut x1, mut y1) = (first % w, first / w, first % w, first / w);
    for &(idx, _) in &mask.pixels {
        let (x, y) = (idx % w, idx / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    if ctx > 0.0 {
        let (gx, gy) = (((x1 - x0 + 1) as f32 * ctx) as u32, ((y1 - y0 + 1) as f32 * ctx) as u32);
        let (nx0, ny0) = (x0.saturating_sub(gx), y0.saturating_sub(gy));
        let (nx1, ny1) = ((x1 + gx).min(img.width() - 1), (y1 + gy).min(img.height() - 1));
        let bbox = image::imageops::crop_imm(img, nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1).to_image();
        return image::imageops::resize(&bbox, size as u32, size as u32, image::imageops::FilterType::Triangle).into_raw();
    }
    let mut bbox = RgbImage::from_pixel(x1 - x0 + 1, y1 - y0 + 1, Rgb([128, 128, 128]));
    for &(idx, _) in &mask.pixels {
        let (x, y) = (idx % w, idx / w);
        bbox.put_pixel(x - x0, y - y0, *img.get_pixel(x, y));
    }
    image::imageops::resize(&bbox, size as u32, size as u32, image::imageops::FilterType::Triangle).into_raw()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConvLayer {
    /// `[out][in][3][3]`, flattened.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Dense {
    /// `[out][in]`, flattened.
    w: Vec<f32>,
    b: Vec<f32>,
    din: usize,
    dout: usize,
}

/// Conv(3×3, same) → ReLU → max-pool(2), three times (batch norm folded into the convs), then
/// two dense layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneCnn {
    kind: String,
    classes: Vec<Obstacle>,
    convs: Vec<ConvLayer>,
    fc1: Dense,
    fc2: Dense,
    #[serde(default)]
    pub note: String,
    /// Crop side in pixels (default 32).
    #[serde(default = "default_crop")]
    pub crop: usize,
    /// Trained on crops cut from the full-resolution frame rather than the 320×180 work image.
    #[serde(default)]
    pub hires: bool,
    /// Context grown around each zone's box (0 = the zone only, outside pixels greyed).
    #[serde(default)]
    pub ctx: f32,
}

fn default_crop() -> usize {
    CROP
}

fn conv_relu_pool(x: &[f32], size: usize, l: &ConvLayer) -> Vec<f32> {
    let (cin, cout) = (l.cin, l.cout);
    let mut y = vec![0.0f32; cout * size * size];
    for co in 0..cout {
        let out = &mut y[co * size * size..(co + 1) * size * size];
        out.fill(l.b[co]);
        for ci in 0..cin {
            let k = &l.w[(co * cin + ci) * 9..(co * cin + ci) * 9 + 9];
            let inp = &x[ci * size * size..(ci + 1) * size * size];
            for ky in 0..3 {
                for kx in 0..3 {
                    let wv = k[ky * 3 + kx];
                    let (dy, dx) = (ky as isize - 1, kx as isize - 1);
                    for oy in 0..size {
                        let iy = oy as isize + dy;
                        if iy < 0 || iy >= size as isize {
                            continue;
                        }
                        let (x_lo, x_hi) = (0.max(-dx) as usize, size.min((size as isize - dx) as usize));
                        let row_in = &inp[iy as usize * size..iy as usize * size + size];
                        let row_out = &mut out[oy * size..oy * size + size];
                        for ox in x_lo..x_hi {
                            row_out[ox] += wv * row_in[(ox as isize + dx) as usize];
                        }
                    }
                }
            }
        }
    }
    let half = size / 2;
    let mut pooled = vec![0.0f32; cout * half * half];
    for c in 0..cout {
        for py in 0..half {
            for px in 0..half {
                let at = |yy: usize, xx: usize| y[c * size * size + yy * size + xx].max(0.0);
                pooled[c * half * half + py * half + px] =
                    at(2 * py, 2 * px).max(at(2 * py, 2 * px + 1)).max(at(2 * py + 1, 2 * px)).max(at(2 * py + 1, 2 * px + 1));
            }
        }
    }
    pooled
}

fn dense(x: &[f32], l: &Dense, relu: bool) -> Vec<f32> {
    (0..l.dout)
        .map(|o| {
            let s = l.b[o] + l.w[o * l.din..(o + 1) * l.din].iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
            if relu { s.max(0.0) } else { s }
        })
        .collect()
}

impl ZoneCnn {
    pub fn load(path: &Path) -> Result<ZoneCnn> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let m: ZoneCnn = serde_json::from_str(&text)?;
        ensure!(m.kind == "cnn", "not a zone CNN");
        ensure!(m.convs.len() == 3 && m.convs[0].cin == 3, "unexpected layer layout");
        let side = m.crop / 8;
        ensure!(m.fc1.din == m.convs[2].cout * side * side, "fc1 input size doesn't match the conv stack");
        Ok(m)
    }

    /// Class and probability for one RGB crop from `zone_crop`.
    pub fn predict(&self, crop: &[u8]) -> (Obstacle, f32) {
        // HWC bytes -> CHW floats in 0..1.
        let n = self.crop;
        let mut x = vec![0.0f32; 3 * n * n];
        for (i, px) in crop.chunks_exact(3).enumerate() {
            for c in 0..3 {
                x[c * n * n + i] = px[c] as f32 / 255.0;
            }
        }
        let mut size = n;
        for l in &self.convs {
            x = conv_relu_pool(&x, size, l);
            size /= 2;
        }
        let logits = dense(&dense(&x, &self.fc1, true), &self.fc2, false);
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|z| (z - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let (i, p) = exp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        (self.classes[i], p / sum)
    }
}

/// Whichever zone classifier the calibration points at.
pub enum Classifier {
    Logistic(ZoneModel),
    Cnn(ZoneCnn),
}

impl Classifier {
    pub fn load(path: &Path) -> Result<Classifier> {
        match ZoneCnn::load(path) {
            Ok(cnn) => Ok(Classifier::Cnn(cnn)),
            Err(_) => Ok(Classifier::Logistic(ZoneModel::load(path)?)),
        }
    }

    pub fn predict(&self, img: &RgbImage, mask: &ZoneMask) -> Obstacle {
        match self {
            Classifier::Logistic(m) => m.predict(&super::model::zone_vector(img, mask)).0,
            Classifier::Cnn(m) => m.predict(&zone_crop(img, mask, m.crop, m.ctx)).0,
        }
    }

    /// Whether this model wants crops from the full-resolution frame.
    pub fn hires(&self) -> bool {
        matches!(self, Classifier::Cnn(m) if m.hires)
    }

    pub fn crop_size(&self) -> usize {
        match self {
            Classifier::Cnn(m) => m.crop,
            Classifier::Logistic(_) => CROP,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::zones::Quad;

    #[test]
    fn crop_is_fixed_size_and_masked() {
        let img = RgbImage::from_pixel(64, 36, Rgb([200, 30, 30]));
        let quad = Quad([[0.2, 0.2], [0.6, 0.2], [0.5, 0.9], [0.3, 0.9]]);
        let crop = zone_crop(&img, &quad.mask(64, 36), CROP, 0.0);
        assert_eq!(crop.len(), CROP * CROP * 3);
        // The middle of the crop is the red image; a corner outside the trapezoid is grey.
        let mid = (CROP / 2 * CROP + CROP / 2) * 3;
        assert!(crop[mid] > 150 && crop[mid + 1] < 80);
        let bottom_left = (CROP - 1) * CROP * 3;
        assert_eq!(&crop[bottom_left..bottom_left + 3], &[128, 128, 128]);
    }

    #[test]
    fn tiny_net_runs_and_picks_a_class() {
        let layer = |cin, cout| ConvLayer { w: vec![0.01; cout * cin * 9], b: vec![0.0; cout], cin, cout };
        let net = ZoneCnn {
            kind: "cnn".into(),
            classes: vec![Obstacle::Free, Obstacle::TrainBody],
            convs: vec![layer(3, 4), layer(4, 4), layer(4, 4)],
            fc1: Dense { w: vec![0.01; 8 * 4 * 16], b: vec![0.0; 8], din: 4 * 16, dout: 8 },
            fc2: Dense { w: [vec![0.0; 8], vec![1.0; 8]].concat(), b: vec![0.0; 2], din: 8, dout: 2 },
            note: String::new(),
            crop: CROP,
            hires: false,
            ctx: 0.0,
        };
        let (o, p) = net.predict(&vec![255; CROP * CROP * 3]);
        assert_eq!(o, Obstacle::TrainBody);
        assert!(p > 0.5 && p <= 1.0);
    }
}

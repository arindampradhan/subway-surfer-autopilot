//! The imitation model (`ssbot run --il`): the CNN trained by `sidecar/il_cnn.py` on data from
//! `ssbot il-data`, run in Rust on the live path.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::policy::Action;

/// Class order shared with `sidecar/il_cnn.py` and the Rust model.
pub const IL_ACTIONS: [Action; 5] = [Action::Stay, Action::Left, Action::Right, Action::Jump, Action::Roll];

pub fn class_of(a: Action) -> Option<usize> {
    IL_ACTIONS.iter().position(|x| *x == a)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Conv {
    /// `[out][in][k][k]`, flattened.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fc {
    w: Vec<f32>,
    b: Vec<f32>,
    din: usize,
    dout: usize,
}

/// The imitation CNN trained by `sidecar/il_cnn.py` (batch norm already folded into the convs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IlNet {
    kind: String,
    pub width: u32,
    pub height: u32,
    convs: Vec<Conv>,
    fc1: Fc,
    fc2: Fc,
    #[serde(default)]
    pub note: String,
}

fn conv_relu(x: &[f32], (h, w): (usize, usize), l: &Conv) -> (Vec<f32>, (usize, usize)) {
    let (oh, ow) = ((h + 2 * l.pad - l.k) / l.stride + 1, (w + 2 * l.pad - l.k) / l.stride + 1);
    let mut y = vec![0.0f32; l.cout * oh * ow];
    for co in 0..l.cout {
        let out = &mut y[co * oh * ow..(co + 1) * oh * ow];
        out.fill(l.b[co]);
        for ci in 0..l.cin {
            let inp = &x[ci * h * w..(ci + 1) * h * w];
            for ky in 0..l.k {
                for kx in 0..l.k {
                    let wv = l.w[((co * l.cin + ci) * l.k + ky) * l.k + kx];
                    for oy in 0..oh {
                        let iy = (oy * l.stride + ky) as isize - l.pad as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        let row = &inp[iy as usize * w..iy as usize * w + w];
                        let orow = &mut out[oy * ow..oy * ow + ow];
                        for (ox, o) in orow.iter_mut().enumerate() {
                            let ix = (ox * l.stride + kx) as isize - l.pad as isize;
                            if ix >= 0 && ix < w as isize {
                                *o += wv * row[ix as usize];
                            }
                        }
                    }
                }
            }
        }
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
    }
    (y, (oh, ow))
}

fn fc(x: &[f32], l: &Fc, relu: bool) -> Vec<f32> {
    (0..l.dout)
        .map(|o| {
            let s = l.b[o] + l.w[o * l.din..(o + 1) * l.din].iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
            if relu { s.max(0.0) } else { s }
        })
        .collect()
}

impl IlNet {
    pub fn load(path: &Path) -> Result<IlNet> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let net: IlNet = serde_json::from_str(&text)?;
        ensure!(net.kind == "il", "not an imitation model");
        ensure!(net.fc2.dout == IL_ACTIONS.len(), "unexpected number of actions");
        Ok(net)
    }

    /// The input size the network expects; frames are resized to it with the same filter the
    /// training data used.
    pub fn prepare(&self, canvas: &RgbImage) -> RgbImage {
        image::imageops::resize(canvas, self.width, self.height, image::imageops::FilterType::Triangle)
    }

    /// Action probabilities in `IL_ACTIONS` order for an image from `prepare`.
    pub fn predict(&self, img: &RgbImage) -> [f32; 5] {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut x = vec![0.0f32; 3 * h * w];
        for (i, px) in img.as_raw().chunks_exact(3).enumerate() {
            for c in 0..3 {
                x[c * h * w + i] = px[c] as f32 / 255.0;
            }
        }
        let mut dims = (h, w);
        for l in &self.convs {
            (x, dims) = conv_relu(&x, dims, l);
        }
        let logits = fc(&fc(&x, &self.fc1, true), &self.fc2, false);
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|z| (z - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut p = [0.0f32; 5];
        for (o, e) in p.iter_mut().zip(&exp) {
            *o = e / sum;
        }
        p
    }

    /// The most likely action and its probability.
    pub fn best(&self, img: &RgbImage) -> (Action, f32) {
        let p = self.predict(img);
        let (i, v) = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        (IL_ACTIONS[i], *v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_order() {
        assert_eq!(class_of(Action::Stay), Some(0));
        assert_eq!(class_of(Action::Roll), Some(4));
        assert_eq!(class_of(Action::Hoverboard), None);
    }
}

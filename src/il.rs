//! Imitation learning data (`ssbot il-data`): frames from recorded runs, each labelled with the
//! action a person pressed shortly after seeing it. Recorded with `ssbot run --human`, which logs
//! the person's key presses. Training happens in `sidecar/il_cnn.py`; the model runs in Rust.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::perception::GameState;
use crate::policy::Action;
use crate::policy::arbiter::{Command, Source};
use crate::recorder::read_events;

/// Class order shared with `sidecar/il_cnn.py` and the Rust model.
pub const IL_ACTIONS: [Action; 5] = [Action::Stay, Action::Left, Action::Right, Action::Jump, Action::Roll];

pub fn class_of(a: Action) -> Option<usize> {
    IL_ACTIONS.iter().position(|x| *x == a)
}

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
            if (kept - 1) % opts.stride.max(1) != 0 {
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

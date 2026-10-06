//! Fitting the v1 heuristics to Claude's labels (SPEC §4.3): grid-search the thresholds in
//! `calibration.toml` on a training split, then report SPEC §9 accuracy on a held-out split.

use std::collections::BTreeMap;
use std::path::Path;

use image::RgbImage;
use serde::Serialize;

use ssbot_core::perception::classify::{ZoneFeatures, classify_zone};
use ssbot_core::perception::zones::{Calibration, Thresholds};
use ssbot_core::perception::{GameState, Obstacle, Perceiver};

use crate::label::FrameLabel;

/// Deterministic 80/20 split by frame id.
pub fn is_test(frame_id: u64) -> bool {
    frame_id.wrapping_mul(0x9E3779B97F4A7C15) >> 60 < 3 // 3/16 ≈ 19%
}

#[derive(Debug, Clone, Copy)]
pub struct ZoneSample {
    pub features: ZoneFeatures,
    pub truth: Obstacle,
    pub band: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Accuracy {
    pub zones: usize,
    pub overall: f64,
    pub near: f64,
    pub per_class: BTreeMap<String, (usize, usize)>,
    pub game_state: Option<f64>,
    pub player_lane: Option<f64>,
}

impl Accuracy {
    /// SPEC §9 targets: obstacle ≥95% (near ≥98%), lane ≥99%, game state ≥99%.
    pub fn meets_targets(&self) -> bool {
        self.overall >= 0.95
            && self.near >= 0.98
            && self.player_lane.is_none_or(|a| a >= 0.99)
            && self.game_state.is_none_or(|a| a >= 0.99)
    }
}

pub fn zone_accuracy(samples: &[ZoneSample], th: &Thresholds) -> Accuracy {
    let mut acc = Accuracy { zones: samples.len(), ..Default::default() };
    let (mut ok, mut near_ok, mut near_n) = (0usize, 0usize, 0usize);
    for s in samples {
        let hit = classify_zone(&s.features, th) == s.truth;
        ok += hit as usize;
        if s.band == 0 {
            near_n += 1;
            near_ok += hit as usize;
        }
        let e = acc.per_class.entry(format!("{:?}", s.truth)).or_default();
        e.1 += 1;
        e.0 += hit as usize;
    }
    acc.overall = ok as f64 / samples.len().max(1) as f64;
    acc.near = near_ok as f64 / near_n.max(1) as f64;
    acc
}

/// Balanced (mean per-class) accuracy blended with overall, so Free doesn't dominate.
fn objective(samples: &[ZoneSample], th: &Thresholds) -> f64 {
    let a = zone_accuracy(samples, th);
    let balanced =
        a.per_class.values().map(|(ok, n)| *ok as f64 / (*n).max(1) as f64).sum::<f64>() / a.per_class.len().max(1) as f64;
    0.5 * balanced + 0.5 * a.overall
}

type Field = (&'static str, fn(&mut Thresholds) -> &mut f32, &'static [f32]);

const FIELDS: &[Field] = &[
    ("free_occ_max", |t| &mut t.free_occ_max, &[0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35]),
    ("train_occ_min", |t| &mut t.train_occ_min, &[0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]),
    ("train_edge_max", |t| &mut t.train_edge_max, &[0.08, 0.12, 0.16, 0.22, 0.28, 0.35, 0.5, 1.0]),
    ("ramp_top_ratio", |t| &mut t.ramp_top_ratio, &[0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]),
    ("barrier_stripe_min", |t| &mut t.barrier_stripe_min, &[0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4]),
    ("high_barrier_occ_min", |t| &mut t.high_barrier_occ_min, &[0.3, 0.4, 0.45, 0.5, 0.6, 0.7]),
    ("overhead_ratio", |t| &mut t.overhead_ratio, &[1.2, 1.5, 1.8, 2.2, 3.0]),
    ("coin_frac_min", |t| &mut t.coin_frac_min, &[0.01, 0.02, 0.03, 0.05, 0.08]),
];

/// Coordinate descent over the classifier thresholds (features fixed).
pub fn fit_classifier(samples: &[ZoneSample], start: Thresholds) -> (Thresholds, f64) {
    let mut best = start;
    let mut best_score = objective(samples, &best);
    for _round in 0..4 {
        let before = best_score;
        for (_, field, candidates) in FIELDS {
            for &c in *candidates {
                let mut t = best;
                *field(&mut t) = c;
                let s = objective(samples, &t);
                if s > best_score + 1e-9 {
                    best_score = s;
                    best = t;
                }
            }
        }
        if best_score <= before + 1e-9 {
            break;
        }
    }
    (best, best_score)
}

pub struct LabelledFrame {
    pub id: u64,
    pub img: RgbImage,
    pub label: FrameLabel,
}

/// Zone samples (sure labels only) for frames labelled `Running`.
pub fn zone_samples(frames: &[&LabelledFrame], calib: &Calibration, work: (u32, u32)) -> Vec<ZoneSample> {
    let mut p = Perceiver::new(calib.clone(), Path::new("."), work);
    let mut out = Vec::new();
    for f in frames {
        let m = p.measure(f.id, &f.img);
        if f.label.game_state != GameState::Running {
            continue;
        }
        for lane in 0..3 {
            for band in 0..3 {
                if let Some(z) = f.label.zone(lane, band).filter(|z| z.sure) {
                    out.push(ZoneSample { features: m.zones[lane][band], truth: z.obstacle, band });
                }
            }
        }
    }
    out
}

/// Game-state and player-lane accuracy, perceiving frames in order (the crash detector
/// needs consecutive frames).
pub fn sequence_accuracy(frames: &[&LabelledFrame], calib: &Calibration, base: &Path, work: (u32, u32)) -> (f64, Option<f64>) {
    let mut p = Perceiver::new(calib.clone(), base, work);
    let (mut s_ok, mut l_ok, mut l_n) = (0usize, 0usize, 0usize);
    for f in frames {
        let obs = p.perceive(f.id, &f.img);
        s_ok += (obs.state == f.label.game_state) as usize;
        if f.label.game_state == GameState::Running && f.label.player_lane != "unknown" {
            l_n += 1;
            let predicted = obs.player_lane.map(|l| format!("{l:?}"));
            l_ok += (predicted.as_deref() == Some(f.label.player_lane.as_str())) as usize;
        }
    }
    (s_ok as f64 / frames.len().max(1) as f64, (l_n > 0).then(|| l_ok as f64 / l_n as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_recovers_a_separating_threshold() {
        // Free zones have occ 0.25, trains 0.9: the default free_occ_max (0.18) misreads Free.
        let free = ZoneFeatures { occ: 0.25, top_occ: 0.25, bottom_occ: 0.25, ..Default::default() };
        let train = ZoneFeatures { occ: 0.9, top_occ: 0.9, bottom_occ: 0.9, edge: 0.1, ..Default::default() };
        let samples: Vec<ZoneSample> = (0..50)
            .flat_map(|i| {
                [
                    ZoneSample { features: free, truth: Obstacle::Free, band: i % 3 },
                    ZoneSample { features: train, truth: Obstacle::TrainBody, band: i % 3 },
                ]
            })
            .collect();
        let before = zone_accuracy(&samples, &Thresholds::default()).overall;
        let (fitted, _) = fit_classifier(&samples, Thresholds::default());
        let after = zone_accuracy(&samples, &fitted);
        assert!(before < 0.6);
        assert_eq!(after.overall, 1.0);
        assert!(after.near == 1.0);
    }

    #[test]
    fn split_is_roughly_a_fifth() {
        let n = (0..10_000u64).filter(|&i| is_test(i)).count();
        assert!((1500..2300).contains(&n), "{n}");
    }
}

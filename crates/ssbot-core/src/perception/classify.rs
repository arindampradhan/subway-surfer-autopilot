//! Zone classifier v1: colour, edge and change heuristics (SPEC §4.3). Features are computed
//! once per zone; the rule set in `classify_zone` only reads features and thresholds, so
//! `ssbot fit` can re-run it cheaply while grid-searching thresholds.

use image::RgbImage;
use serde::{Deserialize, Serialize};

use super::Obstacle;
use super::zones::{Thresholds, ZoneMask};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ZoneFeatures {
    /// Fraction of pixels that aren't track background.
    pub occ: f32,
    pub top_occ: f32,
    pub bottom_occ: f32,
    /// Red or white pixels (barrier stripes).
    pub stripe: f32,
    pub coin: f32,
    pub powerup: f32,
    /// Fraction of pixels on a strong luminance edge.
    pub edge: f32,
    /// Mean absolute change from the previous frame, 0..1.
    pub change: f32,
}

/// RGB (0..255) → hue in degrees (0..360), saturation and value (0..1).
pub fn hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let s = if max == 0.0 { 0.0 } else { d / max };
    (h, s, max)
}

fn luma(p: &image::Rgb<u8>) -> i32 {
    (p[0] as i32 * 299 + p[1] as i32 * 587 + p[2] as i32 * 114) / 1000
}

pub fn is_background(s: f32, v: f32, th: &Thresholds) -> bool {
    s <= th.bg_sat_max && v >= th.bg_val_min && v <= th.bg_val_max
}

pub fn zone_features(img: &RgbImage, prev: Option<&RgbImage>, mask: &ZoneMask, th: &Thresholds) -> ZoneFeatures {
    let w = img.width();
    let raw = img.as_raw();
    let prev = prev.filter(|p| p.dimensions() == img.dimensions());
    let n = mask.pixels.len().max(1) as f32;
    let (mut occ, mut stripe, mut coin, mut powerup, mut edge, mut change) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut top_n, mut top_occ, mut bot_n, mut bot_occ) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(idx, rel_y) in &mask.pixels {
        let (x, y) = (idx % w, idx / w);
        let p = img.get_pixel(x, y);
        let (h, s, v) = hsv(p[0], p[1], p[2]);
        let occupied = !is_background(s, v, th);
        if occupied {
            occ += 1.0;
        }
        if rel_y < 0.5 {
            top_n += 1.0;
            top_occ += occupied as u8 as f32;
        } else {
            bot_n += 1.0;
            bot_occ += occupied as u8 as f32;
        }
        let red = (h < 15.0 || h > 345.0) && s > 0.5 && v > 0.4;
        let white = s < 0.15 && v > 0.85;
        if red || white {
            stripe += 1.0;
        }
        if (40.0..=65.0).contains(&h) && s > 0.5 && v > 0.7 {
            coin += 1.0;
        }
        if h >= th.powerup_hue[0] && h <= th.powerup_hue[1] && s > 0.5 && v > 0.5 {
            powerup += 1.0;
        }
        if x + 1 < w && y + 1 < img.height() {
            let l = luma(p);
            let gx = (luma(img.get_pixel(x + 1, y)) - l).abs();
            let gy = (luma(img.get_pixel(x, y + 1)) - l).abs();
            if gx + gy > 48 {
                edge += 1.0;
            }
        }
        if let Some(prev) = prev {
            let i = idx as usize * 3;
            let q = &prev.as_raw()[i..i + 3];
            let d: i32 = (0..3).map(|c| (raw[i + c] as i32 - q[c] as i32).abs()).sum();
            change += d as f32 / (3.0 * 255.0);
        }
    }
    ZoneFeatures {
        occ: occ / n,
        top_occ: top_occ / top_n.max(1.0),
        bottom_occ: bot_occ / bot_n.max(1.0),
        stripe: stripe / n,
        coin: coin / n,
        powerup: powerup / n,
        edge: edge / n,
        change: change / n,
    }
}

pub fn classify_zone(f: &ZoneFeatures, th: &Thresholds) -> Obstacle {
    if f.occ <= th.free_occ_max {
        return Obstacle::Free;
    }
    if f.stripe >= th.barrier_stripe_min {
        if f.top_occ > f.bottom_occ * th.overhead_ratio {
            return Obstacle::OverheadBar;
        }
        if f.occ >= th.high_barrier_occ_min {
            return Obstacle::HighBarrier;
        }
        return Obstacle::LowBarrier;
    }
    if f.occ >= th.train_occ_min && f.edge <= th.train_edge_max {
        if f.top_occ < f.bottom_occ * th.ramp_top_ratio {
            return Obstacle::TrainRamp;
        }
        return Obstacle::TrainBody;
    }
    // Coins alone make a zone look occupied; a zone that is mostly coins is free.
    if f.coin >= th.coin_frac_min && f.occ - f.coin <= th.free_occ_max {
        return Obstacle::Free;
    }
    Obstacle::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_basics() {
        assert_eq!(hsv(255, 0, 0), (0.0, 1.0, 1.0));
        let (h, s, v) = hsv(0, 0, 255);
        assert_eq!((h, s, v), (240.0, 1.0, 1.0));
        assert_eq!(hsv(128, 128, 128).1, 0.0);
    }

    fn f(occ: f32, top: f32, bottom: f32, stripe: f32, edge: f32) -> ZoneFeatures {
        ZoneFeatures { occ, top_occ: top, bottom_occ: bottom, stripe, edge, ..Default::default() }
    }

    #[test]
    fn rules() {
        let th = Thresholds::default();
        assert_eq!(classify_zone(&f(0.05, 0.05, 0.05, 0.0, 0.0), &th), Obstacle::Free);
        assert_eq!(classify_zone(&f(0.9, 0.9, 0.9, 0.0, 0.1), &th), Obstacle::TrainBody);
        assert_eq!(classify_zone(&f(0.6, 0.3, 0.9, 0.0, 0.1), &th), Obstacle::TrainRamp);
        assert_eq!(classify_zone(&f(0.3, 0.2, 0.4, 0.3, 0.4), &th), Obstacle::LowBarrier);
        assert_eq!(classify_zone(&f(0.6, 0.6, 0.6, 0.3, 0.4), &th), Obstacle::HighBarrier);
        assert_eq!(classify_zone(&f(0.4, 0.7, 0.1, 0.3, 0.4), &th), Obstacle::OverheadBar);
    }
}

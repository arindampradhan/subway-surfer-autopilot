//! Game-state detector (SPEC §4.3, §10.1): screen-marker templates first, then whole-frame
//! fallbacks (black ad-break loader, white video ad, a frozen frame after running).

use image::{GrayImage, RgbImage};

use super::GameState;
use super::classify::hsv;
use super::zones::{MarkerTemplate, Thresholds, crop_gray};

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    pub black: f32,
    pub white: f32,
    /// Mean absolute change from the previous frame, 0..1.
    pub change: f32,
}

pub fn frame_stats(img: &RgbImage, prev: Option<&RgbImage>) -> FrameStats {
    let n = (img.width() * img.height()).max(1) as f32;
    let (mut black, mut white) = (0.0, 0.0);
    for p in img.pixels() {
        let (_, s, v) = hsv(p[0], p[1], p[2]);
        if v < 0.08 {
            black += 1.0;
        } else if v > 0.9 && s < 0.08 {
            white += 1.0;
        }
    }
    let change = match prev.filter(|p| p.dimensions() == img.dimensions()) {
        Some(prev) => {
            let sum: u64 = img.as_raw().iter().zip(prev.as_raw()).map(|(a, b)| a.abs_diff(*b) as u64).sum();
            sum as f32 / (img.as_raw().len() as f32 * 255.0)
        }
        None => 1.0,
    };
    FrameStats { black: black / n, white: white / n, change }
}

/// Mean absolute grey difference (0..255) between a template and the same region of `img`.
pub fn template_diff(img: &RgbImage, t: &MarkerTemplate) -> f32 {
    let crop: GrayImage = crop_gray(img, &t.marker.rect);
    if crop.dimensions() != t.template.dimensions() {
        return f32::INFINITY;
    }
    let n = crop.as_raw().len().max(1) as f32;
    crop.as_raw().iter().zip(t.template.as_raw()).map(|(a, b)| a.abs_diff(*b) as f32).sum::<f32>() / n
}

/// The best-matching marker within its `max_diff`, if any.
pub fn match_marker<'a>(img: &RgbImage, templates: &'a [MarkerTemplate]) -> Option<&'a MarkerTemplate> {
    templates
        .iter()
        .map(|t| (t, template_diff(img, t)))
        .filter(|(t, d)| *d <= t.marker.max_diff)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(t, _)| t)
}

/// Tracks frozen frames across calls, so a run that stops moving reads as `Crashed`.
#[derive(Debug, Default)]
pub struct StateDetector {
    frozen: u32,
    last: Option<GameState>,
}

impl StateDetector {
    /// `track_visible`: whether the lane zones look like track (from the zone classifier).
    pub fn detect(
        &mut self,
        img: &RgbImage,
        stats: FrameStats,
        templates: &[MarkerTemplate],
        track_visible: bool,
        th: &Thresholds,
    ) -> GameState {
        // With a Running marker (the in-run HUD), running must be seen positively; otherwise
        // fall back to "the lane zones look like track".
        let has_running_marker = templates.iter().any(|t| t.marker.state == GameState::Running);
        let matched = match_marker(img, templates).map(|t| t.marker.state);
        let running = match matched {
            Some(GameState::Running) => true,
            Some(_) => false,
            None => !has_running_marker && track_visible && stats.black < th.black_frac_adbreak && stats.white < th.white_frac_ad,
        };
        let state = if running {
            let was_running = matches!(self.last, Some(GameState::Running | GameState::Crashed));
            if was_running && stats.change <= th.frozen_change_max {
                self.frozen += 1;
            } else {
                self.frozen = 0;
            }
            if self.frozen >= th.frozen_frames_crash { GameState::Crashed } else { GameState::Running }
        } else if let Some(s) = matched {
            s
        } else if stats.black >= th.black_frac_adbreak {
            GameState::AdBreak
        } else if stats.white >= th.white_frac_ad {
            GameState::Ad
        } else {
            GameState::Unknown
        };
        if !matches!(state, GameState::Running | GameState::Crashed) {
            self.frozen = 0;
        }
        self.last = Some(state);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::zones::{Marker, Rect};

    #[test]
    fn black_and_white_screens() {
        let th = Thresholds::default();
        let mut d = StateDetector::default();
        let black = RgbImage::new(32, 18);
        let s = frame_stats(&black, None);
        assert_eq!(d.detect(&black, s, &[], false, &th), GameState::AdBreak);
        let white = RgbImage::from_pixel(32, 18, image::Rgb([250, 250, 250]));
        let s = frame_stats(&white, None);
        assert_eq!(d.detect(&white, s, &[], false, &th), GameState::Ad);
    }

    #[test]
    fn frozen_running_becomes_crashed() {
        let th = Thresholds::default();
        let mut d = StateDetector::default();
        let img = RgbImage::from_pixel(32, 18, image::Rgb([120, 110, 100]));
        let first = frame_stats(&img, None);
        assert_eq!(d.detect(&img, first, &[], true, &th), GameState::Running);
        let still = frame_stats(&img, Some(&img));
        let states: Vec<_> = (0..th.frozen_frames_crash).map(|_| d.detect(&img, still, &[], true, &th)).collect();
        assert_eq!(*states.last().unwrap(), GameState::Crashed);
    }

    #[test]
    fn marker_matches_its_own_reference() {
        let mut img = RgbImage::from_pixel(64, 36, image::Rgb([10, 200, 10]));
        img.put_pixel(40, 30, image::Rgb([255, 255, 255]));
        let rect = Rect { x: 0.5, y: 0.7, w: 0.3, h: 0.25 };
        let marker = Marker { state: GameState::ScoreScreen, rect, reference: String::new(), max_diff: 10.0, click: None };
        let t = MarkerTemplate { template: crop_gray(&img, &rect), marker };
        let ts = [t];
        assert_eq!(match_marker(&img, &ts).map(|t| t.marker.state), Some(GameState::ScoreScreen));
        let other = RgbImage::from_pixel(64, 36, image::Rgb([200, 10, 10]));
        assert!(match_marker(&other, &ts).is_none());
    }
}

//! The active power-up, from the bottom-left HUD (SPEC-roofs-and-powerups §4.3). While a power-up
//! runs, the game shows its icon (usually a blue-framed square; the multiplier is a gold star)
//! and a white bar of yellow-green segments that drain. The bar says a power-up may be active;
//! the icon's blue frame, or a close match to a reference icon, confirms it and says which. Both
//! are fixed screen regions, so this costs two small pixel counts and a few template diffs.

use std::path::Path;

use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

use super::PowerUp;
use super::zones::{Rect, crop_gray};

/// A reference icon: `rect` of `HudConfig::icon` cropped from the frame at `reference`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HudIcon {
    pub powerup: PowerUp,
    /// A saved frame showing this power-up's timer, relative to the calibration file.
    pub reference: String,
    /// Mean absolute grey-level difference (0..255) below which the icon matches.
    pub max_diff: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HudConfig {
    /// The timer bar: a white background with yellow-green segments.
    pub bar: Rect,
    /// The power-up icon left of the bar.
    pub icon: Rect,
    /// Fraction of white or segment pixels in `bar` at or above which the timer may be showing.
    pub bar_white_min: f32,
    /// Fraction of the icon's blue frame pixels in `icon` at or above which the icon is showing.
    pub icon_blue_min: f32,
    pub icons: Vec<HudIcon>,
}

impl Default for HudConfig {
    /// Measured on the fullscreen layout (2026-10-15, 640×358 frames): the bar spans x 12..110,
    /// y 331..348; white fills 0.5..0.97 of it with a power-up running and 0 without.
    fn default() -> Self {
        Self {
            bar: Rect { x: 0.06, y: 0.925, w: 0.11, h: 0.05 },
            icon: Rect { x: 0.012, y: 0.92, w: 0.036, h: 0.062 },
            bar_white_min: 0.3,
            icon_blue_min: 0.15,
            icons: Vec::new(),
        }
    }
}

fn is_white(p: &image::Rgb<u8>) -> bool {
    let (lo, hi) = (p.0.iter().min().copied().unwrap_or(0), p.0.iter().max().copied().unwrap_or(0));
    lo > 205 && hi - lo < 35
}

/// A timer segment: bright yellow-green (measured: red 175..238, green 198..253, blue well below
/// green).
fn is_segment(p: &image::Rgb<u8>) -> bool {
    p[1] > 185 && p[0] > 150 && (p[2] as i32) + 40 < p[1] as i32
}

fn is_bar(p: &image::Rgb<u8>) -> bool {
    is_white(p) || is_segment(p)
}

fn is_icon_blue(p: &image::Rgb<u8>) -> bool {
    p[2] > 150 && p[2] as i32 > p[0] as i32 + 50
}

fn fraction(img: &RgbImage, rect: &Rect, test: fn(&image::Rgb<u8>) -> bool) -> f32 {
    let (x0, y0, x1, y1) = rect.pixels(img.width(), img.height());
    let n = ((x1 - x0) * (y1 - y0)).max(1);
    let hits = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| test(img.get_pixel(x, y))).count();
    hits as f32 / n as f32
}

/// Reference icons, with grey templates cut for each frame size seen (the live bot reads the HUD
/// from the full-resolution canvas; tests and tools may pass the 320×180 work image).
#[derive(Debug, Clone)]
pub struct Hud {
    pub cfg: HudConfig,
    references: Vec<(PowerUp, f32, RgbImage)>,
    templates: Option<((u32, u32), Vec<GrayImage>)>,
}

impl Hud {
    pub fn new(cfg: HudConfig, base: &Path) -> Self {
        let references = cfg
            .icons
            .iter()
            .filter_map(|i| match image::open(base.join(&i.reference)) {
                Ok(img) => Some((i.powerup, i.max_diff, img.to_rgb8())),
                Err(err) => {
                    tracing::warn!("HUD icon {:?}: can't open {}: {err}", i.powerup, i.reference);
                    None
                }
            })
            .collect();
        Self { cfg, references, templates: None }
    }

    fn templates(&mut self, size: (u32, u32)) -> &[GrayImage] {
        if self.templates.as_ref().map(|(s, _)| *s) != Some(size) {
            let t = self
                .references
                .iter()
                .map(|(_, _, r)| {
                    let r = image::imageops::resize(r, size.0, size.1, image::imageops::FilterType::Triangle);
                    crop_gray(&r, &self.cfg.icon)
                })
                .collect();
            self.templates = Some((size, t));
        }
        &self.templates.as_ref().unwrap().1
    }

    /// The bar check alone: cheap, and enough to mine candidate frames.
    pub fn bar_showing(&self, img: &RgbImage) -> bool {
        fraction(img, &self.cfg.bar, is_bar) >= self.cfg.bar_white_min
    }

    /// The active power-up: `None` without a timer, `Other` with a blue-framed icon that matches
    /// no reference.
    pub fn detect(&mut self, img: &RgbImage) -> Option<PowerUp> {
        if !self.bar_showing(img) {
            return None;
        }
        let blue = fraction(img, &self.cfg.icon, is_icon_blue) >= self.cfg.icon_blue_min;
        let crop = crop_gray(img, &self.cfg.icon);
        let maxes: Vec<(PowerUp, f32)> = self.references.iter().map(|(p, m, _)| (*p, *m)).collect();
        let best = self
            .templates(img.dimensions())
            .iter()
            .zip(maxes)
            .filter(|(t, _)| t.dimensions() == crop.dimensions())
            .map(|(t, (p, max))| (p, max, mean_abs_diff(&crop, t)))
            .min_by(|a, b| a.2.total_cmp(&b.2));
        match best {
            Some((p, max, d)) if d <= max => Some(p),
            _ if blue => Some(PowerUp::Other),
            // A white patch with no icon frame and no matching icon is scenery.
            _ => None,
        }
    }
}

fn mean_abs_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    let n = a.as_raw().len().max(1);
    a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs()).sum::<u32>() as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    /// A 320×180 frame with the timer painted where the default config looks.
    fn frame(with_timer: bool, icon_shade: u8) -> RgbImage {
        let mut img = RgbImage::from_pixel(320, 180, Rgb([110, 90, 60]));
        if with_timer {
            let cfg = HudConfig::default();
            let (x0, y0, x1, y1) = cfg.bar.pixels(320, 180);
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, Rgb([245, 245, 245]));
                }
            }
            let (x0, y0, x1, y1) = cfg.icon.pixels(320, 180);
            for y in y0..y1 {
                for x in x0..x1 {
                    let edge = x == x0 || y == y0 || x + 1 == x1 || y + 1 == y1;
                    img.put_pixel(x, y, if edge { Rgb([40, 120, 230]) } else { Rgb([icon_shade, icon_shade, 30]) });
                }
            }
        }
        img
    }

    #[test]
    fn no_timer_no_powerup() {
        let mut hud = Hud::new(HudConfig::default(), Path::new("."));
        assert_eq!(hud.detect(&frame(false, 0)), None);
    }

    #[test]
    fn white_scenery_without_the_icon_is_not_a_timer() {
        let mut hud = Hud::new(HudConfig::default(), Path::new("."));
        let img = RgbImage::from_pixel(320, 180, Rgb([240, 240, 240]));
        assert_eq!(hud.detect(&img), None);
    }

    #[test]
    fn timer_without_references_is_other_and_icons_pick_the_closest() {
        let mut hud = Hud::new(HudConfig::default(), Path::new("."));
        assert_eq!(hud.detect(&frame(true, 200)), Some(PowerUp::Other));
        hud.references = vec![(PowerUp::Magnet, 20.0, frame(true, 200)), (PowerUp::Jetpack, 20.0, frame(true, 40))];
        hud.templates = None; // references are fixed at construction outside tests
        assert_eq!(hud.detect(&frame(true, 195)), Some(PowerUp::Magnet));
        assert_eq!(hud.detect(&frame(true, 45)), Some(PowerUp::Jetpack));
        // Far from both references: still a power-up, kind unknown.
        assert_eq!(hud.detect(&frame(true, 120)), Some(PowerUp::Other));
        // Templates follow the frame size: the same icons at full resolution.
        let big = image::imageops::resize(&frame(true, 195), 640, 360, image::imageops::FilterType::Nearest);
        assert_eq!(hud.detect(&big), Some(PowerUp::Magnet));
    }

    #[test]
    fn an_icon_without_the_blue_frame_needs_a_reference() {
        // The multiplier's icon is a gold star: no blue frame.
        let star = |shade| {
            let mut img = frame(true, shade);
            let (x0, y0, x1, y1) = HudConfig::default().icon.pixels(320, 180);
            for y in y0..y1 {
                for x in x0..x1 {
                    img.put_pixel(x, y, Rgb([230, 180, shade]));
                }
            }
            img
        };
        let mut hud = Hud::new(HudConfig::default(), Path::new("."));
        assert_eq!(hud.detect(&star(20)), None, "no reference: bar alone isn't enough");
        hud.references = vec![(PowerUp::Multiplier, 15.0, star(20))];
        hud.templates = None;
        assert_eq!(hud.detect(&star(25)), Some(PowerUp::Multiplier));
    }

    #[test]
    fn a_full_bar_of_segments_counts() {
        let mut img = frame(true, 200);
        let (x0, y0, x1, y1) = HudConfig::default().bar.pixels(320, 180);
        for y in y0..y1 {
            for x in (x0..x1).filter(|x| x % 3 != 0) {
                img.put_pixel(x, y, Rgb([205, 217, 140]));
            }
        }
        let mut hud = Hud::new(HudConfig::default(), Path::new("."));
        assert_eq!(hud.detect(&img), Some(PowerUp::Other));
    }
}

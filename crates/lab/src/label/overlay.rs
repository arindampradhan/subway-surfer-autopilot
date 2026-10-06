//! Zone overlay (SPEC §4.8 step 2): the frame scaled to 640 px wide with each calibrated zone
//! drawn as a thin outline and its ID (`L-near`, …), so Claude labels exactly those zones.
//! Also used by `ssbot calibrate --preview`.

use image::{Rgb, RgbImage};

use ssbot_engine::perception::zones::{Calibration, Quad, Rect, zone_id};

pub const LANE_COLOURS: [Rgb<u8>; 3] = [Rgb([0, 230, 255]), Rgb([255, 60, 255]), Rgb([255, 240, 0])];

/// 3×5 pixel glyphs for the characters zone IDs and marker names use.
fn glyph(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'K' => [0b101, 0b110, 0b100, 0b110, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        _ => [0b000; 5],
    }
}

/// Draws text at scale `s` (each glyph pixel is s×s) on a dark backing box.
pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, colour: Rgb<u8>, s: i32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let tw = text.len() as i32 * 4 * s + s;
    for yy in y - s..y + 6 * s {
        for xx in x - s..x + tw {
            if (0..w).contains(&xx) && (0..h).contains(&yy) {
                let p = img.get_pixel_mut(xx as u32, yy as u32);
                *p = Rgb([p[0] / 4, p[1] / 4, p[2] / 4]);
            }
        }
    }
    for (i, c) in text.chars().enumerate() {
        let g = glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for dy in 0..s {
                        for dx in 0..s {
                            let px = x + i as i32 * 4 * s + col * s + dx;
                            let py = y + row as i32 * s + dy;
                            if (0..w).contains(&px) && (0..h).contains(&py) {
                                img.put_pixel(px as u32, py as u32, colour);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn draw_line(img: &mut RgbImage, (x0, y0): (i32, i32), (x1, y1): (i32, i32), colour: Rgb<u8>) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, colour);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

pub fn draw_quad(img: &mut RgbImage, q: &Quad, colour: Rgb<u8>) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let pts: Vec<(i32, i32)> = q.0.iter().map(|[x, y]| ((x * w).round() as i32, (y * h).round() as i32)).collect();
    for i in 0..4 {
        draw_line(img, pts[i], pts[(i + 1) % 4], colour);
    }
}

pub fn draw_rect(img: &mut RgbImage, r: &Rect, colour: Rgb<u8>) {
    let q = Quad([[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]]);
    draw_quad(img, &q, colour);
}

/// The frame at `width` px wide with the nine zones outlined and named.
pub fn zone_overlay(frame: &RgbImage, calib: &Calibration, width: u32) -> RgbImage {
    let height = (frame.height() as f64 * width as f64 / frame.width() as f64).round() as u32;
    let mut img = image::imageops::resize(frame, width, height, image::imageops::FilterType::Triangle);
    let scale = (width / 320).max(1) as i32;
    for (lane, zones) in calib.lanes.iter().enumerate() {
        for (band, q) in zones.bands().iter().enumerate() {
            draw_quad(&mut img, q, LANE_COLOURS[lane]);
            let (x0, y0, x1, _) = q.bounds();
            let tx = ((x0 + x1) / 2.0 * width as f32) as i32 - 12 * scale;
            let ty = (y0 * height as f32) as i32 + 2 * scale;
            draw_text(&mut img, tx, ty, &zone_id(lane, band), LANE_COLOURS[lane], scale);
        }
    }
    img
}

/// Calibration preview: zones, the player strip and marker boxes.
pub fn calibration_preview(frame: &RgbImage, calib: &Calibration, width: u32) -> RgbImage {
    let mut img = zone_overlay(frame, calib, width);
    let height = img.height() as f32;
    draw_rect(&mut img, &calib.player_strip, Rgb([255, 255, 255]));
    draw_text(&mut img, (calib.player_strip.x * width as f32) as i32 + 2, (calib.player_strip.y * height) as i32 + 2, "player", Rgb([255, 255, 255]), 2);
    for m in &calib.markers {
        draw_rect(&mut img, &m.rect, Rgb([0, 255, 0]));
        let label = format!("{:?}", m.state);
        draw_text(&mut img, (m.rect.x * width as f32) as i32 + 2, (m.rect.y * height) as i32 + 2, &label, Rgb([0, 255, 0]), 2);
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_is_640_wide_and_marks_zones() {
        let frame = RgbImage::from_pixel(320, 180, Rgb([100, 100, 100]));
        let calib = Calibration::default();
        let out = zone_overlay(&frame, &calib, 640);
        assert_eq!(out.dimensions(), (640, 360));
        let coloured = out.pixels().filter(|p| LANE_COLOURS.contains(p)).count();
        assert!(coloured > 500, "{coloured} outline pixels");
    }
}

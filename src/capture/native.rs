//! Backend B: native window capture with `xcap` (ScreenCaptureKit on macOS; SPEC §4.2).
//! Needs the Screen Recording permission. Build with `--features xcap`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use image::DynamicImage;

use super::{Crop, Frame, LatestFrame, prepare};

/// Finds the Chrome window showing the game and captures it on a background thread.
/// `viewport`: the page viewport size, used to skip the window's toolbar at the top;
/// `crop`: the canvas within the viewport (normalised).
pub fn start(viewport: [u32; 2], crop: Crop, max_width: u32, work: (u32, u32), fps: u32) -> Result<LatestFrame> {
    let window = xcap::Window::all()?
        .into_iter()
        .find(|w| {
            let title = w.title().unwrap_or_default().to_lowercase();
            let app = w.app_name().unwrap_or_default().to_lowercase();
            app.contains("chrome") && (title.contains("subway") || title.contains("poki"))
        })
        .context("no Chrome window with the game is open")?;
    let (tx, latest) = LatestFrame::channel();
    std::thread::spawn(move || {
        let period = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
        let mut id = 0u64;
        loop {
            let started = Instant::now();
            let shot = match window.capture_image() {
                Ok(img) => DynamicImage::ImageRgba8(img).to_rgb8(),
                Err(err) => {
                    tracing::warn!("native capture: {err}");
                    std::thread::sleep(period);
                    continue;
                }
            };
            // The viewport is the bottom part of the window; scale for retina captures.
            let (ww, wh) = shot.dimensions();
            let scale = ww as f64 / viewport[0] as f64;
            let vh = (viewport[1] as f64 * scale).min(wh as f64);
            let top = (wh as f64 - vh) / wh as f64;
            let frac_h = vh / wh as f64;
            let canvas_crop = Crop { x: crop.x, y: top + crop.y * frac_h, w: crop.w, h: crop.h * frac_h };
            if let Ok((rgb, canvas)) = prepare(&shot, canvas_crop, work, max_width) {
                id += 1;
                let frame = Frame { id, t_capture: started, rgb, canvas: Arc::new(canvas), delay_ms: None };
                if tx.send(Some(Arc::new(frame))).is_err() {
                    break;
                }
            }
            std::thread::sleep(period.saturating_sub(started.elapsed()));
        }
    });
    Ok(latest)
}

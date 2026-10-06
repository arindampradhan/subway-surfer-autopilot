//! Frame capture (SPEC §4.2). Backends publish into a `watch` channel, so the consumer always
//! gets the newest frame and stale ones are dropped, never queued.

pub mod screencast;
#[cfg(feature = "xcap")]
pub mod native;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use fast_image_resize::images::Image;
use fast_image_resize::{PixelType, ResizeOptions, Resizer};
use image::RgbImage;
use tokio::sync::watch;

/// Canvas crop within a captured image, in normalised coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Crop {
    pub const FULL: Crop = Crop { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub id: u64,
    pub t_capture: Instant,
    /// Working-size image perception runs on (~320×180).
    pub rgb: RgbImage,
    /// The cropped canvas at up to `max_width`, kept for the recorder and labelling.
    pub canvas: Arc<RgbImage>,
    /// How long ago the browser produced the frame when we received it, if known.
    pub delay_ms: Option<f32>,
}

/// Crop then resize with `fast_image_resize` (SIMD), the hot path for every frame.
pub fn crop_resize(src: &RgbImage, crop: Crop, out_w: u32, out_h: u32) -> Result<RgbImage> {
    let (w, h) = src.dimensions();
    let img = Image::from_vec_u8(w, h, src.as_raw().clone(), PixelType::U8x3).context("source image")?;
    let mut dst = Image::new(out_w, out_h, PixelType::U8x3);
    let opts = ResizeOptions::new().crop(crop.x * w as f64, crop.y * h as f64, crop.w * w as f64, crop.h * h as f64);
    Resizer::new().resize(&img, &mut dst, &opts).context("resize")?;
    RgbImage::from_raw(out_w, out_h, dst.into_vec()).context("resized buffer")
}

/// Decoded capture → (working image, canvas image at ≤ `max_width`).
pub fn prepare(decoded: &RgbImage, crop: Crop, work: (u32, u32), max_width: u32) -> Result<(RgbImage, RgbImage)> {
    let cw = (decoded.width() as f64 * crop.w).round().max(1.0) as u32;
    let ch = (decoded.height() as f64 * crop.h).round().max(1.0) as u32;
    let scale = (max_width as f64 / cw as f64).min(1.0);
    let (ow, oh) = (((cw as f64 * scale).round() as u32).max(1), ((ch as f64 * scale).round() as u32).max(1));
    let canvas = crop_resize(decoded, crop, ow, oh)?;
    let rgb = crop_resize(&canvas, Crop::FULL, work.0, work.1)?;
    Ok((rgb, canvas))
}

pub trait FrameSource {
    /// Waits for a frame newer than the last one returned.
    fn next(&mut self) -> impl std::future::Future<Output = Result<Frame>> + Send;
}

/// Receiving side shared by every backend.
pub struct LatestFrame {
    rx: watch::Receiver<Option<Arc<Frame>>>,
}

impl LatestFrame {
    pub fn channel() -> (watch::Sender<Option<Arc<Frame>>>, LatestFrame) {
        let (tx, rx) = watch::channel(None);
        (tx, LatestFrame { rx })
    }
}

impl FrameSource for LatestFrame {
    async fn next(&mut self) -> Result<Frame> {
        loop {
            self.rx.changed().await.context("capture stopped")?;
            if let Some(f) = self.rx.borrow_and_update().as_ref() {
                return Ok((**f).clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_resize_takes_the_right_region() {
        let mut src = RgbImage::from_pixel(200, 100, image::Rgb([0, 0, 0]));
        for y in 0..100 {
            for x in 100..200 {
                src.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        let right = crop_resize(&src, Crop { x: 0.5, y: 0.0, w: 0.5, h: 1.0 }, 10, 10).unwrap();
        assert!(right.pixels().all(|p| p[0] > 200));
        let (work, canvas) = prepare(&src, Crop::FULL, (32, 18), 640).unwrap();
        assert_eq!(work.dimensions(), (32, 18));
        assert_eq!(canvas.dimensions(), (200, 100), "never upscales the canvas");
    }

    #[tokio::test]
    async fn latest_frame_drops_stale() {
        let (tx, mut rx) = LatestFrame::channel();
        let mk = |id| {
            Some(Arc::new(Frame {
                id,
                t_capture: Instant::now(),
                rgb: RgbImage::new(1, 1),
                canvas: Arc::new(RgbImage::new(1, 1)),
                delay_ms: None,
            }))
        };
        tx.send(mk(1)).unwrap();
        tx.send(mk(2)).unwrap();
        tx.send(mk(3)).unwrap();
        assert_eq!(rx.next().await.unwrap().id, 3);
    }
}

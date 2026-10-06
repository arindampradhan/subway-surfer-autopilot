//! Backend A: CDP `Page.startScreencast` (SPEC §4.2). Each frame is acked straight away so
//! Chrome keeps sending, then decoded, cropped to the canvas and published as the newest frame.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use base64::Engine;
use chromiumoxide::cdp::browser_protocol::page::{
    EventScreencastFrame, ScreencastFrameAckParams, StartScreencastFormat, StartScreencastParams,
    StopScreencastParams,
};
use chromiumoxide::page::Page;
use futures::StreamExt;

use super::{Crop, Frame, LatestFrame, prepare};

pub struct Screencast {
    page: Page,
    stop: Arc<AtomicBool>,
}

impl Screencast {
    /// `crop`: the canvas within the viewport (normalised).
    pub async fn start(
        page: &Page,
        crop: Crop,
        jpeg_quality: u8,
        max_width: u32,
        work: (u32, u32),
    ) -> Result<(Screencast, LatestFrame)> {
        let mut events = page.event_listener::<EventScreencastFrame>().await?;
        // Request the whole viewport at a width where the canvas ends up ≤ max_width.
        let capture_width = (max_width as f64 / crop.w).round() as i64;
        let params = StartScreencastParams::builder()
            .format(StartScreencastFormat::Jpeg)
            .quality(jpeg_quality as i64)
            .max_width(capture_width)
            .build();
        page.execute(params).await?;

        let (tx, latest) = LatestFrame::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let ack_page = page.clone();
        tokio::spawn(async move {
            let mut id = 0u64;
            while let Some(ev) = events.next().await {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                let received = Instant::now();
                let _ = ack_page.execute(ScreencastFrameAckParams::new(ev.session_id)).await;
                let delay_ms = ev.metadata.timestamp.as_ref().map(|ts| {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
                    ((now - *ts.inner()) * 1000.0) as f32
                });
                let data: &str = ev.data.as_ref();
                let decoded = match base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(anyhow::Error::from)
                    .and_then(|bytes| Ok(image::load_from_memory(&bytes)?.to_rgb8()))
                {
                    Ok(img) => img,
                    Err(err) => {
                        tracing::warn!("bad screencast frame: {err}");
                        continue;
                    }
                };
                match prepare(&decoded, crop, work, max_width) {
                    Ok((rgb, canvas)) => {
                        id += 1;
                        let frame = Frame { id, t_capture: received, rgb, canvas: Arc::new(canvas), delay_ms };
                        if tx.send(Some(Arc::new(frame))).is_err() {
                            break;
                        }
                    }
                    Err(err) => tracing::warn!("frame prep: {err}"),
                }
            }
        });
        Ok((Screencast { page: page.clone(), stop }, latest))
    }

    pub async fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.page.execute(StopScreencastParams::default()).await;
    }
}

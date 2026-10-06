//! Browser control over CDP (SPEC §4.1): headful Chrome with a fixed viewport and a persistent
//! profile, the game page, cookie consent, the game iframe, canvas focus, keys and clicks.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use chromiumoxide::browser::{Browser, BrowserConfig as CdpConfig};
use chromiumoxide::cdp::browser_protocol::input::{
    DispatchKeyEventParams, DispatchKeyEventType, DispatchMouseEventParams, DispatchMouseEventType, MouseButton,
};
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::page::Page;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::config::{BrowserConfig, expand_home};

/// The game canvas (the game iframe's box) in CSS pixels of the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl CanvasRect {
    pub fn point(&self, norm: [f32; 2]) -> (f64, f64) {
        (self.x + self.w * norm[0] as f64, self.y + self.h * norm[1] as f64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IframeInfo {
    pub src: String,
    pub origin: String,
    pub rect: CanvasRect,
}

pub struct Game {
    browser: Browser,
    pub page: Page,
    handler: JoinHandle<()>,
    pub canvas: CanvasRect,
    pub iframe: Option<IframeInfo>,
    viewport: [u32; 2],
}

/// DOM key, code and Windows virtual key code for the keys the game uses.
fn key_codes(key: &str) -> Result<(&'static str, &'static str, i64)> {
    Ok(match key {
        "ArrowLeft" => ("ArrowLeft", "ArrowLeft", 37),
        "ArrowUp" => ("ArrowUp", "ArrowUp", 38),
        "ArrowRight" => ("ArrowRight", "ArrowRight", 39),
        "ArrowDown" => ("ArrowDown", "ArrowDown", 40),
        "Space" => (" ", "Space", 32),
        other => bail!("unsupported key {other}"),
    })
}

const FIND_IFRAME_JS: &str = r#"(() => {
  const frames = [...document.querySelectorAll('iframe')]
    .map(f => ({ src: f.src, r: f.getBoundingClientRect() }))
    .filter(f => f.src && f.r.width > 200 && f.r.height > 150)
    .sort((a, b) => b.r.width * b.r.height - a.r.width * a.r.height);
  if (!frames.length) return null;
  const f = frames[0];
  let origin = '';
  try { origin = new URL(f.src).origin; } catch (e) {}
  return { src: f.src, origin, rect: { x: f.r.x, y: f.r.y, w: f.r.width, h: f.r.height } };
})()"#;

const ACCEPT_COOKIES_JS: &str = r#"(() => {
  const want = /^(accept|accept all|agree|i agree|allow all|ok|got it|consent)$/i;
  for (const b of document.querySelectorAll('button, [role=button], a')) {
    if (want.test((b.innerText || '').trim())) { b.click(); return true; }
  }
  return false;
})()"#;

/// Makes the game iframe fill the viewport and hides everything else on the page, including
/// Poki's ad slots that would otherwise sit on top of the stretched canvas (SPEC §10.1).
const FILL_VIEWPORT_JS: &str = r#"(() => {
  const frames = [...document.querySelectorAll('iframe')]
    .filter(f => f.src && f.getBoundingClientRect().width > 200)
    .sort((a, b) => b.getBoundingClientRect().width - a.getBoundingClientRect().width);
  if (!frames.length) return false;
  const f = frames[0];
  for (const e of document.querySelectorAll('body *')) {
    if (e !== f && !e.contains(f)) e.style.setProperty('visibility', 'hidden', 'important');
  }
  f.style.cssText += ';visibility:visible!important;position:fixed!important;left:0!important;top:0!important;width:100vw!important;height:100vh!important;z-index:2147483647!important;border:0!important;';
  document.documentElement.style.overflow = 'hidden';
  return true;
})()"#;

/// Injected into every frame of every page the tab opens, including the cross-origin game iframe:
/// records real key presses with their epoch-millisecond times. Iframes forward them to the top
/// window, where `Game::drain_keys` collects them.
const KEY_LOG_JS: &str = r#"(() => {
  if (window.__ssKeyLog) return;
  window.__ssKeyLog = true;
  const push = (m) => { (window.__ssKeys = window.__ssKeys || []).push(m); };
  window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    const m = { __ss: 1, key: e.key, ts: performance.timeOrigin + e.timeStamp };
    if (window === window.top) push(m); else { try { window.top.postMessage(m, '*'); } catch (_) {} }
  }, true);
  if (window === window.top) {
    window.addEventListener('message', (ev) => { if (ev.data && ev.data.__ss) push(ev.data); });
  }
})()"#;

/// A key a human pressed, with the epoch time in milliseconds.
#[derive(Debug, Clone, Deserialize)]
pub struct KeyPress {
    pub key: String,
    pub ts: f64,
}

/// Closes a bot browser left running by an earlier session that was killed (its Chrome keeps
/// the profile locked). Only touches a process whose command line uses this bot profile.
fn close_orphaned_browser(profile: &std::path::Path) {
    let Ok(target) = std::fs::read_link(profile.join("SingletonLock")) else { return };
    let Some(pid) = target.to_string_lossy().rsplit('-').next().and_then(|p| p.parse::<u32>().ok()) else { return };
    let Ok(out) = std::process::Command::new("ps").args(["-p", &pid.to_string(), "-o", "command="]).output() else { return };
    let cmd = String::from_utf8_lossy(&out.stdout);
    if cmd.contains(&format!("--user-data-dir={}", profile.display())) {
        tracing::warn!("closing a bot browser left over from an earlier session (pid {pid})");
        let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
        std::thread::sleep(Duration::from_secs(2));
    }
}

impl Game {
    pub async fn launch(cfg: &BrowserConfig) -> Result<Game> {
        let profile = expand_home(&cfg.profile_dir);
        std::fs::create_dir_all(&profile).with_context(|| format!("creating {}", profile.display()))?;
        close_orphaned_browser(&profile);
        let [w, h] = cfg.viewport;
        let mut builder = CdpConfig::builder()
            .with_head()
            .user_data_dir(&profile)
            .window_size(w, h + 140)
            .viewport(Viewport {
                width: w,
                height: h,
                device_scale_factor: Some(1.0),
                emulating_mobile: false,
                is_landscape: true,
                has_touch: false,
            })
            .args([
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--autoplay-policy=no-user-gesture-required",
            ]);
        if !cfg.chrome.is_empty() {
            builder = builder.chrome_executable(&cfg.chrome);
        }
        let (browser, mut handler) =
            Browser::launch(builder.build().map_err(anyhow::Error::msg)?).await.context("launching Chrome")?;
        let handler = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if let Err(err) = event {
                    tracing::debug!("cdp handler: {err}");
                }
            }
        });
        let page = browser.new_page("about:blank").await?;
        Ok(Game { browser, page, handler, canvas: CanvasRect { x: 0.0, y: 0.0, w: w as f64, h: h as f64 }, iframe: None, viewport: [w, h] })
    }

    async fn eval<T: serde::de::DeserializeOwned>(&self, js: &str) -> Result<T> {
        let v = self.page.evaluate(js).await?.into_value::<T>()?;
        Ok(v)
    }

    /// Startup steps 1–2 and 5 (SPEC §4.1): open the page, accept cookies, find the game
    /// iframe and record its origin, then (optionally) fill the viewport with it.
    /// With `direct`, the iframe's own URL is then opened top-level, so key events never have
    /// to cross into a cross-origin frame.
    pub async fn open(&mut self, cfg: &BrowserConfig) -> Result<()> {
        self.page.goto(cfg.url.as_str()).await.context("opening the game page")?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        let mut cookies_done = false;
        loop {
            if !cookies_done {
                cookies_done = self.eval::<bool>(ACCEPT_COOKIES_JS).await.unwrap_or(false);
                if cookies_done {
                    tracing::info!("accepted cookie dialog");
                }
            }
            if let Ok(Some(info)) = self.eval::<Option<IframeInfo>>(FIND_IFRAME_JS).await {
                tracing::info!("game iframe origin: {} ({:?})", info.origin, info.rect);
                self.canvas = info.rect;
                self.iframe = Some(info);
                break;
            }
            if tokio::time::Instant::now() > deadline {
                bail!("no game iframe after 60 s");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        if cfg.direct {
            let src = self.iframe.as_ref().map(|i| i.src.clone()).unwrap_or_default();
            tracing::info!("direct mode: opening {src} top-level");
            self.page.goto(src.as_str()).await?;
            self.full_canvas();
        } else if cfg.fullscreen {
            if self.eval::<bool>(FILL_VIEWPORT_JS).await.unwrap_or(false) {
                self.full_canvas();
            } else {
                tracing::warn!("couldn't resize the game iframe; keeping the page layout");
            }
        }
        Ok(())
    }

    /// Records the human's key presses from now on (call before `open`, so the page and the
    /// game iframe both load with the logger in place). Collect them with `drain_keys`.
    pub async fn watch_keys(&self) -> Result<()> {
        self.page.evaluate_on_new_document(KEY_LOG_JS).await.context("installing the key logger")?;
        Ok(())
    }

    /// Key presses since the last call, oldest first.
    pub async fn drain_keys(&self) -> Result<Vec<KeyPress>> {
        self.eval("(() => { const k = window.__ssKeys || []; window.__ssKeys = []; return k; })()").await
    }

    fn full_canvas(&mut self) {
        let [w, h] = self.viewport;
        self.canvas = CanvasRect { x: 0.0, y: 0.0, w: w as f64, h: h as f64 };
    }

    /// Startup step 4: click so the game iframe has keyboard focus.
    pub async fn focus(&self) -> Result<()> {
        self.click([0.5, 0.5]).await
    }

    pub async fn click(&self, norm: [f32; 2]) -> Result<()> {
        let (x, y) = self.canvas.point(norm);
        for kind in [DispatchMouseEventType::MousePressed, DispatchMouseEventType::MouseReleased] {
            let ev = DispatchMouseEventParams::builder()
                .r#type(kind)
                .x(x)
                .y(y)
                .button(MouseButton::Left)
                .click_count(1)
                .build()
                .map_err(anyhow::Error::msg)?;
            self.page.execute(ev).await?;
        }
        Ok(())
    }

    pub async fn key(&self, key: &str, down: bool) -> Result<()> {
        let (k, code, vk) = key_codes(key)?;
        let kind = if down { DispatchKeyEventType::RawKeyDown } else { DispatchKeyEventType::KeyUp };
        let ev = DispatchKeyEventParams::builder()
            .r#type(kind)
            .key(k)
            .code(code)
            .windows_virtual_key_code(vk)
            .native_virtual_key_code(vk)
            .build()
            .map_err(anyhow::Error::msg)?;
        self.page.execute(ev).await?;
        Ok(())
    }

    /// Key down now, key up after `hold` (Unity polls key state once per game frame, so an
    /// instant down/up pair can be missed). Returns once the key-down has been dispatched.
    pub async fn press(&self, key: &str, hold: Duration) -> Result<()> {
        self.key(key, true).await?;
        let page = self.page.clone();
        let key = key.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(hold).await;
            let probe = Game::key_up_on(&page, &key).await;
            if let Err(err) = probe {
                tracing::warn!("key up {key}: {err}");
            }
        });
        Ok(())
    }

    async fn key_up_on(page: &Page, key: &str) -> Result<()> {
        let (k, code, vk) = key_codes(key)?;
        let ev = DispatchKeyEventParams::builder()
            .r#type(DispatchKeyEventType::KeyUp)
            .key(k)
            .code(code)
            .windows_virtual_key_code(vk)
            .native_virtual_key_code(vk)
            .build()
            .map_err(anyhow::Error::msg)?;
        page.execute(ev).await?;
        Ok(())
    }

    pub fn viewport(&self) -> [u32; 2] {
        self.viewport
    }

    pub async fn close(mut self) {
        let _ = self.browser.close().await;
        let _ = self.browser.wait().await;
        self.handler.abort();
    }
}

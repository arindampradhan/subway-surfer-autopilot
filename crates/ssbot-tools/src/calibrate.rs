//! `ssbot calibrate` (SPEC §4.3, §4.8 step 1). Two parts:
//! - capture: opens the game and saves one reference frame per screen as you bring it up
//!   (`calibration/<State>.png`); marker templates are cropped from these.
//! - `--preview <dir>`: draws the zones, player strip and marker boxes from `calibration.toml`
//!   onto frames as PNGs, for Claude Code and you to check that zones sit on the tracks.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use ssbot_core::config::Config;
use ssbot_core::perception::GameState;
use ssbot_core::perception::zones::Calibration;
use ssbot_live::browser::Game;
use ssbot_live::capture::{Crop, FrameSource, screencast::Screencast};

use crate::label::overlay::calibration_preview;
use crate::label::select::list_frames;

pub fn preview(frames_dir: &Path, calib_path: &Path, out_dir: &Path, max: usize) -> Result<Vec<PathBuf>> {
    let calib = Calibration::load_or_default(calib_path);
    std::fs::create_dir_all(out_dir)?;
    let frames = list_frames(frames_dir)?;
    let step = (frames.len() / max.max(1)).max(1);
    let mut written = Vec::new();
    for (id, path) in frames.into_iter().step_by(step).take(max) {
        let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
        let out = out_dir.join(format!("{id}.png"));
        calibration_preview(&img, &calib, 960).save(&out)?;
        written.push(out);
    }
    Ok(written)
}

const SCREENS: [GameState; 7] = [
    GameState::Menu,
    GameState::Running,
    GameState::Paused,
    GameState::RevivePrompt,
    GameState::NewHighScore,
    GameState::ScoreScreen,
    GameState::AdBreak,
];

pub async fn capture(cfg: &Config, calib_path: &Path) -> Result<()> {
    let dir = calib_path.parent().unwrap_or(Path::new(".")).join("calibration");
    std::fs::create_dir_all(&dir)?;
    if !calib_path.exists() {
        Calibration::default().save(calib_path)?;
        eprintln!("wrote default {}", calib_path.display());
    }
    let mut game = Game::launch(&cfg.browser).await?;
    game.open(&cfg.browser).await?;
    let [vw, vh] = game.viewport();
    let c = game.canvas;
    let crop = Crop { x: c.x / vw as f64, y: c.y / vh as f64, w: c.w / vw as f64, h: c.h / vh as f64 };
    let (sc, mut frames) = Screencast::start(&game.page, crop, 90, 1280, (320, 180)).await?;

    let stdin = std::io::stdin();
    for state in SCREENS {
        eprint!("Bring the game to {state:?} and press Enter (or type s + Enter to skip): ");
        std::io::stderr().flush()?;
        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        if line.trim() == "s" {
            continue;
        }
        let f = tokio::time::timeout(Duration::from_secs(5), frames.next()).await.context("no frame")??;
        let out = dir.join(format!("{state:?}.png"));
        f.canvas.save(&out)?;
        eprintln!("  saved {}", out.display());
    }
    sc.stop().await;
    game.close().await;
    eprintln!(
        "\nNext (SPEC §4.8 step 1): ask Claude Code to read the frames in {} and propose lane zones, \
         the player strip and marker boxes in {}; check with `ssbot calibrate --preview {}`.",
        dir.display(),
        calib_path.display(),
        dir.display()
    );
    Ok(())
}

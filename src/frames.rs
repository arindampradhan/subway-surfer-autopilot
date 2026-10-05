//! `ssbot frames video.mov` (SPEC §4.8 Inputs): ffmpeg pulls frames at 10 fps, cropped to the
//! game canvas and scaled to 640 px wide, so desktops and notifications never reach labelling.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropPx {
    pub w: u32,
    pub h: u32,
    pub x: u32,
    pub y: u32,
}

impl std::str::FromStr for CropPx {
    type Err = anyhow::Error;
    /// `W:H:X:Y`, the same order as ffmpeg's crop filter.
    fn from_str(s: &str) -> Result<Self> {
        let v: Vec<u32> = s.split(':').map(|p| p.trim().parse()).collect::<Result<_, _>>().context("crop is W:H:X:Y")?;
        let [w, h, x, y] = v[..] else { bail!("crop is W:H:X:Y") };
        Ok(CropPx { w, h, x, y })
    }
}

/// The last `crop=W:H:X:Y` that ffmpeg's `cropdetect` printed. It finds the area that isn't
/// black borders, which in fullscreen mode is the game canvas.
pub fn parse_cropdetect(stderr: &str) -> Option<CropPx> {
    stderr.lines().rev().find_map(|l| l.split("crop=").nth(1)?.split_whitespace().next()?.parse().ok())
}

pub fn detect_crop(video: &Path) -> Result<Option<CropPx>> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-t", "20", "-i"])
        .arg(video)
        .args(["-vf", "fps=2,cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"])
        .output()
        .context("running ffmpeg (is it installed?)")?;
    Ok(parse_cropdetect(&String::from_utf8_lossy(&out.stderr)))
}

/// Extracts frames to `out_dir` as `000001.jpg`, … . Returns the number written.
pub fn extract(video: &Path, out_dir: &Path, crop: Option<CropPx>, fps: u32, width: u32) -> Result<usize> {
    std::fs::create_dir_all(out_dir)?;
    let mut vf = format!("fps={fps}");
    if let Some(c) = crop {
        vf.push_str(&format!(",crop={}:{}:{}:{}", c.w, c.h, c.x, c.y));
    }
    vf.push_str(&format!(",scale={width}:-2"));
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(video)
        .args(["-vf", &vf, "-q:v", "3"])
        .arg(out_dir.join("%06d.jpg"))
        .status()
        .context("running ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg failed ({status})");
    }
    Ok(std::fs::read_dir(out_dir)?.filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "jpg"))).count())
}

pub fn default_out_dir(video: &Path) -> PathBuf {
    let stem = video.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "video".into());
    video.with_file_name(format!("{stem}_frames"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cropdetect_output() {
        let err = "[Parsed_cropdetect_1 @ 0x1] x1:0 x2:3023 ... crop=3024:1700:0:44\n\
                   [Parsed_cropdetect_1 @ 0x1] x1:0 x2:3023 ... crop=2880:1620:72:84\n";
        assert_eq!(parse_cropdetect(err), Some(CropPx { w: 2880, h: 1620, x: 72, y: 84 }));
        assert_eq!(parse_cropdetect("nothing"), None);
        assert_eq!("10:20:3:4".parse::<CropPx>().unwrap(), CropPx { w: 10, h: 20, x: 3, y: 4 });
    }
}

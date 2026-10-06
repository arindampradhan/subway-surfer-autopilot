//! `ssbot import`: turns bounding boxes from a labelling tool (CVAT, Label Studio,
//! makesense.ai, Roboflow) into zone labels. Supports COCO JSON and YOLO text exports.
//!
//! Each zone gets the obstacle whose box covers at least `min_overlap` of it; when several do,
//! the one lowest on screen (closest to the runner) wins, as in the labelling guide. Zones
//! that a box only grazes are marked `sure: false`. A `runner` box sets the player lane, and
//! `coin` / `powerup` boxes set those flags. Every image in the export counts as labelled,
//! so an image with no boxes is a frame of free track.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use ssbot_core::perception::zones::{Calibration, ZoneMask, zone_id};
use ssbot_core::perception::{GameState, Obstacle};

use super::{FrameLabel, ZoneLabel};

/// A box in normalised image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxN {
    pub class: BoxClass,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxClass {
    Obstacle(Obstacle),
    Runner,
    Coin,
    Powerup,
}

/// Maps a tool's class name to a box class, forgiving case, spaces, `_` and `-`.
pub fn parse_class(name: &str) -> Option<BoxClass> {
    let n: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
    Some(match n.as_str() {
        "train" | "trainbody" | "trains" => BoxClass::Obstacle(Obstacle::TrainBody),
        "ramp" | "trainramp" => BoxClass::Obstacle(Obstacle::TrainRamp),
        "lowbarrier" | "low" | "barrierlow" => BoxClass::Obstacle(Obstacle::LowBarrier),
        "highbarrier" | "high" | "barrierhigh" => BoxClass::Obstacle(Obstacle::HighBarrier),
        "overheadbar" | "overhead" | "bar" => BoxClass::Obstacle(Obstacle::OverheadBar),
        "unknown" | "unclear" | "other" => BoxClass::Obstacle(Obstacle::Unknown),
        "runner" | "player" | "jake" => BoxClass::Runner,
        "coin" | "coins" => BoxClass::Coin,
        "powerup" | "power" => BoxClass::Powerup,
        _ => return None,
    })
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: f32,
    height: f32,
}

#[derive(Deserialize)]
struct CocoAnn {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
}

#[derive(Deserialize)]
struct CocoCat {
    id: u64,
    name: String,
}

#[derive(Deserialize)]
struct Coco {
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnn>,
    categories: Vec<CocoCat>,
}

fn basename(p: &str) -> String {
    // Tools prefix file names with folders or upload ids ("images/12.jpg", "a1b2-12.jpg").
    Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| p.to_string())
}

/// Boxes per image file name from a COCO JSON export.
pub fn read_coco(text: &str) -> Result<HashMap<String, Vec<BoxN>>> {
    let coco: Coco = serde_json::from_str(text).context("parsing COCO JSON")?;
    let mut classes = HashMap::new();
    let mut unknown = BTreeSet::new();
    for c in &coco.categories {
        match parse_class(&c.name) {
            Some(k) => {
                classes.insert(c.id, k);
            }
            None => {
                unknown.insert(c.name.clone());
            }
        }
    }
    if !unknown.is_empty() {
        bail!("unknown classes {unknown:?}; use train, ramp, low barrier, high barrier, overhead bar, runner, coin, powerup");
    }
    let images: HashMap<u64, &CocoImage> = coco.images.iter().map(|i| (i.id, i)).collect();
    let mut out: HashMap<String, Vec<BoxN>> = coco.images.iter().map(|i| (basename(&i.file_name), Vec::new())).collect();
    for a in &coco.annotations {
        let (Some(img), Some(class)) = (images.get(&a.image_id), classes.get(&a.category_id)) else { continue };
        let [x, y, w, h] = a.bbox;
        out.entry(basename(&img.file_name)).or_default().push(BoxN {
            class: *class,
            x0: x / img.width,
            y0: y / img.height,
            x1: (x + w) / img.width,
            y1: (y + h) / img.height,
        });
    }
    Ok(out)
}

/// Class names for a YOLO export: `classes.txt` / `obj.names` (one per line) or the `names:`
/// entry of `data.yaml` (inline list, `- name` lines, or `0: name` lines).
pub fn yolo_class_names(dir: &Path) -> Result<Vec<String>> {
    for f in ["classes.txt", "obj.names", "../classes.txt", "../obj.names"] {
        let p = dir.join(f);
        if p.exists() {
            return Ok(std::fs::read_to_string(p)?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect());
        }
    }
    for f in ["data.yaml", "../data.yaml"] {
        let p = dir.join(f);
        if p.exists() {
            return parse_yaml_names(&std::fs::read_to_string(p)?);
        }
    }
    bail!("no classes.txt, obj.names or data.yaml next to the YOLO labels; pass --classes train,ramp,...")
}

fn parse_yaml_names(yaml: &str) -> Result<Vec<String>> {
    let clean = |s: &str| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
    let mut lines = yaml.lines().skip_while(|l| !l.trim_start().starts_with("names:"));
    let first = lines.next().context("data.yaml has no names:")?;
    let rest = first.trim_start().trim_start_matches("names:").trim();
    if rest.starts_with('[') {
        return Ok(rest.trim_matches(|c| c == '[' || c == ']').split(',').map(clean).filter(|s| !s.is_empty()).collect());
    }
    let mut names = Vec::new();
    for l in lines {
        let t = l.trim();
        if !l.starts_with([' ', '\t', '-']) || t.is_empty() {
            break;
        }
        let item = t.strip_prefix('-').map(str::trim).or_else(|| t.split_once(':').map(|(_, v)| v.trim())).unwrap_or(t);
        names.push(clean(item));
    }
    Ok(names)
}

/// Boxes per image stem from a directory of YOLO `.txt` files (`class cx cy w h`, normalised).
pub fn read_yolo(dir: &Path, names: &[String]) -> Result<HashMap<String, Vec<BoxN>>> {
    let classes: Vec<Option<BoxClass>> = names.iter().map(|n| parse_class(n)).collect();
    let unknown: Vec<&String> = names.iter().zip(&classes).filter(|(_, c)| c.is_none()).map(|(n, _)| n).collect();
    if !unknown.is_empty() {
        bail!("unknown classes {unknown:?}; use train, ramp, low barrier, high barrier, overhead bar, runner, coin, powerup");
    }
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "txt") || matches!(path.file_name().and_then(|n| n.to_str()), Some("classes.txt")) {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut boxes = Vec::new();
        for (i, line) in std::fs::read_to_string(&path)?.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: Vec<f32> = line.split_whitespace().map(|t| t.parse()).collect::<Result<_, _>>()
                .with_context(|| format!("{}:{}", path.display(), i + 1))?;
            let [c, cx, cy, w, h] = v[..] else { bail!("{}:{}: expected 5 numbers", path.display(), i + 1) };
            let class = classes.get(c as usize).copied().flatten().with_context(|| format!("{}:{}: class {c} out of range", path.display(), i + 1))?;
            boxes.push(BoxN { class, x0: cx - w / 2.0, y0: cy - h / 2.0, x1: cx + w / 2.0, y1: cy + h / 2.0 });
        }
        out.insert(stem, boxes);
    }
    Ok(out)
}

/// Fraction of a zone's pixels inside a box.
fn coverage(mask: &ZoneMask, w: u32, h: u32, b: &BoxN) -> f32 {
    if mask.pixels.is_empty() {
        return 0.0;
    }
    let inside = mask
        .pixels
        .iter()
        .filter(|(idx, _)| {
            let x = ((idx % w) as f32 + 0.5) / w as f32;
            let y = ((idx / w) as f32 + 0.5) / h as f32;
            x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1
        })
        .count();
    inside as f32 / mask.pixels.len() as f32
}

/// Zone labels for one frame's boxes.
pub fn frame_label(boxes: &[BoxN], calib: &Calibration, min_overlap: f32) -> FrameLabel {
    let (w, h) = (320u32, 180u32);
    let mut zones = Vec::new();
    for (l, lane) in calib.lanes.iter().enumerate() {
        for (b, quad) in lane.bands().iter().enumerate() {
            let mask = quad.mask(w, h);
            let mut best: Option<(f32, Obstacle)> = None; // (box bottom, obstacle)
            let mut grazed = false;
            let (mut coins, mut powerup) = (false, false);
            for bx in boxes {
                let c = coverage(&mask, w, h, bx);
                match bx.class {
                    BoxClass::Obstacle(o) if c >= min_overlap => {
                        if best.is_none_or(|(bottom, _)| bx.y1 > bottom) {
                            best = Some((bx.y1, o));
                        }
                    }
                    BoxClass::Obstacle(_) if c >= 0.05 => grazed = true,
                    BoxClass::Coin if c > 0.0 => coins = true,
                    BoxClass::Powerup if c > 0.0 => powerup = true,
                    _ => {}
                }
            }
            zones.push(ZoneLabel {
                zone: zone_id(l, b),
                obstacle: best.map(|(_, o)| o).unwrap_or(Obstacle::Free),
                coins,
                powerup,
                sure: best.is_some() || !grazed,
            });
        }
    }
    let edges = calib.lane_edges_at_player();
    let player_lane = boxes
        .iter()
        .find(|b| b.class == BoxClass::Runner)
        .map(|b| {
            let x = (b.x0 + b.x1) / 2.0;
            if x < edges[1] { "L" } else if x < edges[2] { "C" } else { "R" }
        })
        .unwrap_or("unknown")
        .to_string();
    FrameLabel { game_state: GameState::Running, player_lane, player_action: "unknown".into(), zones, notes: "imported boxes".into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_names_are_forgiving() {
        assert_eq!(parse_class("Low Barrier"), Some(BoxClass::Obstacle(Obstacle::LowBarrier)));
        assert_eq!(parse_class("train_body"), Some(BoxClass::Obstacle(Obstacle::TrainBody)));
        assert_eq!(parse_class("overhead-bar"), Some(BoxClass::Obstacle(Obstacle::OverheadBar)));
        assert_eq!(parse_class("Runner"), Some(BoxClass::Runner));
        assert_eq!(parse_class("tree"), None);
    }

    #[test]
    fn coco_boxes_become_zone_labels() {
        let calib = Calibration::default();
        // A train covering the whole left lane, a runner in the centre near the bottom.
        let coco = r#"{
          "images": [{"id": 1, "file_name": "images/12.jpg", "width": 640, "height": 360},
                     {"id": 2, "file_name": "13.jpg", "width": 640, "height": 360}],
          "annotations": [{"image_id": 1, "category_id": 7, "bbox": [0, 100, 215, 260]},
                          {"image_id": 1, "category_id": 8, "bbox": [300, 260, 40, 90]}],
          "categories": [{"id": 7, "name": "train"}, {"id": 8, "name": "runner"}]
        }"#;
        let boxes = read_coco(coco).unwrap();
        assert_eq!(boxes["13.jpg"].len(), 0, "listed image without boxes = free track");
        let label = frame_label(&boxes["12.jpg"], &calib, 0.25);
        assert_eq!(label.zone(0, 0).unwrap().obstacle, Obstacle::TrainBody);
        assert_eq!(label.zone(1, 0).unwrap().obstacle, Obstacle::Free);
        assert_eq!(label.player_lane, "C");
        assert_eq!(label.zones.len(), 9);
        let empty = frame_label(&boxes["13.jpg"], &calib, 0.25);
        assert!(empty.zones.iter().all(|z| z.obstacle == Obstacle::Free && z.sure));
    }

    #[test]
    fn lowest_box_wins_and_grazes_are_unsure() {
        let calib = Calibration::default();
        let (x0, y0, x1, y1) = calib.lanes[1].near.bounds();
        let full = |class| BoxN { class, x0: x0 - 0.01, y0: y0 - 0.01, x1: x1 + 0.01, y1: y1 + 0.01 };
        let ramp = full(BoxClass::Obstacle(Obstacle::TrainRamp));
        let train_behind = BoxN { y1: y1 - 0.02, ..full(BoxClass::Obstacle(Obstacle::TrainBody)) };
        let label = frame_label(&[train_behind, ramp], &calib, 0.25);
        assert_eq!(label.zone(1, 0).unwrap().obstacle, Obstacle::TrainRamp);
        // A sliver over the zone's edge: not enough to label, enough to doubt.
        let sliver = BoxN { class: BoxClass::Obstacle(Obstacle::LowBarrier), x0: x0 - 0.05, y0, x1: x0 + (x1 - x0) * 0.15, y1 };
        let z = frame_label(&[sliver], &calib, 0.25).zone(1, 0).cloned().unwrap();
        assert_eq!(z.obstacle, Obstacle::Free);
        assert!(!z.sure);
    }

    #[test]
    fn yolo_and_yaml_names() {
        let dir = std::env::temp_dir().join(format!("ssbot-yolo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("data.yaml"), "path: x\nnames:\n  0: train\n  1: Low Barrier\nnc: 2\n").unwrap();
        std::fs::write(dir.join("000012.txt"), "0 0.2 0.6 0.4 0.5\n1 0.5 0.6 0.1 0.05\n").unwrap();
        std::fs::write(dir.join("000013.txt"), "").unwrap();
        let names = yolo_class_names(&dir).unwrap();
        assert_eq!(names, vec!["train", "Low Barrier"]);
        let boxes = read_yolo(&dir, &names).unwrap();
        assert_eq!(boxes["000012"].len(), 2);
        assert!(boxes["000013"].is_empty());
        assert_eq!(parse_yaml_names("names: ['train', 'ramp']").unwrap(), vec!["train", "ramp"]);
        let _ = std::fs::remove_dir_all(dir);
    }
}

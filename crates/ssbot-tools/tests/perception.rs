//! Perception tests on real frames (SPEC §9).
//!
//! - `session1`: game-state detection on frames from `recordings/session1.mov`, with marker
//!   templates cut from two of those frames, as `ssbot calibrate` does.
//! - `labelled`: when `tests/fixtures/labelled/` holds frames plus `labels.jsonl` (Claude labels,
//!   human spot-checked, held-out split), checks the SPEC §9 accuracy targets.

use std::collections::BTreeMap;
use std::path::Path;

use ssbot_core::perception::zones::{Calibration, Marker, Rect};
use ssbot_core::perception::{GameState, Perceiver};
use ssbot_tools::fit::{LabelledFrame, sequence_accuracy, zone_accuracy, zone_samples};
use ssbot_tools::label::FrameLabel;

const WORK: (u32, u32) = (320, 180);

fn load(path: &Path) -> image::RgbImage {
    let img = image::open(path).unwrap().to_rgb8();
    image::imageops::resize(&img, WORK.0, WORK.1, image::imageops::FilterType::Triangle)
}

fn session1_calibration() -> Calibration {
    let mut c = Calibration::default();
    c.markers = vec![
        Marker {
            state: GameState::NewHighScore,
            rect: Rect { x: 0.36, y: 0.82, w: 0.28, h: 0.06 }, // "Press Space to continue"
            reference: "fixtures/session1/000008.jpg".into(),
            max_diff: 25.0,
            click: None,
        },
        Marker {
            state: GameState::ScoreScreen,
            rect: Rect { x: 0.55, y: 0.80, w: 0.11, h: 0.09 }, // the green PLAY button
            reference: "fixtures/session1/000017.jpg".into(),
            max_diff: 25.0,
            click: Some([0.605, 0.845]),
        },
    ];
    c
}

#[test]
fn session1_game_states() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/session1");
    let expected: BTreeMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("states.json")).unwrap()).unwrap();
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut wrong = Vec::new();
    for (file, want) in expected.iter().filter(|(k, _)| k.ends_with(".jpg")) {
        let mut p = Perceiver::new(session1_calibration(), &base, WORK);
        let m = p.measure(1, &load(&dir.join(file)));
        let got = format!("{:?}", m.obs.state);
        if want == "-" {
            // Other layouts: never a clickable screen.
            if matches!(m.obs.state, GameState::ScoreScreen | GameState::NewHighScore) || m.marker_click.is_some() {
                wrong.push(format!("{file}: matched {got} in the page layout"));
            }
        } else if &got != want {
            wrong.push(format!("{file}: want {want}, got {got}"));
        } else if m.obs.state == GameState::ScoreScreen {
            assert_eq!(m.marker_click, Some([0.605, 0.845]), "PLAY click point");
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn labelled_fixtures_meet_spec_targets() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/labelled");
    let labels_path = dir.join("labels.jsonl");
    if !labels_path.exists() {
        eprintln!("skipped: no tests/fixtures/labelled/labels.jsonl yet (needs Claude labels, SPEC §4.8)");
        return;
    }
    // One line per frame: {"frame": "123.jpg", "label": FrameLabel}
    #[derive(serde::Deserialize)]
    struct Line {
        frame: String,
        label: FrameLabel,
    }
    let calib = Calibration::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../calibration.toml")).unwrap_or_default();
    let frames: Vec<LabelledFrame> = ssbot_tools::label::read_jsonl::<Line>(&labels_path)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, l)| LabelledFrame { id: i as u64, img: load(&dir.join(&l.frame)), label: l.label })
        .collect();
    let refs: Vec<&LabelledFrame> = frames.iter().collect();
    let mut acc = zone_accuracy(&zone_samples(&refs, &calib, WORK), &calib.thresholds);
    let (state, lane) = sequence_accuracy(&refs, &calib, &Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."), WORK);
    acc.game_state = Some(state);
    acc.player_lane = lane;
    assert!(acc.meets_targets(), "{acc:#?}");
}

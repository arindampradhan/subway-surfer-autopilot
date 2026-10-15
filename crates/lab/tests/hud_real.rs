//! The HUD power-up detector on real mined frames (needs `data/mined/hud_icons/check.jsonl`
//! from the 2026-10-15 mining, so it's ignored by default): `cargo test -- --ignored hud`.

use std::path::Path;

use ssbot_engine::perception::PowerUp;
use ssbot_engine::perception::hud::Hud;
use ssbot_engine::perception::zones::Calibration;

#[test]
#[ignore]
fn hud_reads_mined_frames() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let calib = Calibration::load(&root.join("calibration.toml")).unwrap();
    let mut hud = Hud::new(calib.hud.clone(), &root);
    let (mut ok, mut n, mut missed, mut false_alarm, mut wrong_kind) = (0, 0, 0, 0, 0);
    for line in std::fs::read_to_string(root.join("data/mined/hud_icons/check.jsonl")).unwrap().lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let want: Option<PowerUp> = serde_json::from_value(v["powerup"].clone()).unwrap();
        // The saved frame is the live canvas, which is what the bot reads the HUD from.
        let img = image::open(root.join(v["path"].as_str().unwrap())).unwrap().to_rgb8();
        let got = hud.detect(&img);
        n += 1;
        match (want, got) {
            (a, b) if a == b => ok += 1,
            (Some(_), None) => missed += 1,
            (None, Some(p)) => {
                false_alarm += 1;
                eprintln!("  timer read where the old filter saw none: {} ({p:?})", v["path"].as_str().unwrap());
            }
            _ => wrong_kind += 1,
        }
    }
    eprintln!("{ok}/{n} right; missed {missed}, false alarms {false_alarm}, wrong kind {wrong_kind}");
    assert!(ok as f64 / n as f64 >= 0.98, "spec R2 target: active power-up ≥98%");
}

/// The detector against labelled `active_powerup` on the mined sets (SPEC-roofs-and-powerups
/// R2: ≥98%). Frames whose label says `unknown` don't count.
#[test]
#[ignore]
fn hud_matches_labels() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let calib = Calibration::load(&root.join("calibration.toml")).unwrap();
    let mut hud = Hud::new(calib.hud.clone(), &root);
    let (mut ok, mut n) = (0, 0);
    let mut wrong = Vec::new();
    for set in ["powerup_r1", "roofpow_r1"] {
        let Ok(labels) = ssbot_lab::label::load_final_labels(&root.join("labels"), set) else { continue };
        for (id, label) in labels {
            let Some(want) = label.powerup().filter(|_| label.game_state == ssbot_engine::perception::GameState::Running) else { continue };
            let img = image::open(root.join(format!("data/{set}/{id}.jpg"))).unwrap().to_rgb8();
            let got = hud.detect(&img);
            n += 1;
            if got == want {
                ok += 1;
            } else {
                wrong.push(format!("{set}/{id}: label {want:?}, detector {got:?}"));
            }
        }
    }
    eprintln!("{ok}/{n} match the labels");
    for w in wrong.iter().take(15) {
        eprintln!("  {w}");
    }
    assert!(n > 0 && ok as f64 / n as f64 >= 0.98, "{ok}/{n}");
}

/// The checked-in calibration parses with its roof zones and HUD icons: `load_or_default` would
/// otherwise fall back to default zones with only a warning.
#[test]
fn repo_calibration_parses() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let calib = Calibration::load(&root.join("calibration.toml")).unwrap();
    assert!(calib.roof_lanes.is_some());
    assert!(!calib.hud.icons.is_empty());
    for icon in &calib.hud.icons {
        assert!(root.join(&icon.reference).exists(), "{}", icon.reference);
    }
}

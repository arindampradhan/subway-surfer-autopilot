//! Replay regression (SPEC §9): on a fixed set of recorded runs, perception or policy changes
//! must not increase the number of predicted crashes. Put runs under `tests/fixtures/runs/<name>/`
//! (frames/ + events.jsonl) with a `baseline.json` of `{"predicted_crashes": N}`.

use std::path::Path;

use ssbot_engine::config::Config;

#[test]
fn predicted_crashes_do_not_regress() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runs = root.join("tests/fixtures/runs");
    let Ok(entries) = std::fs::read_dir(&runs) else {
        eprintln!("skipped: no tests/fixtures/runs yet (record runs with `ssbot run` first)");
        return;
    };
    let cfg = Config::default();
    for entry in entries.flatten().filter(|e| e.path().join("events.jsonl").exists()) {
        let dir = entry.path();
        let report = ssbot_lab::replay::replay(&dir, &cfg, &root.join("../../calibration.toml"), false).unwrap();
        let baseline: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("baseline.json")).unwrap_or("{}".into())).unwrap();
        if let Some(base) = baseline["predicted_crashes"].as_u64() {
            assert!(report.predicted_crashes as u64 <= base, "{}: {} > baseline {base}", dir.display(), report.predicted_crashes);
        }
    }
}

//! Prints what perception reads from image files: `cargo run --example perceive -- a.jpg b.jpg`.

use std::path::Path;

use ssbot_engine::facts::facts;
use ssbot_engine::perception::Perceiver;
use ssbot_engine::perception::zones::Calibration;

fn main() {
    let calib = Calibration::load_or_default(Path::new("calibration.toml"));
    for (i, path) in std::env::args().skip(1).enumerate() {
        let mut p = Perceiver::new(calib.clone(), Path::new("."), (320, 180));
        let img = image::open(&path).unwrap().to_rgb8();
        let img = image::imageops::resize(&img, 320, 180, image::imageops::FilterType::Triangle);
        let m = p.measure(i as u64, &img);
        if std::env::var_os("MARKERS").is_some() {
            let ts = ssbot_engine::perception::zones::load_marker_templates(&calib, Path::new("."), (320, 180));
            let d: Vec<String> = ts
                .iter()
                .map(|t| format!("{:?}={:.1}", t.marker.state, ssbot_engine::perception::state::template_diff(&img, t)))
                .collect();
            println!("{path}: {:?} | {}", m.obs.state, d.join(" "));
            continue;
        }
        println!("{path}: {:?} lane {:?}", m.obs.state, m.obs.player_lane);
        for (l, lane) in m.zones.iter().enumerate() {
            let v = m.obs.lanes[l];
            println!("  {}  {:?}/{:?}/{:?}  occ {:.2}/{:.2}/{:.2} edge {:.2}/{:.2}/{:.2} stripe {:.2}/{:.2}/{:.2}",
                ["L", "C", "R"][l], v.near, v.mid, v.far,
                lane[0].occ, lane[1].occ, lane[2].occ, lane[0].edge, lane[1].edge, lane[2].edge, lane[0].stripe, lane[1].stripe, lane[2].stripe);
        }
        if let Some(f) = facts(&m.obs) { println!("  facts: {}", f.replace('\n', " ")); }
    }
}

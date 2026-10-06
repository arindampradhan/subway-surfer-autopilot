//! End-to-end check of the OpenJev sidecar through the Rust protocol code. Needs the MLX
//! model cached locally, so it's ignored by default: `cargo test -- --ignored sidecar`.

use std::time::{Duration, Instant};

use ssbot_core::config::AdvisorConfig;
use ssbot_core::facts::facts;
use ssbot_core::perception::{Lane, LaneView, Obstacle, Observation};
use ssbot_core::policy::Action;
use ssbot_core::policy::advisor::{Advisor, options};
use ssbot_core::sidecar::{Reply, Request};
use ssbot_live::sidecar::Sidecar;

fn obs() -> Observation {
    Observation::running(
        1,
        Lane::C,
        [
            LaneView { near: Obstacle::TrainBody, ..LaneView::FREE },
            LaneView { mid: Obstacle::TrainBody, ..LaneView::FREE },
            LaneView { coins: true, ..LaneView::FREE },
        ],
    )
}

#[tokio::test]
#[ignore]
async fn sidecar_decides_and_advisor_caches() {
    // The default python and script paths are relative to the repo root, not this crate.
    std::env::set_current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).unwrap();
    let cfg = AdvisorConfig::default();
    let (sidecar, mut link) = Sidecar::spawn(&cfg).await.expect("sidecar starts");
    eprintln!("ready: {}", sidecar.model);

    let o = obs();
    let premise = facts(&o).unwrap();
    let opts = options(&o);
    let mut times = Vec::new();
    for id in 1..=5u64 {
        // A different premise each time so the sidecar cache doesn't hide the model cost.
        let p = format!("{premise} Step {id}.");
        let t = Instant::now();
        link.requests.send(Request::Decide { id, premise: p, options: opts.clone() }).unwrap();
        let reply = tokio::time::timeout(Duration::from_secs(30), link.replies.recv()).await.unwrap().unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        let Reply::Decision { id: rid, probs, ms } = reply else { panic!("unexpected {reply:?}") };
        assert_eq!(rid, id);
        assert_eq!(probs.len(), opts.len());
        let sum: f32 = probs.values().sum();
        assert!((sum - 1.0).abs() < 0.01, "{probs:?}");
        eprintln!("decision {id}: {probs:?} model {ms} ms");
    }
    times.sort_by(|a, b| a.total_cmp(b));
    eprintln!("round trip p50 {:.0} ms, max {:.0} ms", times[2], times[4]);

    // Through the advisor: first tick sends, a later tick gets the answer, then it's cached.
    let mut adv = Advisor::new(400, Some(link));
    let f = facts(&o).unwrap();
    assert!(adv.tick(&o, Some(&f), Instant::now()).is_none());
    let start = Instant::now();
    let pick = loop {
        if let Some(p) = adv.tick(&o, Some(&f), Instant::now()) {
            break p;
        }
        assert!(start.elapsed() < Duration::from_secs(30));
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    eprintln!("advisor picked {:?} after {:.0} ms", pick.action, start.elapsed().as_secs_f64() * 1000.0);
    assert!(pick.action != Action::Hoverboard);
    let again = adv.tick(&o, Some(&f), Instant::now()).unwrap();
    assert!(again.cached);
    sidecar.shutdown().await;
}

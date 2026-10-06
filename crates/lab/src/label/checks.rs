//! Label checks that don't need hand-labelling everything (SPEC §4.8 step 3): labelling
//! twice, consistency over time, and what actually happened at crashes.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use ssbot_engine::perception::{GameState, Obstacle};

use super::{FrameLabel, LabelRecord};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub frame_id: u64,
    pub reason: String,
}

/// Ground truth from crashes: the player's lane held something deadly (SPEC §4.8 step 3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DangerLabel {
    pub frame_id: u64,
    pub lane: String,
    pub ms_before_crash: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckReport {
    /// Per obstacle class: (agreeing zones, compared zones) between the two labellings.
    pub agreement: BTreeMap<String, (u32, u32)>,
    pub failing_classes: Vec<String>,
    pub flags: Vec<Flag>,
    pub danger: Vec<DangerLabel>,
    pub unsure_zones: Vec<(u64, String)>,
}

impl CheckReport {
    pub fn agreement_rate(&self, class: &str) -> Option<f64> {
        self.agreement.get(class).filter(|(_, n)| *n > 0).map(|(a, n)| *a as f64 / *n as f64)
    }
}

/// Check 1: zones labelled twice must agree ≥90% per class (by the first labelling's class).
pub fn double_label_agreement(records: &[LabelRecord]) -> BTreeMap<String, (u32, u32)> {
    let firsts: HashMap<u64, &FrameLabel> =
        records.iter().filter(|r| !r.repeat).filter_map(|r| Some((r.frame_id, r.label.as_ref()?))).collect();
    let mut out: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for r in records.iter().filter(|r| r.repeat) {
        let (Some(second), Some(first)) = (r.label.as_ref(), firsts.get(&r.frame_id)) else { continue };
        for z in &first.zones {
            let Some(z2) = second.zones.iter().find(|x| x.zone == z.zone) else { continue };
            if !(z.sure && z2.sure) {
                continue;
            }
            let e = out.entry(format!("{:?}", z.obstacle)).or_default();
            e.1 += 1;
            if z.obstacle == z2.obstacle {
                e.0 += 1;
            }
        }
    }
    out
}

fn lane_index(lane: &str) -> Option<usize> {
    ["L", "C", "R"].iter().position(|l| *l == lane)
}

/// Check 2: a hazard labelled `far` should be in `mid` or `near` of the same lane 0.2–0.8 s later.
pub fn time_consistency(seq: &[(u64, f64, &FrameLabel)]) -> Vec<Flag> {
    let mut flags = Vec::new();
    for (i, (id, t, label)) in seq.iter().enumerate() {
        if label.game_state != GameState::Running {
            continue;
        }
        for lane in 0..3 {
            let Some(far) = label.zone(lane, 2).filter(|z| z.sure && z.obstacle.is_hazard()) else { continue };
            let later: Vec<_> = seq[i + 1..]
                .iter()
                .take_while(|(_, t2, _)| t2 - t <= 800.0)
                .filter(|(_, t2, l)| t2 - t >= 200.0 && l.game_state == GameState::Running)
                .collect();
            if later.is_empty() {
                continue; // nothing to compare against (sampling gap)
            }
            let seen = later.iter().any(|(_, _, l)| {
                [0, 1].iter().any(|&b| l.zone(lane, b).is_some_and(|z| z.obstacle == far.obstacle || z.obstacle.is_hazard()))
            });
            if !seen {
                flags.push(Flag {
                    frame_id: *id,
                    reason: format!("{:?} at {}-far never reached mid/near", far.obstacle, ["L", "C", "R"][lane]),
                });
            }
        }
    }
    flags
}

/// Check 3: in the ~0.3–1.0 s before a crash, the player's lane must hold something deadly.
/// Frames labelled `Free` there are flagged; every such frame also yields a danger label.
pub fn crash_check(seq: &[(u64, f64, &FrameLabel)]) -> (Vec<Flag>, Vec<DangerLabel>) {
    let (mut flags, mut danger) = (Vec::new(), Vec::new());
    let crashes: Vec<f64> = seq
        .windows(2)
        .filter(|w| w[1].2.game_state == GameState::Crashed && w[0].2.game_state != GameState::Crashed)
        .map(|w| w[1].1)
        .collect();
    for crash_t in crashes {
        for (id, t, label) in seq.iter().filter(|(_, t, _)| crash_t - t >= 300.0 && crash_t - t <= 1000.0) {
            let Some(lane) = lane_index(&label.player_lane) else { continue };
            danger.push(DangerLabel { frame_id: *id, lane: label.player_lane.clone(), ms_before_crash: crash_t - t });
            let deadly = [0, 1].iter().any(|&b| label.zone(lane, b).is_some_and(|z| z.obstacle.is_hazard() || z.obstacle == Obstacle::Unknown));
            if !deadly {
                flags.push(Flag { frame_id: *id, reason: format!("crash {:.0} ms later but {} lane labelled free", crash_t - t, label.player_lane) });
            }
        }
    }
    (flags, danger)
}

/// Runs every check. `times` maps frame id → ms (from the recorder, or 100 ms per frame for
/// 10 fps video frames).
pub fn run_checks(records: &[LabelRecord], times: &HashMap<u64, f64>) -> CheckReport {
    let mut report = CheckReport { agreement: double_label_agreement(records), ..Default::default() };
    report.failing_classes = report
        .agreement
        .iter()
        .filter(|(_, (a, n))| *n >= 5 && (*a as f64) < 0.9 * *n as f64)
        .map(|(c, _)| c.clone())
        .collect();
    let mut seq: Vec<(u64, f64, &FrameLabel)> = records
        .iter()
        .filter(|r| !r.repeat)
        .filter_map(|r| Some((r.frame_id, *times.get(&r.frame_id).unwrap_or(&(r.frame_id as f64 * 100.0)), r.label.as_ref()?)))
        .collect();
    seq.sort_by(|a, b| a.1.total_cmp(&b.1));
    seq.dedup_by_key(|x| x.0);
    report.flags = time_consistency(&seq);
    let (crash_flags, danger) = crash_check(&seq);
    report.flags.extend(crash_flags);
    report.danger = danger;
    for (id, _, label) in &seq {
        for z in label.zones.iter().filter(|z| !z.sure) {
            report.unsure_zones.push((*id, z.zone.clone()));
        }
    }
    for r in records.iter().filter(|r| r.status != "succeeded") {
        report.flags.push(Flag { frame_id: r.frame_id, reason: format!("{}: {}", r.status, r.error.clone().unwrap_or_default()) });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{ZoneLabel, all_zone_ids};

    fn label(state: GameState, lane: &str, set: &[(&str, Obstacle)]) -> FrameLabel {
        FrameLabel {
            game_state: state,
            player_lane: lane.into(),
            player_action: "running".into(),
            zones: all_zone_ids()
                .into_iter()
                .map(|z| {
                    let o = set.iter().find(|(id, _)| *id == z).map(|(_, o)| *o).unwrap_or(Obstacle::Free);
                    ZoneLabel { zone: z, obstacle: o, coins: false, powerup: false, sure: true }
                })
                .collect(),
            notes: String::new(),
        }
    }

    fn rec(id: u64, repeat: bool, l: FrameLabel) -> LabelRecord {
        LabelRecord {
            custom_id: format!("r__{id}"),
            run: "r".into(),
            frame_id: id,
            frame: format!("{id}.jpg"),
            repeat,
            label: Some(l),
            status: "succeeded".into(),
            error: None,
            model: "claude-opus-5-5".into(),
            request_id: None,
            prompt_version: "x".into(),
            usage: None,
        }
    }

    #[test]
    fn agreement_counts_by_first_class() {
        let a = label(GameState::Running, "C", &[("L-near", Obstacle::TrainBody)]);
        let b = label(GameState::Running, "C", &[("L-near", Obstacle::TrainRamp)]);
        let recs = vec![rec(1, false, a), rec(1, true, b)];
        let ag = double_label_agreement(&recs);
        assert_eq!(ag["TrainBody"], (0, 1));
        assert_eq!(ag["Free"], (8, 8));
    }

    #[test]
    fn crash_with_free_lane_is_flagged_and_gives_danger_labels() {
        let run = label(GameState::Running, "C", &[]);
        let crash = label(GameState::Crashed, "C", &[]);
        let recs: Vec<_> = (0..20).map(|i| rec(i, false, if i < 15 { run.clone() } else { crash.clone() })).collect();
        let times: HashMap<u64, f64> = (0..20).map(|i| (i, i as f64 * 100.0)).collect();
        let report = run_checks(&recs, &times);
        // Crash at 1500 ms: frames at 500..=1200 ms are in the window.
        assert_eq!(report.danger.len(), 8);
        assert!(report.flags.iter().any(|f| f.reason.contains("labelled free")));
    }

    #[test]
    fn far_obstacle_that_vanishes_is_flagged() {
        let far = label(GameState::Running, "C", &[("C-far", Obstacle::TrainBody)]);
        let empty = label(GameState::Running, "C", &[]);
        let near = label(GameState::Running, "C", &[("C-near", Obstacle::TrainBody)]);
        let ok: Vec<(u64, f64, &FrameLabel)> = vec![(1, 0.0, &far), (2, 300.0, &near)];
        assert!(time_consistency(&ok).is_empty());
        let bad: Vec<(u64, f64, &FrameLabel)> = vec![(1, 0.0, &far), (2, 300.0, &empty)];
        assert_eq!(time_consistency(&bad).len(), 1);
    }
}

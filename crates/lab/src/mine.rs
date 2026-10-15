//! Candidate frames for the rare concepts (SPEC-roofs-and-powerups §4.1, step 1): cheap filters
//! over recorded runs' `events.jsonl` plus the HUD timer check, then near-duplicate removal with
//! the labelling selector's perceptual hash. `sidecar/frame_search.py` ranks what this keeps.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use ssbot_engine::perception::hud::{Hud, HudConfig};
use ssbot_engine::perception::{GameState, Obstacle};
use ssbot_runtime::recorder::read_events;

use crate::label::select::list_frames;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Concept {
    /// Airborne for longer than a jump, or most zones reading train: the ground zones on a roof.
    Roof,
    /// The power-up timer is showing.
    Hud,
    /// A ramp in some zone, or the second before a roof candidate.
    Ramp,
    /// A lane with a power-up item.
    PowerupItem,
}

impl Concept {
    /// Keep order when the cap bites: rarest and most needed first.
    pub const PRIORITY: [Concept; 4] = [Concept::Roof, Concept::Hud, Concept::Ramp, Concept::PowerupItem];
}

/// What the filters need from one logged frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameEvent {
    pub frame_id: u64,
    pub t: f64,
    pub running: bool,
    pub airborne: bool,
    /// `[lane][band]`
    pub zones: [[Obstacle; 3]; 3],
    pub powerup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub run: String,
    pub frame_id: u64,
    pub path: String,
    pub t: f64,
    pub concepts: Vec<Concept>,
}

/// Concepts per event from the logs alone (no pixels). `airborne_ms`: a longer airborne streak
/// than a jump lasts means the sprite sits high because the runner is on a roof.
pub fn log_concepts(events: &[FrameEvent], airborne_ms: f64) -> Vec<BTreeSet<Concept>> {
    let mut out = vec![BTreeSet::new(); events.len()];
    let mut streak_start: Option<f64> = None;
    let mut roof_times = Vec::new();
    for (i, e) in events.iter().enumerate() {
        if !e.running {
            streak_start = None;
            continue;
        }
        streak_start = if e.airborne { Some(streak_start.unwrap_or(e.t)) } else { None };
        let trains = e.zones.iter().flatten().filter(|o| **o == Obstacle::TrainBody).count();
        if streak_start.is_some_and(|s| e.t - s > airborne_ms) || trains >= 7 {
            out[i].insert(Concept::Roof);
            roof_times.push(e.t);
        }
        if e.zones.iter().flatten().any(|o| *o == Obstacle::TrainRamp) {
            out[i].insert(Concept::Ramp);
        }
        if e.powerup {
            out[i].insert(Concept::PowerupItem);
        }
    }
    // The climb: the second before each roof frame.
    for (i, e) in events.iter().enumerate() {
        if e.running && !out[i].contains(&Concept::Roof) && roof_times.iter().any(|&r| e.t < r && e.t >= r - 1000.0) {
            out[i].insert(Concept::Ramp);
        }
    }
    out
}

fn run_candidates(dir: &Path, hud: &Hud, airborne_ms: f64, dup_bits: u32) -> Result<Vec<Candidate>> {
    let events: Vec<FrameEvent> = read_events(dir)?
        .into_iter()
        .map(|e| FrameEvent {
            frame_id: e.frame_id,
            t: e.t,
            running: e.obs.state == GameState::Running,
            airborne: e.obs.airborne,
            zones: e.obs.lanes.map(|l| l.bands()),
            powerup: e.obs.lanes.iter().any(|l| l.powerup),
        })
        .collect();
    let concepts = log_concepts(&events, airborne_ms);
    let by_id: std::collections::HashMap<u64, usize> = events.iter().enumerate().map(|(i, e)| (e.frame_id, i)).collect();
    let run = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let mut out = Vec::new();
    let mut last_hash = None;
    for (id, path) in list_frames(&dir.join("frames"))? {
        let Some(&i) = by_id.get(&id) else { continue };
        if !events[i].running {
            continue;
        }
        let Ok(img) = image::open(&path) else { continue };
        let mut c = concepts[i].clone();
        if hud.bar_showing(&img.to_rgb8()) {
            c.insert(Concept::Hud);
        }
        if c.is_empty() {
            continue;
        }
        let hash = image_hasher::HasherConfig::new().hash_size(8, 8).to_hasher().hash_image(&img);
        if last_hash.as_ref().is_some_and(|h: &image_hasher::ImageHash| h.dist(&hash) <= dup_bits) {
            continue;
        }
        last_hash = Some(hash);
        out.push(Candidate { run: run.clone(), frame_id: id, path: path.to_string_lossy().to_string(), t: events[i].t, concepts: c.into_iter().collect() });
    }
    Ok(out)
}

/// Run folders under `sources`: each source is a run folder (has `events.jsonl`) or a folder of them.
pub fn run_dirs(sources: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for s in sources {
        if s.join("events.jsonl").exists() {
            dirs.push(s.clone());
        } else if let Ok(rd) = std::fs::read_dir(s) {
            let mut sub: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("events.jsonl").exists()).collect();
            sub.sort();
            dirs.extend(sub);
        }
    }
    dirs
}

/// Keeps every candidate of the higher-priority concepts first and samples the rest evenly, so
/// at most `cap` remain.
pub fn cap_by_priority(mut all: Vec<Candidate>, cap: usize) -> Vec<Candidate> {
    if all.len() <= cap {
        return all;
    }
    let top = |c: &Candidate| Concept::PRIORITY.iter().position(|p| c.concepts.contains(p)).unwrap_or(usize::MAX);
    all.sort_by_key(|c| top(c));
    let mut kept = Vec::new();
    for p in 0..Concept::PRIORITY.len() {
        let group: Vec<Candidate> = all.iter().filter(|c| top(c) == p).cloned().collect();
        let room = cap - kept.len();
        if group.len() <= room {
            kept.extend(group);
        } else {
            let step = group.len() as f64 / room as f64;
            kept.extend((0..room).map(|k| group[(k as f64 * step) as usize].clone()));
            break;
        }
    }
    kept
}

/// Filters every run in parallel and writes the candidates as JSON lines to `out`. Returns
/// (runs read, runs skipped, candidates written).
pub fn mine_candidates(sources: &[PathBuf], out: &Path, airborne_ms: f64, dup_bits: u32, cap: usize) -> Result<(usize, usize, usize)> {
    let dirs = run_dirs(sources);
    let hud = Hud::new(HudConfig::default(), Path::new("."));
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunks: Vec<&[PathBuf]> = dirs.chunks(dirs.len().div_ceil(threads).max(1)).collect();
    let results: Vec<(Vec<Candidate>, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let hud = &hud;
                s.spawn(move || {
                    let mut found = Vec::new();
                    let mut skipped = 0;
                    for d in *chunk {
                        match run_candidates(d, hud, airborne_ms, dup_bits) {
                            Ok(c) => found.extend(c),
                            Err(err) => {
                                tracing::warn!("{}: {err:#}", d.display());
                                skipped += 1;
                            }
                        }
                    }
                    (found, skipped)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("mining thread")).collect()
    });
    let skipped = results.iter().map(|(_, s)| s).sum();
    let all: Vec<Candidate> = results.into_iter().flat_map(|(c, _)| c).collect();
    let mut counts = std::collections::BTreeMap::new();
    for c in &all {
        for k in &c.concepts {
            *counts.entry(format!("{k:?}")).or_insert(0usize) += 1;
        }
    }
    eprintln!("before the cap: {} candidates {counts:?}", all.len());
    let kept = cap_by_priority(all, cap);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    for c in &kept {
        writeln!(f, "{}", serde_json::to_string(c)?)?;
    }
    Ok((dirs.len() - skipped, skipped, kept.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(i: u64, airborne: bool, zones: [[Obstacle; 3]; 3]) -> FrameEvent {
        FrameEvent { frame_id: i, t: i as f64 * 100.0, running: true, airborne, zones, powerup: false }
    }

    #[test]
    fn long_airborne_streak_is_a_roof_and_the_climb_before_it_a_ramp() {
        let free = [[Obstacle::Free; 3]; 3];
        // Airborne from frame 10 for 1.2 s: frames past 0.8 s into the streak are roof candidates.
        let events: Vec<FrameEvent> = (0..30).map(|i| ev(i, (10..22).contains(&i), free)).collect();
        let c = log_concepts(&events, 800.0);
        assert!(!c[17].contains(&Concept::Roof) && c[19].contains(&Concept::Roof) && c[21].contains(&Concept::Roof));
        assert!(!c[22].contains(&Concept::Roof));
        // The second before the first roof frame (t 1900) is the climb.
        assert!(c[10].contains(&Concept::Ramp) && !c[8].contains(&Concept::Ramp));
    }

    #[test]
    fn a_short_jump_is_not_a_roof_but_a_wall_of_trains_is() {
        let free = [[Obstacle::Free; 3]; 3];
        let events: Vec<FrameEvent> = (0..10).map(|i| ev(i, (2..7).contains(&i), free)).collect();
        assert!(log_concepts(&events, 800.0).iter().all(|c| c.is_empty()));
        let mut trains = [[Obstacle::TrainBody; 3]; 3];
        trains[1][0] = Obstacle::Free;
        trains[1][1] = Obstacle::Free;
        let c = log_concepts(&[ev(0, false, trains)], 800.0);
        assert!(c[0].contains(&Concept::Roof), "7 of 9 train zones");
    }

    #[test]
    fn cap_keeps_higher_priorities_whole() {
        let mk = |i: u64, c: Concept| Candidate { run: "r".into(), frame_id: i, path: String::new(), t: 0.0, concepts: vec![c] };
        let all: Vec<Candidate> = (0..5).map(|i| mk(i, Concept::Roof)).chain((5..25).map(|i| mk(i, Concept::Ramp))).collect();
        let kept = cap_by_priority(all, 10);
        assert_eq!(kept.len(), 10);
        assert_eq!(kept.iter().filter(|c| c.concepts == [Concept::Roof]).count(), 5);
    }
}

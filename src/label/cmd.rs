//! `ssbot label` and `ssbot review` (SPEC §4.8, §7).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;

use super::batch::{Answer, Client, MAX_BATCH_REQUESTS, Outcome, cost_usd, label_params};
use super::checks::run_checks;
use super::select::{FrameInfo, list_frames, phash, select};
use super::{LabelRecord, Usage, labels_path, load_final_labels, overlay, prompt_version, read_jsonl, review, run_name, zone_list};
use crate::perception::zones::Calibration;
use crate::perception::{GameState, Perceiver};
use crate::recorder::read_events;

pub struct LabelOpts {
    pub frames_dir: PathBuf,
    pub model: String,
    pub effort: String,
    pub max: usize,
    pub sync: bool,
    pub dry_run: bool,
    pub labels_dir: PathBuf,
    pub calibration: PathBuf,
    pub guide: PathBuf,
    /// Every n-th selected frame is labelled a second time with the zones reordered.
    pub repeat_every: usize,
}

/// Times and game states per frame: from the recorder's events when this is a run directory,
/// otherwise 10 fps video frames with states from perception.
fn frame_infos(frames_dir: &Path, calib: &Calibration, base: &Path) -> Result<Vec<FrameInfo>> {
    let frames = list_frames(frames_dir)?;
    let events = frames_dir.parent().filter(|p| p.join("events.jsonl").exists()).map(read_events).transpose()?;
    if let Some(events) = events {
        let by_id: HashMap<u64, (f64, GameState)> = events.iter().map(|e| (e.frame_id, (e.t, e.obs.state))).collect();
        return Ok(frames
            .into_iter()
            .map(|(id, path)| {
                let (t, state) = by_id.get(&id).copied().unwrap_or((id as f64 * 33.0, GameState::Unknown));
                FrameInfo { id, path, t, state }
            })
            .collect());
    }
    let mut p = Perceiver::new(calib.clone(), base, (320, 180));
    frames
        .into_iter()
        .map(|(id, path)| {
            let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
            let img = image::imageops::resize(&img, 320, 180, image::imageops::FilterType::Triangle);
            let state = p.perceive(id, &img).state;
            Ok(FrameInfo { id, path, t: id as f64 * 100.0, state })
        })
        .collect()
}

fn overlay_jpeg(path: &Path, calib: &Calibration) -> Result<Vec<u8>> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
    let over = overlay::zone_overlay(&img, calib, 640);
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85).encode_image(&over)?;
    Ok(buf)
}

fn record(run: &str, info: &FrameInfo, custom_id: String, repeat: bool, answer: Answer, model: &str, version: &str) -> LabelRecord {
    let (label, status, error) = match answer.outcome {
        Outcome::Label(l) => (Some(l), "succeeded".to_string(), None),
        Outcome::Refused(e) => (None, "refused".into(), Some(e)),
        Outcome::Invalid(e) => (None, "invalid".into(), Some(e)),
        Outcome::Failed(e) => {
            let status = e.split(':').next().unwrap_or("errored").to_string();
            (None, status, Some(e))
        }
    };
    LabelRecord {
        custom_id,
        run: run.to_string(),
        frame_id: info.id,
        frame: info.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
        repeat,
        label,
        status,
        error,
        model: model.to_string(),
        request_id: answer.request_id,
        prompt_version: version.to_string(),
        usage: answer.usage,
    }
}

pub async fn label(opts: &LabelOpts) -> Result<()> {
    let calib = Calibration::load_or_default(&opts.calibration);
    let base = opts.calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
    let guide = std::fs::read_to_string(&opts.guide).with_context(|| format!("reading {}", opts.guide.display()))?;
    let version = prompt_version(&guide);
    let run = run_name(&opts.frames_dir);
    std::fs::create_dir_all(&opts.labels_dir)?;
    let out_path = labels_path(&opts.labels_dir, &run);

    let infos = frame_infos(&opts.frames_dir, &calib, &base)?;
    let hashes = infos.iter().map(|f| phash(&f.path)).collect::<Result<Vec<_>>>()?;
    let picked = select(&infos, &hashes, opts.max, 6);
    let done: HashSet<(u64, bool)> = if out_path.exists() {
        read_jsonl::<LabelRecord>(&out_path)?.into_iter().filter(|r| r.status == "succeeded").map(|r| (r.frame_id, r.repeat)).collect()
    } else {
        HashSet::new()
    };
    let mut jobs: Vec<(usize, bool)> = Vec::new();
    for (k, (i, _)) in picked.iter().enumerate() {
        jobs.push((*i, false));
        if opts.repeat_every > 0 && k % opts.repeat_every == 0 {
            jobs.push((*i, true));
        }
    }
    jobs.retain(|(i, rep)| !done.contains(&(infos[*i].id, *rep)));
    eprintln!(
        "{run}: {} frames, {} selected, {} requests to send ({} already labelled)",
        infos.len(),
        picked.len(),
        jobs.len(),
        done.len()
    );
    if jobs.is_empty() {
        return check_and_report(&opts.labels_dir, &run, &infos);
    }

    if opts.dry_run {
        let dir = opts.labels_dir.join(format!("{run}.preview"));
        std::fs::create_dir_all(&dir)?;
        for (i, _) in jobs.iter().filter(|(_, rep)| !rep).take(20) {
            std::fs::write(dir.join(format!("{}.jpg", infos[*i].id)), overlay_jpeg(&infos[*i].path, &calib)?)?;
        }
        // Rough estimate (SPEC §4.8 Cost): ~350 fresh input, ~1k cached system, ~600 output tokens.
        let per = Usage { input_tokens: 350, output_tokens: 600, cache_read_input_tokens: 1000, cache_creation_input_tokens: 0 };
        let est = cost_usd(&opts.model, &per, !opts.sync) * jobs.len() as f64;
        eprintln!("dry run: overlays in {}; estimated cost ≈ ${est:.2} (verify with count_tokens)", dir.display());
        return Ok(());
    }

    let client = Client::new()?;
    let mut requests = Vec::new();
    for (i, rep) in &jobs {
        let info = &infos[*i];
        let b64 = base64::engine::general_purpose::STANDARD.encode(overlay_jpeg(&info.path, &calib)?);
        let custom_id = format!("{run}__{}{}", info.id, if *rep { "__r" } else { "" });
        requests.push((custom_id, label_params(&opts.model, &opts.effort, &guide, &b64, &zone_list(*rep)), *i, *rep));
    }

    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path)?;
    let mut usage = Usage::default();
    let mut write = |rec: LabelRecord, usage: &mut Usage| -> Result<()> {
        if let Some(u) = &rec.usage {
            usage.add(u);
        }
        writeln!(out, "{}", serde_json::to_string(&rec)?)?;
        Ok(())
    };
    if opts.sync {
        for (custom_id, params, i, rep) in requests {
            let answer = client.message(params).await?;
            let rec = record(&run, &infos[i], custom_id, rep, answer, &opts.model, &version);
            eprintln!("{} → {}", rec.custom_id, rec.status);
            write(rec, &mut usage)?;
        }
    } else {
        for chunk in requests.chunks(MAX_BATCH_REQUESTS) {
            let pairs: Vec<(String, serde_json::Value)> = chunk.iter().map(|(id, p, _, _)| (id.clone(), p.clone())).collect();
            let batch_id = client.create_batch(&pairs).await?;
            eprintln!("batch {batch_id}: {} requests submitted; polling (most finish within an hour)", pairs.len());
            let mut results = client.wait_results(&batch_id, Duration::from_secs(30)).await?;
            for (custom_id, _, i, rep) in chunk {
                let answer = results.remove(custom_id).unwrap_or(Answer {
                    outcome: Outcome::Failed("missing: no result for this custom_id".into()),
                    request_id: None,
                    usage: None,
                });
                write(record(&run, &infos[*i], custom_id.clone(), *rep, answer, &opts.model, &version), &mut usage)?;
            }
        }
    }
    let cost = cost_usd(&opts.model, &usage, !opts.sync);
    eprintln!(
        "usage: {} in, {} out, {} cache read, {} cache write → ${cost:.2}{}",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        if opts.sync { "" } else { " (batch price)" }
    );
    if usage.cache_read_input_tokens == 0 && jobs.len() > 2 {
        eprintln!("warning: no cache reads — the system prompt may be below the model's minimum cacheable length");
    }
    check_and_report(&opts.labels_dir, &run, &infos)
}

fn check_and_report(labels_dir: &Path, run: &str, infos: &[FrameInfo]) -> Result<()> {
    let records: Vec<LabelRecord> = read_jsonl(&labels_path(labels_dir, run))?;
    let times: HashMap<u64, f64> = infos.iter().map(|f| (f.id, f.t)).collect();
    let report = run_checks(&records, &times);
    std::fs::write(labels_dir.join(format!("{run}.checks.json")), serde_json::to_string_pretty(&report)?)?;
    let mut danger = String::new();
    for d in &report.danger {
        danger.push_str(&serde_json::to_string(d)?);
        danger.push('\n');
    }
    std::fs::write(labels_dir.join(format!("{run}.danger.jsonl")), danger)?;
    eprintln!("labelling-twice agreement per class:");
    for (class, (ok, n)) in &report.agreement {
        eprintln!("  {class:12} {ok}/{n} ({:.0}%)", 100.0 * *ok as f64 / (*n).max(1) as f64);
    }
    if !report.failing_classes.is_empty() {
        eprintln!("below 90%: {} — fix prompts/label_guide.md before continuing", report.failing_classes.join(", "));
    }
    eprintln!("{} frames flagged, {} unsure zones, {} danger labels; see {run}.checks.json", report.flags.len(), report.unsure_zones.len(), report.danger.len());
    Ok(())
}

/// Perception's reading of a frame as a starting label for manual labelling.
fn prefill(p: &mut Perceiver, id: u64, path: &Path) -> Result<super::FrameLabel> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
    let img = image::imageops::resize(&img, 320, 180, image::imageops::FilterType::Triangle);
    let obs = p.perceive(id, &img);
    let zones = (0..3)
        .flat_map(|l| (0..3).map(move |b| (l, b)))
        .map(|(l, b)| super::ZoneLabel {
            zone: crate::perception::zones::zone_id(l, b),
            obstacle: obs.lanes[l].bands()[b],
            coins: obs.lanes[l].coins,
            powerup: obs.lanes[l].powerup,
            sure: true,
        })
        .collect();
    Ok(super::FrameLabel {
        game_state: obs.state,
        player_lane: obs.player_lane.map(|l| format!("{l:?}")).unwrap_or_else(|| "unknown".into()),
        player_action: if obs.airborne { "jumping" } else if obs.rolling { "rolling" } else { "running" }.into(),
        zones,
        notes: String::new(),
    })
}

/// `manual`: label from scratch. Frames without a label start from perception's guess, and
/// the gallery downloads `<run>.jsonl` instead of fixes.
pub fn review(frames_dir: &Path, labels_dir: &Path, calibration: &Path, manual: bool) -> Result<PathBuf> {
    let run = run_name(frames_dir);
    let path = labels_path(labels_dir, &run);
    let records: Vec<LabelRecord> = if path.exists() { read_jsonl(&path)? } else { Vec::new() };
    if records.is_empty() && !manual {
        bail!("no labels at {}; run `ssbot label` first, or label by hand with --manual", path.display());
    }
    let mut labels = if path.exists() { load_final_labels(labels_dir, &run)? } else { HashMap::new() };
    let list = list_frames(frames_dir)?;
    let calib = Calibration::load_or_default(calibration);
    if manual {
        let base = calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut p = Perceiver::new(calib.clone(), &base, (320, 180));
        for (id, fpath) in &list {
            if !labels.contains_key(id) {
                labels.insert(*id, prefill(&mut p, *id, fpath)?);
            }
        }
    }
    let frames: Vec<(u64, String)> =
        list.into_iter().map(|(id, p)| (id, p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())).collect();
    let times: HashMap<u64, f64> = frames.iter().map(|(id, _)| (*id, *id as f64 * 100.0)).collect();
    let report = run_checks(&records, &times);
    let html = review::render(&run, &frames, &labels, &report.flags, &calib, manual);
    let out = frames_dir.join("review.html");
    review::write(&out, &html)?;
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BoxFormat {
    Coco,
    Yolo,
}

/// `ssbot import`: bounding boxes from a labelling tool → `labels/<run>.jsonl`.
/// `annotations`: a COCO JSON file, or a directory of YOLO `.txt` files.
pub fn import(
    frames_dir: &Path,
    annotations: &Path,
    format: Option<BoxFormat>,
    classes: Option<Vec<String>>,
    labels_dir: &Path,
    calibration: &Path,
    min_overlap: f32,
    force: bool,
) -> Result<(usize, usize)> {
    use super::boxes::{frame_label, read_coco, read_yolo, yolo_class_names};
    let format = format.unwrap_or(if annotations.is_dir() { BoxFormat::Yolo } else { BoxFormat::Coco });
    let boxes = match format {
        BoxFormat::Coco => read_coco(&std::fs::read_to_string(annotations).with_context(|| format!("reading {}", annotations.display()))?)?,
        BoxFormat::Yolo => {
            let names = match classes {
                Some(c) => c,
                None => yolo_class_names(annotations)?,
            };
            read_yolo(annotations, &names)?
        }
    };
    let calib = Calibration::load_or_default(calibration);
    let run = run_name(frames_dir);
    std::fs::create_dir_all(labels_dir)?;
    let out = labels_path(labels_dir, &run);
    if out.exists() && !force {
        bail!("{} exists; pass --force to replace it (the old file is kept as .bak)", out.display());
    }
    if out.exists() {
        std::fs::rename(&out, out.with_extension("jsonl.bak"))?;
    }
    let mut file = std::fs::File::create(&out)?;
    let (mut written, mut frames) = (0, 0);
    for (id, path) in list_frames(frames_dir)? {
        frames += 1;
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let stem = path.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let Some(b) = boxes.get(&name).or_else(|| boxes.get(&stem)) else { continue };
        let rec = LabelRecord {
            custom_id: format!("{run}__{id}"),
            run: run.clone(),
            frame_id: id,
            frame: name,
            repeat: false,
            label: Some(frame_label(b, &calib, min_overlap)),
            status: "succeeded".into(),
            error: None,
            model: "human".into(),
            request_id: None,
            prompt_version: format!("import:{format:?}").to_lowercase(),
            usage: None,
        };
        writeln!(file, "{}", serde_json::to_string(&rec)?)?;
        written += 1;
    }
    if written == 0 {
        bail!("none of the {} annotated images matched a frame in {} (names must match)", boxes.len(), frames_dir.display());
    }
    Ok((written, frames))
}

/// `ssbot select`: copies a varied subset of frames (state changes, pre-crash, then evenly
/// sampled after dropping near-duplicates; SPEC §4.8) into `out`, numbered in order, for
/// labelling by hand or with Claude.
pub fn select_frames(frames_dir: &Path, out: &Path, max: usize, calibration: &Path) -> Result<usize> {
    let calib = Calibration::load_or_default(calibration);
    let base = calibration.parent().unwrap_or(Path::new(".")).to_path_buf();
    let infos = frame_infos(frames_dir, &calib, &base)?;
    let hashes = infos.iter().map(|f| phash(&f.path)).collect::<Result<Vec<_>>>()?;
    let picked = select(&infos, &hashes, max, 6);
    std::fs::create_dir_all(out)?;
    for (i, _) in &picked {
        let f = &infos[*i];
        let ext = f.path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_else(|| "jpg".into());
        std::fs::copy(&f.path, out.join(format!("{:06}.{ext}", f.id)))?;
    }
    Ok(picked.len())
}

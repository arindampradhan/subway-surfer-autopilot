//! Training pipelines: the learned zone classifier (`ssbot train-zones`) and the end-to-end
//! "train by human" flow (`ssbot train-by-human`): a recording of a person playing → frames →
//! a varied selection → labels → zone classifier → OpenJev decision head.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::config::Config;
use crate::label::select::list_frames;
use crate::label::{labels_path, load_final_labels, run_name};
use crate::perception::fit::LabelledFrame;
use crate::perception::model::ZoneModel;
use crate::perception::zones::Calibration;
use crate::perception::{GameState, Obstacle, Perceiver};
use crate::policy::dataset::{self, obs_from_label};
use crate::policy::reflex::Reflex;

/// Labelled frames from several directories, resized to the working size. Frame ids are
/// offset per directory so runs that reuse ids stay distinct.
pub fn load_labelled(dirs: &[PathBuf], labels_dir: &Path, work: (u32, u32)) -> Result<Vec<LabelledFrame>> {
    let mut out = Vec::new();
    for (d, dir) in dirs.iter().enumerate() {
        let labels = load_final_labels(labels_dir, &run_name(dir))?;
        for (id, path) in list_frames(dir)? {
            let Some(label) = labels.get(&id) else { continue };
            let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
            // A work width of 0 means keep the frame at its recorded resolution.
            let img = if work.0 == 0 { img } else { image::imageops::resize(&img, work.0, work.1, image::imageops::FilterType::Triangle) };
            out.push(LabelledFrame { id: d as u64 * 10_000_000 + id, img, label: label.clone() });
        }
    }
    Ok(out)
}

/// Every directory under `data_dir` that has labels, except the advisor datasets and the full
/// (unselected) frame dumps.
pub fn labelled_dirs(data_dir: &Path, labels_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            !name.starts_with("advisor") && !name.ends_with("_all") && labels_path(labels_dir, &run_name(p)).exists()
        })
        .collect();
    dirs.sort();
    Ok(dirs)
}

#[derive(Debug, Default, Serialize)]
pub struct ZoneReport {
    pub frames: usize,
    pub zones: usize,
    pub folds: usize,
    /// Cross-validated (grouped by frame) zone accuracy.
    pub accuracy: f64,
    pub near: f64,
    pub per_class: BTreeMap<String, (usize, usize)>,
    /// Held-out frames whose facts sentence for OpenJev matches the labelled scene exactly,
    /// and whose implied reflex move matches.
    pub facts_exact: (usize, usize),
    pub same_move: (usize, usize),
    pub facts_misses: Vec<String>,
}

/// Trains the zone classifier on every sure zone of every labelled running frame, reports
/// cross-validated accuracy and facts correctness, and with `write` saves the model and
/// points the calibration at it.
pub fn train_zones(
    cfg: &Config,
    calib_path: &Path,
    frames_dirs: &[PathBuf],
    labels_dir: &Path,
    out: &Path,
    epochs: usize,
    l2: f32,
    write: bool,
) -> Result<ZoneReport> {
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let mut calib = Calibration::load_or_default(calib_path);
    calib.zone_model = None;
    let base = calib_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let frames = load_labelled(frames_dirs, labels_dir, work)?;
    let mut p = Perceiver::new(calib.clone(), &base, work);
    // (frame, features, truth, band) for sure zones on running frames.
    let mut samples = Vec::new();
    let mut vectors = BTreeMap::new();
    for f in frames.iter().filter(|f| f.label.game_state == GameState::Running) {
        let v = p.zone_vectors(&f.img);
        for lane in 0..3 {
            for band in 0..3 {
                if let Some(z) = f.label.zone(lane, band).filter(|z| z.sure) {
                    samples.push((f.id, v[lane][band].clone(), z.obstacle, band));
                }
            }
        }
        vectors.insert(f.id, v);
    }
    if samples.is_empty() {
        bail!("no sure zone labels on Running frames");
    }
    let ids: Vec<u64> = samples.iter().map(|s| s.0).collect::<BTreeSet<_>>().into_iter().collect();
    // Leave-one-frame-out for small sets, otherwise 10 folds grouped by frame.
    let folds = ids.len().min(10);
    let fold_of = |id: u64| ids.iter().position(|x| *x == id).unwrap() % folds;
    let reflex = Reflex::new(cfg.policy.clone());
    let mut r = ZoneReport { frames: ids.len(), zones: samples.len(), folds, ..Default::default() };
    let (mut ok, mut near_ok, mut near_n) = (0usize, 0usize, 0usize);
    for fold in 0..folds {
        let train: Vec<_> = samples.iter().filter(|s| fold_of(s.0) != fold).map(|s| (s.1.clone(), s.2)).collect();
        let m = ZoneModel::train(&train, epochs, l2);
        for s in samples.iter().filter(|s| fold_of(s.0) == fold) {
            let hit = m.predict(&s.1).0 == s.2;
            ok += hit as usize;
            if s.3 == 0 {
                near_n += 1;
                near_ok += hit as usize;
            }
            let e = r.per_class.entry(format!("{:?}", s.2)).or_default();
            e.1 += 1;
            e.0 += hit as usize;
        }
        // Facts check on this fold's frames: what OpenJev would be told vs the labelled scene.
        for f in frames.iter().filter(|f| ids.contains(&f.id) && fold_of(f.id) == fold) {
            let Some(truth) = obs_from_label(f.id, &f.label) else { continue };
            let v = &vectors[&f.id];
            let mut seen = truth.clone();
            for lane in 0..3 {
                let pred = [0, 1, 2].map(|b| m.predict(&v[lane][b]).0);
                (seen.lanes[lane].near, seen.lanes[lane].mid, seen.lanes[lane].far) = (pred[0], pred[1], pred[2]);
            }
            let same_facts = crate::facts::facts(&seen) == crate::facts::facts(&truth);
            let (ma, mb) = (reflex.evaluate(&seen).action, reflex.evaluate(&truth).action);
            r.facts_exact.1 += 1;
            r.same_move.1 += 1;
            r.facts_exact.0 += same_facts as usize;
            r.same_move.0 += (ma == mb) as usize;
            if ma != mb {
                r.facts_misses.push(format!("frame {}: move {ma:?} instead of {mb:?}", f.id));
            }
        }
    }
    r.accuracy = ok as f64 / samples.len() as f64;
    r.near = near_ok as f64 / near_n.max(1) as f64;
    if write {
        let all: Vec<_> = samples.iter().map(|s| (s.1.clone(), s.2)).collect();
        let mut m = ZoneModel::train(&all, epochs, l2);
        m.note = format!(
            "{} zones from {} frames; {folds}-fold accuracy {:.3}, near {:.3}",
            samples.len(),
            ids.len(),
            r.accuracy,
            r.near
        );
        m.save(&base.join(out))?;
        calib.zone_model = Some(out.to_string_lossy().to_string());
        calib.save(calib_path)?;
    }
    Ok(r)
}

#[derive(Debug, Default, Serialize)]
pub struct EvalReport {
    pub frames: usize,
    pub zones: usize,
    pub accuracy: f64,
    /// truth -> predicted -> count
    pub confusion: BTreeMap<String, BTreeMap<String, usize>>,
    /// Zones labelled with a hazard (train body, barrier, overhead bar) that the model called free.
    pub missed_hazards: (usize, usize),
    /// Zones labelled free that the model called a hazard (these make the bot dodge for nothing).
    pub false_alarms: (usize, usize),
    /// Barrier zones (low, high, overhead) read as exactly the right barrier type: the class
    /// decides jump versus roll.
    pub barrier_recall: (usize, usize),
}

/// Scores a saved zone model on labelled frames it may not have trained on. Only sure zones of
/// running frames count.
pub fn eval_zones(cfg: &Config, calib_path: &Path, model_path: &Path, frames_dirs: &[PathBuf], labels_dir: &Path) -> Result<EvalReport> {
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let calib = Calibration::load_or_default(calib_path);
    let model = crate::perception::cnn::Classifier::load(model_path).with_context(|| format!("loading {}", model_path.display()))?;
    // A model trained on full-resolution crops is scored on full-resolution frames.
    let frames = load_labelled(frames_dirs, labels_dir, if model.hires() { (0, 0) } else { work })?;
    let mut r = EvalReport::default();
    let mut ok = 0;
    for f in frames.iter().filter(|f| f.label.game_state == GameState::Running) {
        let mut any = false;
        let masks = calib.lanes.map(|lane| lane.bands().map(|q| q.mask(f.img.width(), f.img.height())));
        for lane in 0..3 {
            for band in 0..3 {
                let Some(z) = f.label.zone(lane, band).filter(|z| z.sure) else { continue };
                any = true;
                let pred = model.predict(&f.img, &masks[lane][band], band);
                r.zones += 1;
                ok += (pred == z.obstacle) as usize;
                *r.confusion.entry(format!("{:?}", z.obstacle)).or_default().entry(format!("{pred:?}")).or_default() += 1;
                if matches!(z.obstacle, Obstacle::LowBarrier | Obstacle::HighBarrier | Obstacle::OverheadBar) {
                    r.barrier_recall.1 += 1;
                    r.barrier_recall.0 += (pred == z.obstacle) as usize;
                }
                if z.obstacle.is_hazard() {
                    r.missed_hazards.1 += 1;
                    r.missed_hazards.0 += (pred == Obstacle::Free) as usize;
                } else if z.obstacle == Obstacle::Free {
                    r.false_alarms.1 += 1;
                    r.false_alarms.0 += pred.is_hazard() as usize;
                }
            }
        }
        r.frames += any as usize;
    }
    r.accuracy = ok as f64 / r.zones.max(1) as f64;
    Ok(r)
}

/// Writes every sure zone of every labelled running frame as a CNN crop, for
/// `sidecar/zone_cnn.py`: `crops.bin` (N × CROP × CROP × 3 bytes) and `meta.jsonl` (one line per
/// crop: set, frame id, zone, class). The crops come from the same code the live perceiver
/// uses, so what the CNN trains on is what it sees while playing.
pub fn dump_zone_crops(cfg: &Config, calib_path: &Path, frames_dirs: &[PathBuf], labels_dir: &Path, out: &Path, native: bool, crop: usize, ctx: f32, dual: bool, ctx_far: Option<f32>) -> Result<usize> {
    use std::io::Write;
    let work = (cfg.capture.work_size[0], cfg.capture.work_size[1]);
    let calib = Calibration::load_or_default(calib_path);
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("crops.json"), serde_json::json!({"crop": crop, "native": native, "ctx": ctx, "dual": dual, "ctx_far": ctx_far}).to_string())?;
    let mut bin = std::io::BufWriter::new(std::fs::File::create(out.join("crops.bin"))?);
    let mut meta = std::io::BufWriter::new(std::fs::File::create(out.join("meta.jsonl"))?);
    let mut n = 0;
    for dir in frames_dirs {
        let set = run_name(dir);
        let labels = load_final_labels(labels_dir, &set)?;
        for (id, path) in list_frames(dir)? {
            let Some(label) = labels.get(&id).filter(|l| l.game_state == GameState::Running) else { continue };
            let img = image::open(&path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
            // Native: crop from the frame as recorded (the live canvas, 640 wide) instead of the
            // small work image, so far barriers keep their detail.
            let img = if native { img } else { image::imageops::resize(&img, work.0, work.1, image::imageops::FilterType::Triangle) };
            let masks = calib.lanes.map(|lane| lane.bands().map(|q| q.mask(img.width(), img.height())));
            for lane in 0..3 {
                for band in 0..3 {
                    let Some(z) = label.zone(lane, band).filter(|z| z.sure && z.obstacle != Obstacle::Unknown) else { continue };
                    let c = if band == 2 { ctx_far.unwrap_or(ctx) } else { ctx };
                    bin.write_all(&crate::perception::cnn::zone_input(&img, &masks[lane][band], crop, c, dual))?;
                    writeln!(meta, "{}", serde_json::json!({"set": set, "id": id, "lane": lane, "band": band, "class": format!("{:?}", z.obstacle)}))?;
                    n += 1;
                }
            }
        }
    }
    Ok(n)
}

pub enum Labeller {
    /// Claude if credentials exist, otherwise the manual gallery.
    Auto,
    Claude,
    Manual,
    /// A COCO JSON file or YOLO directory from a box-labelling tool.
    Import(PathBuf),
}

pub struct HumanOpts {
    pub video: PathBuf,
    pub name: Option<String>,
    pub max_frames: usize,
    pub labeller: Labeller,
    pub situations: usize,
    pub train_head: bool,
}

fn has_claude_credentials() -> bool {
    std::env::var("ANTHROPIC_API_KEY").is_ok_and(|k| !k.is_empty())
        || std::env::var("ANTHROPIC_AUTH_TOKEN").is_ok_and(|k| !k.is_empty())
        || Command::new("ant").args(["auth", "status"]).output().is_ok_and(|o| o.status.success())
}

fn step(n: usize, text: &str) {
    eprintln!("\n== {n}. {text}");
}

/// Waits for `labels/<name>.jsonl`, also picking it up from ~/Downloads, where the gallery's
/// Download button saves it.
fn wait_for_labels(target: &Path, file_name: &str) -> Result<()> {
    let downloads = crate::config::expand_home("~/Downloads").join(file_name);
    eprintln!("Waiting for {} (or {}). Press Ctrl-C to stop.", target.display(), downloads.display());
    loop {
        if target.exists() {
            return Ok(());
        }
        if downloads.exists() {
            std::fs::create_dir_all(target.parent().unwrap_or(Path::new(".")))?;
            std::fs::copy(&downloads, target)?;
            std::fs::remove_file(&downloads)?;
            eprintln!("moved {} → {}", downloads.display(), target.display());
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Held-out accuracy (benchmark + real situations) from a candidate's `report.json`: its own
/// (`baseline = false`) or the current head's, scored by train_head.py on the same tests.
fn head_score(dir: &Path, baseline: bool) -> Option<(usize, usize)> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).ok()?).ok()?;
    let v = if baseline { v.get("baseline")?.clone() } else { v };
    let bench = |k: &str| v[k]["trained_head"].as_array().map(|a| (a[1].as_u64().unwrap_or(0) as usize, a[2].as_u64().unwrap_or(0) as usize));
    let (b_ok, b_n) = bench("test_bench")?;
    let (r_ok, r_n) = bench("test_real").unwrap_or((0, 0));
    let (o_ok, o_n) = bench("test_offscreen").unwrap_or((0, 0));
    Some((b_ok + r_ok + o_ok, b_n + r_n + o_n))
}

pub async fn train_by_human(cfg: &Config, calib_path: &Path, opts: &HumanOpts) -> Result<()> {
    if !opts.video.exists() {
        bail!("no video or recorded run at {}", opts.video.display());
    }
    // A recorded run (`ssbot run --human`): its frames are already cropped to the canvas.
    let run_frames = Some(opts.video.join("frames")).filter(|f| opts.video.is_dir() && f.is_dir());
    // A run directory's name is a timestamp like `20261005-041856.952`: take the whole name,
    // not a file stem (which would drop ".952").
    let stem = if opts.video.is_dir() { opts.video.file_name() } else { opts.video.file_stem() }
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "human".into());
    let stem = if opts.video.is_dir() { format!("human_{}", stem.replace('.', "_")) } else { stem };
    let name: String = opts
        .name
        .clone()
        .unwrap_or(stem)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let data = Path::new("data");
    let labels_dir = Path::new("labels");
    let all_dir = data.join(format!("{name}_all"));
    let pick_dir = data.join(&name);
    let labels_file = labels_path(labels_dir, &name);

    step(1, "Extract frames (10 fps, canvas crop, 640 px)");
    let all_dir = run_frames.clone().unwrap_or(all_dir);
    if run_frames.is_some() {
        eprintln!("using the recorded frames in {}", all_dir.display());
    } else if all_dir.exists() && std::fs::read_dir(&all_dir)?.next().is_some() {
        eprintln!("{} already has frames; reusing them", all_dir.display());
    } else {
        let crop = crate::frames::detect_crop(&opts.video)?;
        if let Some(c) = crop {
            eprintln!("detected canvas crop {}:{}:{}:{}", c.w, c.h, c.x, c.y);
        }
        let n = crate::frames::extract(&opts.video, &all_dir, crop, 10, 640)?;
        eprintln!("{n} frames in {}", all_dir.display());
    }

    step(2, "Select a varied set of frames to label");
    if pick_dir.exists() && std::fs::read_dir(&pick_dir)?.next().is_some() {
        eprintln!("{} already exists; reusing the selection", pick_dir.display());
    } else {
        let n = crate::label::cmd::select_frames(&all_dir, &pick_dir, opts.max_frames, calib_path)?;
        eprintln!("{n} frames selected into {}", pick_dir.display());
    }

    step(3, "Label them");
    if labels_file.exists() {
        eprintln!("{} already exists; using it", labels_file.display());
    } else {
        let labeller = match &opts.labeller {
            Labeller::Auto if has_claude_credentials() => Labeller::Claude,
            Labeller::Auto => Labeller::Manual,
            Labeller::Claude => Labeller::Claude,
            Labeller::Manual => Labeller::Manual,
            Labeller::Import(p) => Labeller::Import(p.clone()),
        };
        match labeller {
            Labeller::Claude => {
                let lo = crate::label::cmd::LabelOpts {
                    frames_dir: pick_dir.clone(),
                    model: "claude-opus-5-5".into(),
                    effort: "low".into(),
                    max: opts.max_frames,
                    sync: false,
                    dry_run: false,
                    labels_dir: labels_dir.to_path_buf(),
                    calibration: calib_path.to_path_buf(),
                    guide: PathBuf::from("prompts/label_guide.md"),
                    repeat_every: 10,
                };
                crate::label::cmd::label(&lo).await?;
            }
            Labeller::Import(ann) => {
                let (n, total) =
                    crate::label::cmd::import(&pick_dir, &ann, None, None, labels_dir, calib_path, 0.4, false)?;
                eprintln!("imported {n} of {total} frames");
            }
            Labeller::Manual | Labeller::Auto => {
                let html = crate::label::cmd::review(&pick_dir, labels_dir, calib_path, true)?;
                let _ = Command::new("open").arg(&html).status();
                eprintln!(
                    "Opened {}. Fix each frame's zones (click a zone to cycle its class), set the lane and \
                     screen, tick \"checked\", then click Download.",
                    html.display()
                );
                wait_for_labels(&labels_file, &format!("{name}.jsonl"))?;
            }
        }
    }

    let dirs = labelled_dirs(data, labels_dir)?;
    eprintln!("labelled sets: {}", dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "));

    step(4, "Train the zone classifier on all labelled frames");
    let report = train_zones(cfg, calib_path, &dirs, labels_dir, Path::new("zone_model.json"), 3000, 1e-3, true)?;
    eprintln!(
        "{} frames, {} zones: {}-fold accuracy {:.1}% (near {:.1}%); facts exact {}/{}, same move {}/{}",
        report.frames,
        report.zones,
        report.folds,
        report.accuracy * 100.0,
        report.near * 100.0,
        report.facts_exact.0,
        report.facts_exact.1,
        report.same_move.0,
        report.same_move.1
    );
    std::fs::write(labels_dir.join("zone_report.json"), serde_json::to_string_pretty(&report)?)?;

    if !opts.train_head {
        eprintln!("\nskipping the OpenJev head (--no-head)");
        return Ok(());
    }

    step(5, "Build OpenJev training data (situations from all labelled frames held out for testing)");
    let mut labelled = Vec::new();
    for (d, dir) in dirs.iter().enumerate() {
        for (id, l) in load_final_labels(labels_dir, &run_name(dir))? {
            labelled.push((d as u64 * 10_000_000 + id, l));
        }
    }
    labelled.sort_by_key(|(id, _)| *id);
    let ds = dataset::build(opts.situations, 1, cfg.advisor.wording, &labelled);
    let adv = data.join("advisor");
    std::fs::create_dir_all(&adv)?;
    for (split, recs) in [
        ("train", &ds.train),
        ("test_bench", &ds.test_bench),
        ("test_real", &ds.test_real),
        ("test_offscreen", &ds.test_offscreen),
    ] {
        let text: String = recs.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        std::fs::write(adv.join(format!("{split}.jsonl")), text)?;
        eprintln!("{split}: {} situations", recs.len());
    }

    step(6, "Train the OpenJev decision head (about 0.3 s per situation)");
    let candidate = PathBuf::from(format!("models/head-{}-candidate", cfg.advisor.model));
    let current = PathBuf::from(if cfg.advisor.head.is_empty() { format!("models/head-{}", cfg.advisor.model) } else { cfg.advisor.head.clone() });
    let status = Command::new(&cfg.advisor.python)
        .args(["sidecar/train_head.py", "--data"])
        .arg(&adv)
        .arg("--out")
        .arg(&candidate)
        .args(["--model", &cfg.advisor.model])
        .arg("--baseline")
        .arg(&current)
        .stdout(std::process::Stdio::null())
        .status()
        .context("running sidecar/train_head.py")?;
    if !status.success() {
        bail!("train_head.py failed ({status})");
    }

    step(7, "Keep the better head");
    let new = head_score(&candidate, false).context("candidate head has no report")?;
    // Both heads scored on the same held-out tests.
    match head_score(&candidate, true) {
        Some(old) if old.0 * new.1 > new.0 * old.1 => {
            eprintln!(
                "new head {}/{} is worse than the current {}/{} on held-out tests; keeping the current one ({} kept for inspection)",
                new.0,
                new.1,
                old.0,
                old.1,
                candidate.display()
            );
        }
        old => {
            if current.exists() {
                let prev = PathBuf::from(format!("{}.prev", current.display()));
                let _ = std::fs::remove_dir_all(&prev);
                std::fs::rename(&current, &prev)?;
            }
            std::fs::rename(&candidate, &current)?;
            eprintln!(
                "new head {}/{} on held-out tests (was {}); installed at {}",
                new.0,
                new.1,
                old.map(|o| format!("{}/{}", o.0, o.1)).unwrap_or_else(|| "none".into()),
                current.display()
            );
        }
    }
    eprintln!("\nDone. Watch it with `ssbot run`; set advisor.head = \"{}\" in ssbot.toml if it isn't already.", current.display());
    Ok(())
}


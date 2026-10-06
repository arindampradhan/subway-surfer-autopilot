//! The OpenJev advisor: `advisor-bench`, `advisor-data`.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use ssbot_engine::config::Config;
use ssbot_lab::label::{load_final_labels, run_name};

#[derive(Args)]
pub struct AdvisorBenchArgs {
    #[arg(long)]
    model: Option<String>,
}

#[derive(Args)]
pub struct AdvisorDataArgs {
    #[arg(long, default_value = "data/advisor")]
    out: PathBuf,
    #[arg(long, default_value_t = 2000)]
    n: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Labelled frame directories whose running frames become the real test set.
    #[arg(long)]
    frames: Vec<PathBuf>,
    #[arg(long, default_value = "labels")]
    labels: PathBuf,
}

pub async fn advisor_bench(cfg: &mut Config, args: AdvisorBenchArgs) -> Result<()> {
    let AdvisorBenchArgs { model } = args;
    if let Some(m) = model {
        cfg.advisor.model = m;
    }
    let (sidecar, mut link) = ssbot_runtime::sidecar::Sidecar::spawn(&cfg.advisor).await?;
    eprintln!("{} · {} cases per wording", sidecar.model, ssbot_lab::advisor_bench::cases().len());
    let scores = ssbot_lab::advisor_bench::run(&mut link, &ssbot_engine::config::Wording::ALL).await?;
    println!("{}", serde_json::to_string_pretty(&scores)?);
    drop(link);
    sidecar.shutdown().await;
    Ok(())
}

pub fn advisor_data(cfg: &Config, args: AdvisorDataArgs) -> Result<()> {
    let AdvisorDataArgs { out, n, seed, frames, labels } = args;
    let mut labelled = Vec::new();
    for (d, dir) in frames.iter().enumerate() {
        for (id, l) in load_final_labels(&labels, &run_name(dir))? {
            labelled.push((d as u64 * 10_000_000 + id, l));
        }
    }
    labelled.sort_by_key(|(id, _)| *id);
    let ds = ssbot_lab::advisor_data::build(n, seed, cfg.advisor.wording, &labelled);
    std::fs::create_dir_all(&out)?;
    for (name, recs) in [("train", &ds.train), ("test_bench", &ds.test_bench), ("test_real", &ds.test_real), ("test_offscreen", &ds.test_offscreen)] {
        let text: String = recs.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        std::fs::write(out.join(format!("{name}.jsonl")), text)?;
        eprintln!("{name}: {} records", recs.len());
    }
    Ok(())
}

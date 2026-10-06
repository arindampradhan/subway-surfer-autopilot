//! Benchmark (`ssbot bench`): many runs, one aggregate, so two configurations can be compared.
//! A single run is mostly noise (few seconds, different levels), so judge changes on the median.

use serde::{Deserialize, Serialize};

use ssbot_live::recorder::Summary;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Aggregate {
    pub runs: usize,
    pub crashed: usize,
    pub survival_median_s: f64,
    pub survival_mean_s: f64,
    pub survival_p25_s: f64,
    pub survival_p75_s: f64,
    /// Runs whose HUD score could be read.
    pub scored: usize,
    pub score_median: Option<f64>,
    pub score_mean: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub tag: String,
    pub advisor: bool,
    pub aggregate: Aggregate,
    pub survival_s: Vec<f64>,
    pub scores: Vec<Option<u64>>,
    /// Run folders, so `ssbot crash-frames` can build training data from a bench.
    pub run_dirs: Vec<String>,
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * q;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

pub fn aggregate(summaries: &[Summary]) -> Aggregate {
    let surv = sorted(summaries.iter().map(|s| s.survival_s).collect());
    let scores = sorted(summaries.iter().filter_map(|s| s.score.map(|x| x as f64)).collect());
    Aggregate {
        runs: summaries.len(),
        crashed: summaries.iter().filter(|s| s.crashed).count(),
        survival_median_s: quantile(&surv, 0.5),
        survival_mean_s: mean(&surv),
        survival_p25_s: quantile(&surv, 0.25),
        survival_p75_s: quantile(&surv, 0.75),
        scored: scores.len(),
        score_median: (!scores.is_empty()).then(|| quantile(&scores, 0.5)),
        score_mean: (!scores.is_empty()).then(|| mean(&scores)),
    }
}

pub fn report(tag: &str, advisor: bool, summaries: &[Summary]) -> BenchReport {
    BenchReport {
        tag: tag.into(),
        advisor,
        aggregate: aggregate(summaries),
        survival_s: summaries.iter().map(|s| s.survival_s).collect(),
        scores: summaries.iter().map(|s| s.score).collect(),
        run_dirs: summaries.iter().filter_map(|s| s.run_dir.clone()).collect(),
    }
}

impl std::fmt::Display for BenchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a = &self.aggregate;
        writeln!(f, "{} (advisor {}): {} runs, {} ended in a crash", self.tag, if self.advisor { "on" } else { "off" }, a.runs, a.crashed)?;
        writeln!(
            f,
            "  survival  median {:.1} s   mean {:.1} s   middle half {:.1}-{:.1} s",
            a.survival_median_s, a.survival_mean_s, a.survival_p25_s, a.survival_p75_s
        )?;
        match (a.score_median, a.score_mean) {
            (Some(med), Some(mean)) => write!(f, "  score     median {med:.0}   mean {mean:.0}   ({} of {} read)", a.scored, a.runs),
            _ => write!(f, "  score     not read (is `tesseract` installed?)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(survival_s: f64, crashed: bool, score: Option<u64>) -> Summary {
        Summary { survival_s, crashed, score, ..Default::default() }
    }

    #[test]
    fn aggregates_runs() {
        let runs = [run(2.0, true, Some(100)), run(4.0, true, Some(300)), run(10.0, false, None), run(6.0, true, Some(200))];
        let a = aggregate(&runs);
        assert_eq!((a.runs, a.crashed, a.scored), (4, 3, 3));
        assert_eq!(a.survival_median_s, 5.0);
        assert_eq!(a.survival_mean_s, 5.5);
        assert_eq!(a.score_median, Some(200.0));
        assert!(aggregate(&[]).score_median.is_none());
    }
}

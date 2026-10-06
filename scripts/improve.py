#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["typer>=0.12"]
# ///
"""One round of the improvement loop. The bot plays, its own crashes become training data, a
candidate zone model is trained, and it replaces the live one only if it benchmarks better.

  scripts/improve.py base   <tag> [runs]   bench the live model, pull the pre-crash frames
  (label data/crash_<tag> -> labels/crash_<tag>.jsonl: ask Claude Code, or `ssbot label`)
  scripts/improve.py train  <tag>          train a candidate zone CNN on every labelled set
  scripts/improve.py verify <tag> [runs]   bench the candidate; promote it if it wins

Judge on the median over 20+ runs: single runs are noise.

Offline glue (CLAUDE.md "Scripts are glue"): it only chains `ssbot` and sidecar commands. Run it
as `uv run scripts/improve.py ...` or `../.venv/bin/python scripts/improve.py ...`. The env vars
SSBOT, PYTHON, EXCLUDE and FROZEN still override the matching options.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

LIVE_MODEL = Path("zone_model.json")
CANDIDATE_MODEL = Path("models/candidate/zone_model.json")  # a CNN, see sidecar/zone_cnn.py
# Stays at the root: every relative path inside a calibration file (marker references,
# `zone_model`) resolves against the calibration file's own directory.
CANDIDATE_CALIB = Path("calibration.candidate.toml")
ARCHIVE = Path("models/archive")

Tag = Annotated[str, typer.Argument(help="Round tag, e.g. r1.")]
Runs = Annotated[int, typer.Argument(help="Bench runs.")]
Ssbot = Annotated[str, typer.Option(envvar="SSBOT", help="The ssbot binary.")]
Python = Annotated[str, typer.Option(envvar="PYTHON", help="Python with torch, for sidecar/zone_cnn.py.")]
# Recordings whose frames don't line up with the zones (cyan border: the game didn't fill the
# window) would teach the classifier the wrong crops.
Exclude = Annotated[str, typer.Option(envvar="EXCLUDE", help="Space-separated labelled sets never trained on (frames that don't line up with the zones).")]
Frozen = Annotated[str, typer.Option(envvar="FROZEN", help="Never trained on: the fixed test set the eyes are scored on (`ssbot eval-zones data/<frozen>`).")]

app = typer.Typer(help=__doc__.split("\n\n")[0], add_completion=False, no_args_is_help=True)


def run(*cmd: str) -> None:
    """Run a command with its output streamed; stop with its exit code if it fails, like `set -e`."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode)
    except OSError as e:
        typer.echo(f"{cmd[0]}: {e.strerror}", err=True)
        raise typer.Exit(127)


def fail(msg: str) -> None:
    typer.echo(msg, err=True)
    raise typer.Exit(1)


def non_empty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def labelled_dirs(exclude: str) -> list[str]:
    """`data/<name>` for every non-empty `labels/<name>.jsonl` with recorded frames, minus `exclude`."""
    skip = exclude.split()
    return [
        f"data/{f.stem}"
        for f in sorted(Path("labels").glob("*.jsonl"))
        if non_empty(f) and Path("data", f.stem).is_dir() and f.stem not in skip
    ]


def candidate_wins(tag: str) -> bool:
    """True when the candidate's median survival beats the live model's, from the two bench reports."""

    def median(kind: str) -> float:
        with open(f"runs/bench_{tag}-{kind}.json") as f:
            return json.load(f)["aggregate"]["survival_median_s"]

    return median("candidate") > median("base")


@app.command()
def base(tag: Tag, runs: Runs = 20, ssbot: Ssbot = "./target/release/ssbot") -> None:
    """Bench the live model, pull the pre-crash frames into data/crash_<tag>."""
    run(ssbot, "bench", "--runs", str(runs), "--tag", f"{tag}-base")
    run(ssbot, "crash-frames", f"runs/bench_{tag}-base.json", "--out", f"data/crash_{tag}")
    typer.echo()
    typer.echo(f"Next: label data/crash_{tag} into labels/crash_{tag}.jsonl, then: scripts/improve.py train {tag}")


@app.command()
def train(
    tag: Tag,
    ssbot: Ssbot = "./target/release/ssbot",
    python: Python = "../.venv/bin/python",
    exclude: Exclude = "human_round2",
    frozen: Frozen = "crash_r3",
) -> None:
    """Train a candidate zone CNN on every labelled set."""
    if not non_empty(Path(f"labels/crash_{tag}.jsonl")):
        fail(f"labels/crash_{tag}.jsonl is missing or empty: label data/crash_{tag} first")
    # Full-resolution crops with 2x context around each zone: held-out missed hazards fell 36%.
    run(ssbot, "zone-crops", *labelled_dirs(exclude), "--out", "data/zone_crops", "--native", "--crop", "48", "--ctx", "2.0")
    CANDIDATE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    run(python, "sidecar/zone_cnn.py", "--holdout", frozen, "--export", str(CANDIDATE_MODEL))
    line = f'zone_model = "{CANDIDATE_MODEL}"'.encode()
    calib = Path("calibration.toml").read_bytes()
    CANDIDATE_CALIB.write_bytes(re.sub(rb"(?m)^zone_model = .*", lambda _: line, calib))
    typer.echo()
    typer.echo(f"Candidate saved as {CANDIDATE_MODEL}. Next: scripts/improve.py verify {tag}")


@app.command()
def verify(tag: Tag, runs: Runs = 20, ssbot: Ssbot = "./target/release/ssbot") -> None:
    """Bench the candidate; promote it if it wins."""
    if not (CANDIDATE_MODEL.is_file() and CANDIDATE_CALIB.is_file()):
        fail("no candidate: run 'train' first")
    run(ssbot, "--calibration", str(CANDIDATE_CALIB), "bench", "--runs", str(runs), "--tag", f"{tag}-candidate")
    typer.echo()
    run(ssbot, "bench-compare", f"runs/bench_{tag}-base.json", f"runs/bench_{tag}-candidate.json")
    if candidate_wins(tag):
        backup = ARCHIVE / f"zone_model.before-{tag}.json"
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LIVE_MODEL, backup)
        shutil.copyfile(CANDIDATE_MODEL, LIVE_MODEL)
        typer.echo(f"Promoted: candidate beat the live model on median survival (old model kept as {backup}).")
    else:
        typer.echo("Not promoted: the candidate did not beat the live model on median survival.")


if __name__ == "__main__":
    os.chdir(Path(__file__).absolute().parent.parent)
    app()

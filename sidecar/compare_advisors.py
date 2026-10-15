"""Score advisors side by side on the held-out advisor tests: accuracy and time per decision.

Each advisor is the sidecar's own `Engine`, so this measures what the bot would run. Decisions
are not cached, so every one pays the full cost (for Clef Flash, a network round trip). An
advisor that fails to load (no head yet, no Cloudflare credentials) is reported and skipped.

  ../.venv/bin/python sidecar/compare_advisors.py
  ../.venv/bin/python sidecar/compare_advisors.py --advisors embeddinggemma:models/head-embeddinggemma clef-flash
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from openjev_sidecar import Engine  # noqa: E402

DEFAULT = ["0.8b:models/head-0.8b", "embeddinggemma:models/head-embeddinggemma", "clef-flash"]
SPLITS = ["test_bench", "test_real", "test_offscreen", "test_roof_powerup"]


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def run(engine, records):
    hits, ms = 0, []
    for r in records:
        start = time.perf_counter()
        probs = engine.probabilities(r["premise"], r["options"])
        ms.append((time.perf_counter() - start) * 1000)
        hits += max(probs, key=probs.get) in r["best"]
    return hits, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/advisor")
    ap.add_argument("--advisors", nargs="+", default=DEFAULT, help="MODEL or MODEL:HEAD_DIR")
    ap.add_argument("--out", default="", help="also write the results as JSON here")
    args = ap.parse_args()

    splits = {s: load(p) for s in SPLITS if os.path.exists(p := os.path.join(args.data, f"{s}.jsonl"))}
    results = {}
    for spec in args.advisors:
        model, _, head = spec.partition(":")
        try:
            engine = Engine(model, "mlx", head)
            run(engine, list(splits.values())[0][:1])  # warm-up: kernels, connection, option cache
        except (Exception, SystemExit) as exc:
            print(f"{spec}: skipped ({exc})", file=sys.stderr)
            continue
        row, all_ms = {}, []
        for name, records in splits.items():
            hits, ms = run(engine, records)
            row[name] = [hits, len(records)]
            all_ms += ms
        row["ms_p50"], row["ms_p95"] = (round(float(np.percentile(all_ms, q)), 1) for q in (50, 95))
        results[engine.name] = row
        print(f"{engine.name}: done", file=sys.stderr)

    header = ["advisor", *splits, "p50 ms", "p95 ms"]
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for name, row in results.items():
        cells = [f"{row[s][0]}/{row[s][1]}" for s in splits] + [str(row["ms_p50"]), str(row["ms_p95"])]
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

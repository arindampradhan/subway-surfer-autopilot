"""Train OpenJev's decision head for the Subway Surfers advisor.

The OpenJev backbone stays frozen. For each (facts premise, option sentence) pair we take the
last-token latent (the input of OpenJev's NLI `score` head) and train
`modeling_openjev.LatentMLPHead` to score the correct option highest, which is how the model
card adapts OpenJev to a decision task. Data comes from `ssbot advisor-data`:
  train.jsonl       random situations, correct move from the tested reflex policy
  test_bench.jsonl  hand-written benchmark cases (held out)
  test_real.jsonl   situations from labelled gameplay frames (held out)

Reports zero-shot accuracy (OpenJev's own P(entailment), what the sidecar used before) and
the trained head's accuracy on each split, then saves the head for `openjev_sidecar.py --head`.

  ../.venv/bin/python sidecar/train_head.py --data data/advisor --out models/head-0.8b
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from openjev_sidecar import MODELS, REPO  # noqa: E402


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def digest(records):
    import hashlib

    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def featurize(ce, records, cache_path):
    """Latents for every option of every record, cached in an .npz next to the data. The cache
    is keyed by the records' content, so a regenerated dataset is never served stale latents."""
    h = digest(records)
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if "digest" in z.files and str(z["digest"]) == h:
            return z["X"], z["gold"], z["qid"], list(z["keys"])
    X, gold, qid, keys = [], [], [], []
    t0 = time.perf_counter()
    for q, r in enumerate(records):
        ks = sorted(r["options"])
        lat = ce.latents_hypotheses(r["premise"], [r["options"][k] for k in ks])
        X.append(lat)
        gold += [1.0 if k in r["best"] else 0.0 for k in ks]
        qid += [q] * len(ks)
        keys += ks
        if q and q % 200 == 0:
            print(f"  {q}/{len(records)} ({(time.perf_counter() - t0) / q * 1000:.0f} ms each)", file=sys.stderr)
    X = np.concatenate(X).astype(np.float32)
    gold, qid = np.array(gold, np.float32), np.array(qid)
    np.savez(cache_path, X=X, gold=gold, qid=qid, keys=np.array(keys), n=len(records), digest=h)
    return X, gold, qid, keys


def accuracy(scores, gold, qid):
    """Fraction of questions whose top-scored option is one of the correct ones."""
    hits = []
    for q in np.unique(qid):
        m = qid == q
        hits.append(gold[m][scores[m].argmax()] == 1)
    return float(np.mean(hits)), int(np.sum(hits)), len(hits)


def misses(scores, gold, qid, keys, records):
    out = []
    keys = np.array(keys)
    for q in np.unique(qid):
        m = qid == q
        pick = keys[m][scores[m].argmax()]
        if gold[m][scores[m].argmax()] != 1:
            out.append(f"{records[q]['source']}: picked {pick}, want {records[q]['best']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/advisor")
    ap.add_argument("--out", default="models/head-0.8b")
    ap.add_argument("--model", choices=MODELS, default="0.8b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="", help="current head to score on the same test sets")
    args = ap.parse_args()

    from mlx_openjev import ENT, MlxOpenJev
    from modeling_openjev import LatentMLPHead

    ce = MlxOpenJev(REPO, MODELS[args.model])
    score_w = np.array(ce.score)  # (3, hidden): OpenJev's own NLI head, for the zero-shot baseline
    splits = {}
    for name in ["train", "test_bench", "test_real", "test_offscreen"]:
        path = os.path.join(args.data, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        recs = load(path)
        print(f"{name}: {len(recs)} situations", file=sys.stderr)
        X, gold, qid, keys = featurize(ce, recs, os.path.join(args.data, f"latents_{args.model}_{name}.npz"))
        splits[name] = (recs, X, gold, qid, keys)

    def zero_shot(X):
        logits = X @ score_w.T
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True))[:, ENT]

    _, Xtr, gtr, qtr, _ = splits["train"]
    head = LatentMLPHead(d=Xtr.shape[1], seed=args.seed, device="cpu", epochs=200, patience=20)
    head.fit(Xtr, gtr, qtr, val_frac=0.15)
    print(f"head: validation accuracy {head.val_acc:.3f} (grouped hold-out of train)", file=sys.stderr)

    report = {"model": args.model, "train_situations": len(splits["train"][0]), "val_acc": head.val_acc}
    for name, (recs, X, gold, qid, keys) in splits.items():
        zs, trained = zero_shot(X), head.predict(X)
        report[name] = {
            "zero_shot": accuracy(zs, gold, qid),
            "trained_head": accuracy(trained, gold, qid),
            "head_misses": misses(trained, gold, qid, keys, recs)[:15] if name != "train" else [],
        }
    if args.baseline and os.path.exists(os.path.join(args.baseline, "head.pt")):
        base = LatentMLPHead.load(args.baseline, device="cpu")
        report["baseline"] = {
            name: {"trained_head": accuracy(base.predict(X), gold, qid)}
            for name, (recs, X, gold, qid, keys) in splits.items()
            if name != "train"
        }
    head.save(args.out)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

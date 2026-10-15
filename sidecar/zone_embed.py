"""EmbeddingGemma 2 on zone crops: can it label zones, and which labels does it doubt?

Offline only (see CLAUDE.md). Three ways to label a held-out set, all scored with
`zone_cnn.report` so the numbers line up with the CNN's (paper Table 4):
  zero-shot     each crop takes the class whose text description it is closest to
  kNN           majority label of the k nearest labelled crops (k chosen on --val)
  linear probe  logistic regression on the embeddings (C chosen on --val)
Then a label audit over every labelled crop: a crop is flagged when most of its nearest
neighbours carry one other label. The audit is written as jsonl plus a contact sheet.

Embeddings are cached next to the crops, so only the first run pays (~170 ms per crop on MPS).

  ../.venv/bin/python sidecar/zone_embed.py --out runs/zone_embed
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from zone_cnn import CLASSES, ROOT, load_crops, report  # noqa: E402

# One description per class, for the zero-shot baseline; wording follows labels/label_guide.md.
DESCRIPTIONS = {
    "Free": "a clear railway track: rails, sleepers and gravel only",
    "TrainBody": "the side or front of a large boxy coloured train carriage with windows standing on the track",
    "TrainRamp": "a sloped ramp leading up onto the roof of a train",
    "LowBarrier": "a short striped barrier near the ground on the track, low enough to jump over",
    "HighBarrier": "a taller barrier or sign at body height on the track, passed by rolling under it",
    "OverheadBar": "a horizontal bar raised above the track with open space beneath it",
}


def embed(samples, cache, tokens):
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["E"]) == len(samples):
            return z["E"], z["T"]
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-2", device=device, model_kwargs={"torch_dtype": torch.float32})
    # 280 image tokens by default; a 48x48 crop holds far less detail, and 70 is ~4.6x faster.
    model[0].processor.image_processor.max_soft_tokens = tokens
    t0, E = time.perf_counter(), []
    for i in range(0, len(samples), 256):
        batch = [Image.fromarray(s["crop"]) for s in samples[i : i + 256]]
        E.append(model.encode(batch, batch_size=32, normalize_embeddings=True))
        print(f"  {i + len(batch)}/{len(samples)} ({(time.perf_counter() - t0) / (i + len(batch)) * 1000:.0f} ms each)", file=sys.stderr)
    E = np.concatenate(E).astype(np.float32)
    T = model.encode([DESCRIPTIONS[c] for c in CLASSES], prompt_name="Classification", normalize_embeddings=True)
    np.savez(cache, E=E, T=T)
    return E, T


def knn_votes(E_query, E_ref, y_ref, k, exclude_self=False):
    """Label counts among the k nearest reference crops (cosine), one row per query."""
    votes = np.zeros((len(E_query), len(CLASSES)), np.int32)
    for i in range(0, len(E_query), 1024):
        sim = E_query[i : i + 1024] @ E_ref.T
        if exclude_self:
            sim[np.arange(len(sim)), np.arange(i, i + len(sim))] = -np.inf
        nn = np.argpartition(-sim, k, axis=1)[:, :k]
        for row, idx in enumerate(nn):
            votes[i + row] = np.bincount(y_ref[idx], minlength=len(CLASSES))
    return votes


def accuracy(pred, y):
    return float(np.mean(pred == y))


def contact_sheet(samples, flagged, path, n=60, scale=3):
    from PIL import Image, ImageDraw

    flagged = flagged[:n]
    side, cols = 48 * scale, 10
    sheet = Image.new("RGB", (cols * side, ((len(flagged) + cols - 1) // cols) * (side + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, f in enumerate(flagged):
        x, y = (j % cols) * side, (j // cols) * (side + 24)
        sheet.paste(Image.fromarray(samples[f["index"]]["crop"]).resize((side, side), Image.NEAREST), (x, y))
        draw.text((x + 2, y + side + 1), f"{f['label'][:9]}>{f['neighbours'][:9]}", fill="black")
        draw.text((x + 2, y + side + 12), f"{f['set'][:12]}#{f['id']}", fill="gray")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.join(ROOT, "data", "zone_crops"))
    ap.add_argument("--holdout", default="crash_r3", help="frozen test set (paper Table 4)")
    ap.add_argument("--val", default="crash_r2", help="set used to choose k and C")
    ap.add_argument("--tokens", type=int, default=70, help="image tokens per crop")
    ap.add_argument("--audit-k", type=int, default=10)
    ap.add_argument("--audit-agree", type=float, default=0.8, help="share of neighbours that must agree on another label")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "zone_embed"))
    args = ap.parse_args()

    samples = load_crops(args.crops)
    E, T = embed(samples, os.path.join(args.crops, f"embeddinggemma_t{args.tokens}.npz"), args.tokens)
    y = np.array([s["y"] for s in samples])
    sets = np.array([s["set"] for s in samples])
    test, val = sets == args.holdout, sets == args.val
    train = ~test & ~val
    test_samples = [s for s, t in zip(samples, test) if t]
    print(f"{len(samples)} crops: train {train.sum()}, val {val.sum()} ({args.val}), test {test.sum()} ({args.holdout})")

    print("\n== zero-shot (closest class description)")
    report(args.holdout, test_samples, list((E[test] @ T.T).argmax(1)))

    print("\n== kNN")
    ks = [1, 3, 5, 10, 20]
    k = max(ks, key=lambda k: accuracy(knn_votes(E[val], E[train], y[train], k).argmax(1), y[val]))
    print(f"k = {k} (best on {args.val})")
    ref = train | val
    report(args.holdout, test_samples, list(knn_votes(E[test], E[ref], y[ref], k).argmax(1)))

    print("\n== linear probe")
    from sklearn.linear_model import LogisticRegression

    def probe(C, mask):
        return LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(E[mask], y[mask])

    C = max([0.1, 1.0, 10.0], key=lambda C: accuracy(probe(C, train).predict(E[val]), y[val]))
    print(f"C = {C} (best on {args.val})")
    report(args.holdout, test_samples, list(probe(C, ref).predict(E[test])))

    # Audit: every labelled crop against all the others.
    votes = knn_votes(E, E, y, args.audit_k, exclude_self=True)
    top = votes.argmax(1)
    share = votes.max(1) / args.audit_k
    flagged = [
        {
            "index": int(i), "set": samples[i]["set"], "id": samples[i]["id"], "lane": samples[i]["lane"],
            "band": samples[i]["band"], "label": CLASSES[y[i]], "neighbours": CLASSES[top[i]], "agree": float(share[i]),
        }
        for i in np.argsort(-share)
        if top[i] != y[i] and share[i] >= args.audit_agree
    ]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "audit.jsonl"), "w") as f:
        for row in flagged:
            f.write(json.dumps(row) + "\n")
    contact_sheet(samples, flagged, os.path.join(args.out, "audit.png"))
    print(f"\n== audit: {len(flagged)} of {len(samples)} crops disagree with >= {args.audit_agree:.0%} of their {args.audit_k} nearest neighbours")
    for (label, other), n in Counter((r["label"], r["neighbours"]) for r in flagged).most_common(10):
        print(f"   labelled {label:12s} neighbours say {other:12s} {n}")
    print(f"   by set: {dict(Counter(r['set'] for r in flagged).most_common())}")
    print(f"   written to {args.out}/audit.jsonl and audit.png")


if __name__ == "__main__":
    main()

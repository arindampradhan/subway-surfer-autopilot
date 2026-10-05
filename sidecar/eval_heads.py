"""Score trained decision heads on a test set: python sidecar/eval_heads.py TEST.jsonl HEAD_DIR..."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_head import accuracy, featurize, load  # noqa: E402
from openjev_sidecar import MODELS, REPO  # noqa: E402


def main():
    test, heads = sys.argv[1], sys.argv[2:]
    from mlx_openjev import MlxOpenJev
    from modeling_openjev import LatentMLPHead

    ce = MlxOpenJev(REPO, MODELS["0.8b"])
    recs = load(test)
    X, gold, qid, _ = featurize(ce, recs, test.replace(".jsonl", "_latents.npz"))
    for h in heads:
        ok, n = accuracy(LatentMLPHead.load(h, device="cpu").predict(X), gold, qid)[1:]
        print(f"{h}: {ok}/{n} = {ok / n:.3f}")


if __name__ == "__main__":
    main()

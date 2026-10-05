"""OpenJev cross-encoder on Apple MLX.

mlx-lm's Qwen3.5 backbone has fused Metal kernels for the Gated DeltaNet layers, which
PyTorch on MPS runs as ~11k tiny ops per forward. On top of it sits the checkpoint's
3-way NLI `score` head, applied to the last real token like OpenJevCrossEncoder does.
"""

import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.models import qwen3_5
from transformers import AutoTokenizer

CON, ENT, NEU = 0, 1, 2


class MlxOpenJev:
    def __init__(self, repo: str, subfolder: str):
        path = Path(snapshot_download(repo, allow_patterns=[f"{subfolder}/*"])) / subfolder
        config = json.loads((path / "config.json").read_text())
        self.template = config["nli_template"]

        self.model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
        weights = {}
        for f in glob.glob(str(path / "model*.safetensors")):
            weights.update(mx.load(f))
        self.score = weights.pop("score.weight").astype(mx.float32)  # (3, hidden)
        # strict: every backbone weight must be present and match, or loading fails loudly
        self.model.load_weights(list(self.model.sanitize(weights).items()), strict=True)
        mx.eval(self.model.parameters())

        self.tok = AutoTokenizer.from_pretrained(path)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def predict_hypotheses(self, premise: str, hypotheses: list) -> np.ndarray:
        """[contradiction, entailment, neutral] probabilities, one row per hypothesis."""
        last = self._last(premise, hypotheses)
        return np.array(mx.softmax(last @ self.score.T, axis=-1))

    def latents_hypotheses(self, premise: str, hypotheses: list) -> np.ndarray:
        """Last-token hidden state (the `score` head's input), one row per hypothesis. This is
        the latent `modeling_openjev.LatentMLPHead` is trained on."""
        return np.array(self._last(premise, hypotheses))

    def _last(self, premise: str, hypotheses: list):
        """Final-norm hidden state at each input's last real token.

        The token prefix all inputs share is run once at batch size 1; its cache is then
        copied into every row and only the differing suffixes are run as a batch.
        """
        texts = [self.template.format(premise=premise.strip(), hypothesis=h.strip()) for h in hypotheses]
        # Tokenize whole texts: the tokenizer can merge across the premise/hypothesis boundary.
        ids = [self.tok(t)["input_ids"] for t in texts]
        common = 0
        for column in zip(*ids):
            if len(set(column)) != 1:
                break
            common += 1
        common = min(common, min(map(len, ids)) - 1)  # every row keeps at least one suffix token

        backbone = self.model.language_model.model
        cache = self.model.make_cache()
        if common:
            backbone(mx.array([ids[0][:common]]), cache=cache)
            n = len(ids)
            for c in cache:
                if hasattr(c, "keys"):  # full-attention KV cache
                    c.keys = mx.repeat(c.keys[..., : c.offset, :], n, axis=0)
                    c.values = mx.repeat(c.values[..., : c.offset, :], n, axis=0)
                else:  # linear-attention conv + recurrent state
                    c.cache = [mx.repeat(s, n, axis=0) for s in c.cache]

        suffixes = [x[common:] for x in ids]
        lengths = [len(s) for s in suffixes]
        width = max(lengths)
        # Right padding is safe: the backbone is causal, so pads never reach earlier tokens.
        batch = mx.array([s + [self.pad_id] * (width - len(s)) for s in suffixes])
        hidden = backbone(batch, cache=cache)  # final-norm hidden states
        return hidden[mx.arange(len(ids)), mx.array(lengths) - 1].astype(mx.float32)

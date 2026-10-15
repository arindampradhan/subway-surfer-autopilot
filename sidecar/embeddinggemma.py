"""EmbeddingGemma 2 as the advisor's encoder (a bi-encoder, unlike OpenJev's cross-encoder).

The premise is embedded as a search query and each option sentence as a document. The option
sentences come from a small fixed vocabulary, so their embeddings are cached and a decision
costs one forward pass over the premise. `latents_hypotheses` matches `MlxOpenJev`'s, so
`train_head.py` trains `modeling_openjev.LatentMLPHead` on these features unchanged.

Needs sentence-transformers, transformers >= 5.19 and torchvision (the model's processor
imports the Gemma 4 image processor even for text).
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO = "google/embeddinggemma-2"


class EmbeddingGemma:
    def __init__(self, repo: str = REPO, dim: int = 768):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # The model card warns float16 can give NaN or degraded embeddings; float32 is safe on MPS.
        self.model = SentenceTransformer(repo, device=device, model_kwargs={"torch_dtype": torch.float32})
        self.dim = dim
        self.options = {}  # option sentence -> embedding

    def _encode(self, texts: list, prompt_name: str) -> np.ndarray:
        return self.model.encode(texts, prompt_name=prompt_name, truncate_dim=self.dim, normalize_embeddings=True)

    def _embed(self, premise: str, hypotheses: list):
        new = [h for h in dict.fromkeys(hypotheses) if h not in self.options]
        if new:
            self.options.update(zip(new, self._encode(new, "Document")))
        return self._encode([premise], "SearchQuery")[0], np.stack([self.options[h] for h in hypotheses])

    def latents_hypotheses(self, premise: str, hypotheses: list) -> np.ndarray:
        """One row per hypothesis: [premise, option, premise * option], the head's input."""
        p, o = self._embed(premise, hypotheses)
        p = np.broadcast_to(p, o.shape)
        return np.concatenate([p, o, p * o], axis=1).astype(np.float32)

    @staticmethod
    def zero_shot(latents: np.ndarray) -> np.ndarray:
        """Cosine similarity of premise and option: the sum of the normalised product block."""
        d = latents.shape[1] // 3
        return latents[:, 2 * d :].sum(axis=1)

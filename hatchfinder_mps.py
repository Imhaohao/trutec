"""Preserve native query geometry across MPS's adaptive pooling limitation."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from hatchfinder import HatchFinder


def adaptive_pool(feature: torch.Tensor, pool_size: int) -> torch.Tensor:
    needs_cpu = feature.device.type == 'mps' and (
        feature.shape[-2] % pool_size != 0 or feature.shape[-1] % pool_size != 0)
    source = feature.cpu() if needs_cpu else feature
    pooled = F.adaptive_avg_pool2d(source, (pool_size, pool_size))
    return pooled.to(feature.device)


class MPSHatchFinder(HatchFinder):
    """Only query pooling crosses to CPU; drawing features remain on MPS.

    Extends the matching-vector layer of pinned hatchfinder 0.1.2. The upstream
    adaptive-pooling values and native crop dimensions are preserved exactly.
    """

    def _get_vectors(self, drawing_features: list[torch.Tensor], hatch: torch.Tensor
                     ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        hatch_features = self.hatch_encoder(hatch)
        drawing_vectors: list[torch.Tensor | None] = [None]*len(drawing_features)
        hatch_vectors: list[torch.Tensor | None] = [None]*len(drawing_features)
        for level, feature in enumerate(hatch_features):
            key = str(level)
            if key in self.hatch_projections:
                pooled = adaptive_pool(feature, self.config.model.hatch_pool_sizes[level])
                hatch_vectors[level] = self.hatch_projections[key](pooled.flatten(1))
        for level, feature in enumerate(drawing_features):
            key = str(level)
            if key in self.drawing_projections:
                drawing_vectors[level] = self.drawing_projections[key](feature)
        return drawing_vectors, hatch_vectors

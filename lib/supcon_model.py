"""ResNet-34 encoder with projection head and Anomaly Contrastive Loss.

References
----------
- Bommes et al., "Anomaly Detection in IR Images of PV Modules using
  Supervised Contrastive Learning" (arXiv:2112.02922)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResNet34Encoder(nn.Module):
    """ResNet-34 backbone → 2-layer projection → L2-normalised embeddings."""

    def __init__(
        self,
        embed_dim: int = 128,
        proj_hidden: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet34(weights=weights)
        feat_dim = backbone.fc.in_features          # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)


class AnomalyContrastiveLoss(nn.Module):
    r"""Loss from Eq. 4 of arXiv:2112.02922.

    For every *normal* embedding z_i the loss encourages high cosine
    similarity with the batch-mean normal embedding and low similarity
    with every *anomalous* embedding.

    Falls back to standard SupCon when the batch is single-class.

    Parameters
    ----------
    temperature : float
        Scaling factor τ in the softmax denominator.
    normal_label : int
        Integer label that denotes the "No-Anomaly" / healthy class.
    """

    def __init__(self, temperature: float = 0.07, normal_label: int = 11):
        super().__init__()
        self.temperature = temperature
        self.normal_label = normal_label

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        device = embeddings.device
        normal_mask = labels == self.normal_label
        anomaly_mask = ~normal_mask

        n_normal = normal_mask.sum().item()
        n_anomaly = anomaly_mask.sum().item()

        if n_normal == 0 or n_anomaly == 0:
            return self._fallback_supcon(embeddings, labels)

        z_normal = embeddings[normal_mask]      # (N, D)
        z_anomaly = embeddings[anomaly_mask]    # (A, D)

        z_bar = F.normalize(z_normal.mean(dim=0, keepdim=True), dim=-1)  # (1, D)

        all_z = torch.cat([z_normal, z_anomaly], dim=0)  # (N+A, D)
        logits = (all_z @ z_bar.T).squeeze(-1) / self.temperature  # (N+A,)

        log_softmax_all = logits - torch.logsumexp(logits, dim=0)
        loss = -log_softmax_all[:n_normal].mean()
        return loss

    # ── fallback: vanilla SupCon when only one class present ──

    def _fallback_supcon(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        B = embeddings.size(0)
        sim = (embeddings @ embeddings.T) / self.temperature  # (B, B)

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        positives = label_eq & ~eye

        if positives.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        log_softmax = sim - torch.logsumexp(sim * (~eye).float() + eye.float() * -1e9, dim=1, keepdim=True)
        loss = -(log_softmax * positives.float()).sum() / positives.sum()
        return loss

"""
Hybrid Vision Transformer + LSTM for ordered MRI slice series.

    slices (B, T, 3, 224, 224)
      -> ViT encoder, applied to every slice     -> (B, T, D) one [CLS] vector per slice
      -> bidirectional LSTM across the T slices  -> (B, T, 2H)
      -> attention pooling over slices           -> (B, 2H)
      -> classifier                              -> (B, n_classes)

Two explanation signals come out with the logits:
    slice_attention (B, T)       how much each slice contributed to the decision
    patch_attention (B, T, 196)  the ViT's last-layer [CLS] attention over the 14 x 14 patches
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTLSTM(nn.Module):
    def __init__(self, n_classes: int = 3, backbone: str = "vit_tiny_patch16_224", pretrained: bool = True,
                 lstm_hidden: int = 128, lstm_layers: int = 1, dropout: float = 0.3, use_lstm: bool = True):
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.vit.num_features
        self.use_lstm = use_lstm
        if not use_lstm:          # ablation baseline: ViT features averaged over slices, no sequence model
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes))
            return
        self.lstm = nn.LSTM(d, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.slice_score = nn.Linear(2 * lstm_hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * lstm_hidden), nn.Dropout(dropout), nn.Linear(2 * lstm_hidden, n_classes))

    # -- ViT ---------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        """x: (N, 3, 224, 224) -> cls (N, D), cls-to-patch attention (N, 196)."""
        v = self.vit
        t = v.patch_embed(x)
        t = v._pos_embed(t)
        t = v.patch_drop(t)
        t = v.norm_pre(t)
        for blk in v.blocks[:-1]:
            t = blk(t)
        last = v.blocks[-1]
        y = last.norm1(t)
        n, tokens, c = y.shape
        heads = last.attn.num_heads
        qkv = last.attn.qkv(y).reshape(n, tokens, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        att = (q @ k.transpose(-2, -1)) * (c // heads) ** -0.5
        att = att.softmax(dim=-1)
        prefix = getattr(v, "num_prefix_tokens", 1)
        cls_att = att[:, :, 0, prefix:].mean(dim=1)
        cls_att = cls_att / (cls_att.sum(dim=-1, keepdim=True) + 1e-8)
        t = last(t)
        t = v.norm(t)
        return t[:, 0], cls_att

    # -- full model --------------------------------------------------------
    def forward(self, x: torch.Tensor):
        b, t = x.shape[0], x.shape[1]
        cls, patch_att = self.encode(x.flatten(0, 1))
        seq = cls.reshape(b, t, -1)
        if not self.use_lstm:
            w = torch.full((b, t), 1.0 / t, device=x.device)
            return self.head(seq.mean(dim=1)), w, patch_att.reshape(b, t, -1)
        out, _ = self.lstm(seq)
        w = torch.softmax(self.slice_score(out).squeeze(-1), dim=1)
        pooled = (out * w.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled)
        return logits, w, patch_att.reshape(b, t, -1)

    # -- fine-tuning helpers -------------------------------------------------
    def freeze_vit(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n: int):
        for blk in self.vit.blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.vit.norm.parameters():
            p.requires_grad = True

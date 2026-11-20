"""Render slice thumbnails and attention overlays for the result page."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import IMG_SIZE

# charcoal -> gold -> ember ramp, sampled at 0, 0.5, 1
_RAMP = np.array([[226, 181, 90], [240, 150, 60], [226, 90, 60]], dtype=np.float32)


def _colour(h: np.ndarray) -> np.ndarray:
    t = np.clip(h, 0, 1) * 2.0
    lo = np.clip(np.floor(t).astype(int), 0, 1)
    f = (t - lo)[..., None]
    return _RAMP[lo] * (1 - f) + _RAMP[lo + 1] * f


def slice_png(g: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8), mode="L")


def overlay_png(g: np.ndarray, patch_map: np.ndarray) -> Image.Image:
    """Blend a 14 x 14 attention map over a 224 x 224 slice."""
    m = patch_map.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heat = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    h = np.asarray(heat, dtype=np.float32) / 255.0
    h = h * (g > 0.05)                      # keep the map inside the head
    base = np.repeat((np.clip(g, 0, 1) * 255)[..., None], 3, axis=2)
    alpha = (np.clip(h, 0, 1) ** 1.3 * 0.72)[..., None]
    out = base * (1 - alpha) + _colour(h) * alpha
    return Image.fromarray(out.astype(np.uint8), mode="RGB")

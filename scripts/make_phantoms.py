"""
Generate SYNTHETIC brain-like slice series for testing the interface and the
training pipeline. These are drawn with simple geometry. They are not MRI data
and say nothing about how the model performs on real scans.

    python scripts/make_phantoms.py --out samples                 # two demo series
    python scripts/make_phantoms.py --dataset /tmp/phantom_ds     # tiny 3-class dataset in the generic layout
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter


def phantom_slice(z: float, vent: float, rng, size: int = 208) -> Image.Image:
    """z in 0..1 through the brain, vent = ventricle scale (1 typical, larger = enlarged)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, 1.2), size / 2 + rng.normal(0, 1.2)
    bulge = 0.80 + 0.20 * np.sin(np.pi * (0.15 + 0.7 * z))
    rx, ry = size * 0.36 * bulge, size * 0.43 * bulge
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    img = np.zeros((size, size), np.float32)
    img[r < 1.10] = 0.22                      # scalp
    img[r < 1.04] = 0.08                      # skull
    brain = r < 0.97
    ang = np.arctan2(yy - cy, xx - cx)
    cortex = 0.52 + 0.06 * np.sin(ang * 14 + r * 22)
    white = 0.74 + 0.03 * np.sin(ang * 5)
    img[brain] = np.where(r[brain] > 0.72 + 0.05 * np.sin(ang[brain] * 9), cortex[brain], white[brain])
    img[brain & (np.abs(xx - cx) < 1.4) & (r > 0.25)] = 0.18      # longitudinal fissure
    sulci = brain & (r > 0.80) & (np.sin(ang * 22 + z * 6) > 0.86 - 0.10 * (vent - 1))
    img[sulci] = 0.15
    w = np.exp(-((z - 0.5) / 0.22) ** 2) * vent               # ventricles are largest mid-brain
    for side in (-1, 1):
        vx, vy = cx + side * size * 0.075, cy - size * 0.02
        v = ((xx - vx - side * (yy - vy) * 0.25) / (size * 0.040 * w + 1e-3)) ** 2 + ((yy - vy) / (size * 0.125 * w + 1e-3)) ** 2
        img[brain & (v < 1)] = 0.06
    img += rng.normal(0, 0.022, img.shape).astype(np.float32) * (r < 1.10)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.8))
    return im


def series(folder: str, vent: float, n: int = 16, seed: int = 0):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        phantom_slice(0.2 + 0.6 * i / max(n - 1, 1), vent, rng).save(os.path.join(folder, f"slice_{i + 1:02d}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subjects", type=int, default=8)
    a = ap.parse_args()
    if a.out:
        series(os.path.join(a.out, "phantom_typical_ventricles"), 1.0, seed=1)
        series(os.path.join(a.out, "phantom_enlarged_ventricles"), 1.75, seed=2)
        print("wrote sample series to", a.out)
    if a.dataset:
        rng = np.random.default_rng(7)
        for cls, vent in (("CN", 0.95), ("MCI", 1.35), ("AD", 1.8)):
            for s in range(a.subjects):
                series(os.path.join(a.dataset, cls, f"sub-{cls}{s:02d}"), vent + rng.normal(0, 0.07), n=20, seed=1000 + s + int(vent * 100))
        print("wrote phantom dataset to", a.dataset)

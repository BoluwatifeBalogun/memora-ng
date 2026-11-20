"""
Shared MRI preprocessing.

This module is imported by BOTH the training code (training/dataset.py) and the
web application (memora/ai/engine.py). Keeping one implementation guarantees
that a slice is prepared for inference exactly the way it was prepared for
training. It depends only on NumPy and Pillow.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

SEQ_LEN = 16          # slices fed to the ViT-LSTM per study
IMG_SIZE = 224        # ViT input resolution
PATCH = 16            # ViT patch size -> 14 x 14 patch grid
GRID = IMG_SIZE // PATCH
NORM_MEAN = 0.5
NORM_STD = 0.5

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED_VOLUME_EXT = {".nii", ".nii.gz"}


class SliceRejected(ValueError):
    """Raised when an uploaded file does not look like a brain MRI slice."""


@dataclass
class SliceCheck:
    ok: bool
    reason: str = ""


def natural_key(name: str):
    """Sort key so that slice_2 comes before slice_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # corrupt or not an image at all
        raise SliceRejected("The file could not be read as an image.") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo + 1e-6) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return img.convert("RGB")


def check_slice(img: Image.Image) -> SliceCheck:
    """Heuristic gate that keeps photographs, documents and blank files out."""
    w, h = img.size
    if min(w, h) < 64:
        return SliceCheck(False, "Image is smaller than 64 pixels on one side.")
    small = img.resize((128, 128), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    if float(sat.mean()) > 0.10:
        return SliceCheck(False, "Image is in colour. MRI slices are greyscale.")
    g = rgb.mean(axis=2)
    if float(g.std()) < 0.06:
        return SliceCheck(False, "Image is almost blank.")
    b = 8  # border ring thickness in the 128 px thumbnail
    ring = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    if float(ring.mean()) > 0.25:
        return SliceCheck(False, "No dark background around the head, so this does not look like an MRI slice.")
    fg = float((g > 0.12).mean())
    if fg < 0.05:
        return SliceCheck(False, "Too little tissue is visible in this slice.")
    if fg > 0.92:
        return SliceCheck(False, "The image is filled edge to edge, unlike a brain MRI slice.")
    return SliceCheck(True)


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def normalise_intensity(g: np.ndarray) -> np.ndarray:
    """Percentile stretch computed on tissue pixels, output in 0..1."""
    tissue = g[g > 0.04]
    if tissue.size < 50:
        return np.clip(g, 0, 1)
    lo, hi = np.percentile(tissue, [1.0, 99.0])
    if hi - lo < 1e-3:
        return np.clip(g, 0, 1)
    out = (g - lo) / (hi - lo)
    out[g <= 0.04] = 0.0
    return np.clip(out, 0.0, 1.0)


def fit_square(g: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Pad to a square with black, then resize to size x size."""
    h, w = g.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = g
    im = Image.fromarray((canvas * 255).astype(np.uint8), mode="L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def prepare_slice(img: Image.Image) -> np.ndarray:
    """RGB PIL image -> float32 array (224, 224) in 0..1."""
    return fit_square(normalise_intensity(to_gray(img)))


def pick_indices(n: int, t: int = SEQ_LEN) -> np.ndarray:
    """Evenly spaced, order preserving indices. Repeats slices when n < t."""
    if n <= 0:
        raise ValueError("empty series")
    return np.round(np.linspace(0, n - 1, t)).astype(int)


def series_quality(n: int) -> str:
    if n >= SEQ_LEN:
        return "full"
    if n >= 4:
        return "sparse"
    return "single"


def to_model_input(series: np.ndarray) -> np.ndarray:
    """(T, 224, 224) in 0..1 -> (1, T, 3, 224, 224) normalised float32."""
    x = (series.astype(np.float32) - NORM_MEAN) / NORM_STD
    x = np.repeat(x[:, None, :, :], 3, axis=1)
    return x[None]


def volume_to_slices(path: str, count: int = 32) -> list[Image.Image]:
    """Read a NIfTI volume and return evenly spaced axial slices as images."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise SliceRejected("NIfTI support needs the nibabel package on the server.") from exc
    try:
        vol = nib.as_closest_canonical(nib.load(path))
        data = np.asarray(vol.get_fdata(dtype=np.float32))
    except Exception as exc:
        raise SliceRejected("The NIfTI volume could not be read.") from exc
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3 or min(data.shape) < 16:
        raise SliceRejected("The NIfTI file is not a 3D brain volume.")
    lo, hi = np.percentile(data, [0.5, 99.5])
    data = np.clip((data - lo) / (hi - lo + 1e-6), 0, 1)
    z = data.shape[2]
    start, stop = int(z * 0.38), int(z * 0.78)   # band that covers ventricles and hippocampus
    idx = np.round(np.linspace(start, max(start, stop - 1), count)).astype(int)
    out = []
    for i in idx:
        sl = np.rot90(data[:, :, i])              # canonical RAS -> anterior at the top
        out.append(Image.fromarray((sl * 255).astype(np.uint8), mode="L").convert("RGB"))
    return out

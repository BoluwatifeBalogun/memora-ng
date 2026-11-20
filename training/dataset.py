"""
Dataset discovery, subject-level splitting and the slice-series Dataset.

Two folder layouts are understood.

1. OASIS-1 slice export (Kaggle "imagesoasis"): class folders that contain files
   named  OAS1_0001_MR1_mpr-1_100.jpg . Subject, scan run and slice number are
   parsed from the file name, so every (subject, run) becomes one ordered series.

2. Generic layout, for an ADNI export or any other source:
       root/<class folder>/<subject folder>/<slice images in order>

Class folders are mapped to the three classes of the study:
   CN   non demented, normal, CN
   MCI  very mild, MCI, EMCI, LMCI           (OASIS CDR 0.5)
   AD   mild, moderate, AD, Alzheimer        (OASIS CDR 1 and 2)
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memora.ai.preprocess import (ALLOWED_IMAGE_EXT, IMG_SIZE, NORM_MEAN, NORM_STD, SEQ_LEN,  # noqa: E402
                                  natural_key, open_image, pick_indices, prepare_slice)

OASIS = re.compile(r"(OAS1_\d{4})_MR(\d+)_mpr-(\d+)_(\d+)", re.I)
CACHE_SLICES = 32   # slices kept per series in the cache; 16 are drawn from them


def class_of(folder: str) -> int | None:
    s = re.sub(r"[^a-z]", "", folder.lower())
    if s in ("cn", "nc", "normal", "control", "controls") or "nondement" in s or "cognitivelynormal" in s:
        return 0
    if s in ("mci", "emci", "lmci") or "verymild" in s or "mildcognitive" in s:
        return 1
    if s in ("ad", "dementia") or "alzheimer" in s or "moderate" in s or "milddement" in s or s == "demented":
        return 2
    return None


@dataclass
class Series:
    subject: str
    key: str
    label: int
    files: list = field(default_factory=list)


def discover(root: str) -> list[Series]:
    groups: dict[tuple, Series] = {}
    order: dict[tuple, list] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        label, depth = None, None
        for i, p in enumerate(parts):
            c = class_of(p)
            if c is not None:
                label, depth = c, i
                break
        if label is None:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() not in ALLOWED_IMAGE_EXT:
                continue
            m = OASIS.search(f)
            if m:
                subject, key, pos = m.group(1), f"{m.group(1)}_MR{m.group(2)}_mpr{m.group(3)}", int(m.group(4))
            elif len(parts) > depth + 1:
                subject = parts[depth + 1]
                key, pos = "/".join(parts[depth + 1:]), natural_key(f)
            else:
                continue   # loose files without a subject cannot form an ordered series
            gk = (label, key)
            if gk not in groups:
                groups[gk] = Series(subject=subject, key=key, label=label)
            order[gk].append((pos, os.path.join(dirpath, f)))
    out = []
    for gk, s in groups.items():
        s.files = [p for _pos, p in sorted(order[gk], key=lambda t: t[0])]
        if len(s.files) >= 4:
            out.append(s)
    return sorted(out, key=lambda s: s.key)


def split_by_subject(series: list[Series], seed: int = 42, val: float = 0.15, test: float = 0.15):
    """Stratified split on SUBJECTS so one person never appears in two splits."""
    from sklearn.model_selection import train_test_split

    subj_label = {}
    for s in series:
        subj_label[s.subject] = s.label
    subjects = sorted(subj_label)
    labels = [subj_label[s] for s in subjects]
    train_s, rest_s, _tl, rest_l = train_test_split(subjects, labels, test_size=val + test, stratify=labels, random_state=seed)
    val_s, test_s = train_test_split(rest_s, test_size=test / (val + test), stratify=rest_l, random_state=seed)
    pick = lambda names: [s for s in series if s.subject in set(names)]  # noqa: E731
    return pick(train_s), pick(val_s), pick(test_s)


def cap_per_subject(series: list[Series], n: int) -> list[Series]:
    seen, out = defaultdict(int), []
    for s in series:
        if seen[s.subject] < n:
            out.append(s)
            seen[s.subject] += 1
    return out


def load_stack(s: Series, cache_dir: str | None) -> np.ndarray:
    """Return uint8 array (K, 224, 224), K <= CACHE_SLICES, cached on disk."""
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", f"{s.label}_{s.key}") + ".npy")
        if os.path.exists(path):
            return np.load(path)
    files = s.files
    if len(files) > CACHE_SLICES:
        files = [files[i] for i in pick_indices(len(files), CACHE_SLICES)]
    stack = []
    for f in files:
        with open(f, "rb") as fh:
            stack.append((prepare_slice(open_image(fh.read())) * 255).astype(np.uint8))
    arr = np.stack(stack)
    if path:
        np.save(path, arr)
    return arr


class SeriesDataset:
    """torch Dataset yielding (T, 3, 224, 224) tensors and a label."""

    def __init__(self, series: list[Series], train: bool, cache_dir: str | None = None, seq_len: int = SEQ_LEN):
        self.series, self.train, self.cache_dir, self.seq_len = series, train, cache_dir, seq_len

    def __len__(self):
        return len(self.series)

    def _augment(self, x):
        import math

        import torch
        import torch.nn.functional as F

        ang = math.radians(float(np.random.uniform(-8, 8)))
        sc = float(np.random.uniform(0.94, 1.06))
        tx, ty = np.random.uniform(-0.04, 0.04, 2)
        cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)
        theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1)          # same transform for every slice
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        gain, gamma = float(np.random.uniform(0.9, 1.1)), float(np.random.uniform(0.85, 1.18))
        return (x.clamp(0, 1) ** gamma * gain).clamp(0, 1)

    def __getitem__(self, i):
        import torch

        s = self.series[i]
        stack = load_stack(s, self.cache_dir)
        n = len(stack)
        if self.train and n > self.seq_len:
            idx = np.sort(np.random.choice(n, self.seq_len, replace=False))   # ordered random subset
        else:
            idx = pick_indices(n, self.seq_len)
        x = torch.from_numpy(stack[idx].astype(np.float32) / 255.0).unsqueeze(1)   # (T, 1, H, W)
        if self.train:
            x = self._augment(x)
        x = ((x - NORM_MEAN) / NORM_STD).repeat(1, 3, 1, 1)
        return x, s.label


assert IMG_SIZE == 224

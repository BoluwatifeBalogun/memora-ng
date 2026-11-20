"""
Inference engines.

OnnxEngine  runs the trained hybrid ViT-LSTM exported by training/export_onnx.py.
DemoEngine  is a transparent rule used ONLY while no trained model file exists,
            so the whole system can be shown end to end. It is not a trained
            model and every screen and record says so.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from .labels import CLASS_NAMES
from .preprocess import GRID, IMG_SIZE, SEQ_LEN, to_model_input

MODEL_FILE = "memora_vit_lstm.onnx"
CARD_FILE = "model_card.json"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _package(probs, slice_att, patch_att, mode, version):
    probs = np.asarray(probs, dtype=np.float64)
    idx = int(probs.argmax())
    return {
        "label_index": idx,
        "label": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "slice_attention": [float(v) for v in slice_att],
        "patch_attention": np.asarray(patch_att, dtype=np.float32).reshape(SEQ_LEN, GRID, GRID),
        "engine": mode,
        "model_version": version,
    }


class OnnxEngine:
    mode = "model"

    def __init__(self, model_path: str, card: dict):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("MEMORA_THREADS", "2"))
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.card = card
        self.version = card.get("version", "vit-lstm")
        self._lock = threading.Lock()
        labels = card.get("labels")
        if labels and list(labels) != CLASS_NAMES:
            raise RuntimeError("model_card.json labels do not match the application's classes")

    def analyse(self, series: np.ndarray) -> dict:
        x = to_model_input(series)
        with self._lock:
            logits, slice_att, patch_att = self.session.run(None, {self.input_name: x})
        return _package(_softmax(logits[0]), slice_att[0], patch_att[0], self.mode, self.version)


class DemoEngine:
    """Ventricle-size rule of thumb. Deterministic. For demonstration only."""

    mode = "demo"
    version = "demo-rule-1"
    card: dict = {}

    def __init__(self):
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
        self._yy, self._xx = yy, xx

    def _central(self, g: np.ndarray):
        mask = g > 0.12
        if mask.sum() < 200:
            return np.zeros_like(g, dtype=bool), 0.0
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()
        ry, rx = (ys.max() - ys.min()) / 2 * 0.42, (xs.max() - xs.min()) / 2 * 0.42
        ell = ((self._yy - cy) / (ry + 1e-6)) ** 2 + ((self._xx - cx) / (rx + 1e-6)) ** 2 <= 1.0
        dark = ell & (g < 0.30)
        return dark, float(dark.sum()) / float(ell.sum() + 1e-6)

    def analyse(self, series: np.ndarray) -> dict:
        fracs, maps = [], []
        for g in series:
            dark, frac = self._central(g)
            fracs.append(frac)
            m = dark.astype(np.float32).reshape(GRID, IMG_SIZE // GRID, GRID, IMG_SIZE // GRID).mean(axis=(1, 3))
            maps.append(m)
        fracs = np.array(fracs)
        top = np.sort(fracs)[len(fracs) // 2:]
        z = (float(top.mean()) - 0.36) / 0.12     # thresholds tuned on the bundled synthetic phantoms only
        probs = _softmax(np.array([-2.0 * z, 1.1 - 1.4 * abs(z), 2.0 * z]))
        slice_att = _softmax(fracs * 9.0)
        patch = np.stack(maps)
        patch = patch / (patch.reshape(SEQ_LEN, -1).sum(axis=1)[:, None, None] + 1e-6)
        return _package(probs, slice_att, patch, self.mode, self.version)


_engine = None
_engine_lock = threading.Lock()


def load_engine(store_dir: str, force: bool = False):
    """Return the trained engine when model files are present, else the demo engine."""
    global _engine
    with _engine_lock:
        if _engine is not None and not force:
            return _engine
        model_path = os.path.join(store_dir, MODEL_FILE)
        card_path = os.path.join(store_dir, CARD_FILE)
        if os.path.exists(model_path):
            card = {}
            if os.path.exists(card_path):
                with open(card_path, encoding="utf-8") as fh:
                    card = json.load(fh)
            try:
                _engine = OnnxEngine(model_path, card)
                return _engine
            except Exception as exc:  # fall back, but say why
                eng = DemoEngine()
                eng.load_error = str(exc)
                _engine = eng
                return _engine
        _engine = DemoEngine()
        return _engine

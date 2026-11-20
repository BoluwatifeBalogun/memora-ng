"""Turns an upload into a stored study: validate, preprocess, predict, save."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
from flask import current_app
from werkzeug.utils import secure_filename

from .ai import overlay
from .ai.engine import load_engine
from .ai.preprocess import (ALLOWED_IMAGE_EXT, SEQ_LEN, SliceRejected, check_slice, natural_key, open_image,
                            pick_indices, prepare_slice, series_quality, volume_to_slices)
from .db import MRIImage, Prediction, db

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_VOLUME_BYTES = 80 * 1024 * 1024


class StudyError(ValueError):
    pass


def _ext(name: str) -> str:
    n = name.lower()
    return ".nii.gz" if n.endswith(".nii.gz") else os.path.splitext(n)[1]


def _read_images(files, max_slices):
    named = sorted(((secure_filename(f.filename or "") or "slice", f) for f in files), key=lambda t: natural_key(t[0]))
    if len(named) > max_slices:
        raise StudyError(f"Upload at most {max_slices} slices per study. You sent {len(named)}.")
    good, skipped = [], []
    for name, f in named:
        if _ext(name) not in ALLOWED_IMAGE_EXT:
            raise StudyError(f"{name} is not a supported file. Use PNG, JPG, BMP or TIFF slices, or one NIfTI volume.")
        data = f.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise StudyError(f"{name} is larger than 16 MB.")
        try:
            img = open_image(data)
        except SliceRejected as exc:
            raise StudyError(f"{name}: {exc}") from exc
        verdict = check_slice(img)
        if verdict.ok:
            good.append((name, img))
        else:
            skipped.append((name, verdict.reason))
    return good, skipped


def _read_volume(f):
    name = secure_filename(f.filename or "volume.nii")
    tmp = tempfile.mkdtemp(prefix="memora_")
    try:
        path = os.path.join(tmp, name)
        f.save(path)
        if os.path.getsize(path) > MAX_VOLUME_BYTES:
            raise StudyError("The NIfTI volume is larger than 80 MB.")
        try:
            imgs = volume_to_slices(path)
        except SliceRejected as exc:
            raise StudyError(str(exc)) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    good, skipped = [], []
    for i, img in enumerate(imgs):
        verdict = check_slice(img)
        (good if verdict.ok else skipped).append((f"{name} · axial {i + 1}", img) if verdict.ok else (f"axial {i + 1}", verdict.reason))
    return name, good, skipped


def run_study(patient, files, exam_date, user):
    """Returns (Prediction, warnings). Raises StudyError with a user-facing message."""
    files = [f for f in files if f and f.filename]
    if not files:
        raise StudyError("Choose at least one MRI slice or a NIfTI volume.")
    volumes = [f for f in files if _ext(f.filename) in (".nii", ".nii.gz")]
    if volumes and len(files) > 1:
        raise StudyError("Upload either one NIfTI volume or a set of slice images, not both.")
    if volumes:
        label, good, skipped = _read_volume(volumes[0])
        kind = "nifti"
    else:
        good, skipped = _read_images(files, current_app.config["MAX_SLICES"])
        label = good[0][0] if good else ""
        kind = "images"

    total = len(good) + len(skipped)
    if not good or len(good) < 0.6 * total:
        why = skipped[0][1] if skipped else "No usable slices were found."
        raise StudyError(f"This upload was rejected. {why}")
    warnings = []
    if skipped:
        warnings.append(f"{len(skipped)} of {total} slices were left out because they did not pass the MRI check.")

    prepared = np.stack([prepare_slice(img) for _n, img in good])
    series = prepared[pick_indices(len(prepared), SEQ_LEN)]
    quality = series_quality(len(prepared))
    if quality == "single":
        warnings.append("Fewer than 4 slices were supplied, so the sequence model had almost no slice-to-slice context.")
    elif quality == "sparse":
        warnings.append(f"Only {len(prepared)} slices were supplied. 16 or more gives the sequence model full context.")

    engine = load_engine(current_app.config["MODEL_STORE"])
    t0 = time.perf_counter()
    result = engine.analyse(series)
    latency = int((time.perf_counter() - t0) * 1000)

    uid = uuid.uuid4().hex
    folder = os.path.join(current_app.config["UPLOAD_DIR"], uid)
    os.makedirs(folder, exist_ok=True)
    try:
        for i, g in enumerate(series):
            overlay.slice_png(g).save(os.path.join(folder, f"slice_{i:02d}.png"))
        top = [int(i) for i in np.argsort(result["slice_attention"])[::-1][:3]]
        for i in top:
            overlay.overlay_png(series[i], result["patch_attention"][i]).save(os.path.join(folder, f"overlay_{i:02d}.png"))

        name = label if len(good) == 1 or kind == "nifti" else f"{label} (+{len(good) - 1} slices)"
        image = MRIImage(Patient_ID=patient.Patient_ID, Image_Name=name[:200], Image_Path=uid, Processing_Status="processed",
                         Slice_Count=len(good), Source_Kind=kind, Exam_Date=exam_date)
        db.session.add(image)
        db.session.flush()
        pred = Prediction(Patient_ID=patient.Patient_ID, Image_ID=image.Image_ID, User_ID=user.User_ID,
                          Disease_Category=result["label"], Confidence_Score=result["confidence"],
                          Probabilities=json.dumps(result["probabilities"]), Slice_Attention=json.dumps(result["slice_attention"]),
                          Top_Slices=json.dumps(top), Series_Quality=quality, Engine=result["engine"],
                          Model_Version=result["model_version"], Latency_Ms=latency)
        db.session.add(pred)
        db.session.commit()
    except Exception:
        db.session.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return pred, warnings


def delete_study_files(image: MRIImage):
    folder = os.path.join(current_app.config["UPLOAD_DIR"], os.path.basename(image.Image_Path))
    shutil.rmtree(folder, ignore_errors=True)

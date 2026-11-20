"""
Train and evaluate the hybrid ViT-LSTM, then write the files the web app loads.

    python training/train.py --data /path/to/dataset --out model_store

Outputs in --out:
    memora_vit_lstm.pt     best PyTorch weights (not needed by the web app)
    memora_vit_lstm.onnx   model served by the Flask application
    model_card.json        labels, dataset summary, training settings, test metrics

Stage A trains the LSTM and classifier on a frozen ViT. Stage B unfreezes the
last ViT blocks and fine-tunes everything at a small learning rate. The split is
made on subjects, so the test metrics come from people the model never saw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memora.ai.labels import CLASS_NAMES  # noqa: E402
from memora.ai.preprocess import SEQ_LEN  # noqa: E402
from training.dataset import SeriesDataset, cap_per_subject, discover, split_by_subject  # noqa: E402
from training.export_onnx import export  # noqa: E402
from training.model import ViTLSTM  # noqa: E402


def metrics_report(y_true, y_prob) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score)

    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels))
    except ValueError:      # a class is missing from this split
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "roc_auc_ovr_macro": auc,
        "per_class": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits, _w, _a = model(x.to(device))
        ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(np.asarray(y))
    return metrics_report(np.concatenate(ys), np.concatenate(ps))


def run_epochs(model, loaders, device, epochs, lr_head, lr_vit, class_w, best, out_pt, patience, tag):
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vit.")]
    vit_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")]
    groups = [{"params": head_params, "lr": lr_head}]
    if vit_params:
        groups.append({"params": vit_params, "lr": lr_vit})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(1, epochs * len(loaders["train"])), pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _w, _a = model(x)
                loss = crit(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, seen = tot + loss.item() * len(y), seen + len(y)
        val = evaluate(model, loaders["val"], device)
        print(f"[{tag}] epoch {ep}/{epochs}  loss {tot / max(seen, 1):.4f}  val acc {val['accuracy']:.3f}  val F1 {val['f1_macro']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if val["f1_macro"] > best["f1"]:
            best.update(f1=val["f1_macro"], val=val)
            torch.save(model.state_dict(), out_pt)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{tag}] early stop")
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="model_store")
    ap.add_argument("--cache", default=None, help="folder for preprocessed slice stacks")
    ap.add_argument("--dataset-name", default="OASIS-1 cross-sectional (axial slice export)")
    ap.add_argument("--backbone", default="vit_tiny_patch16_224")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs-a", type=int, default=4)
    ap.add_argument("--epochs-b", type=int, default=14)
    ap.add_argument("--unfreeze", type=int, default=4, help="ViT blocks fine-tuned in stage B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-series-per-subject", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vit-only", action="store_true", help="ablation baseline without the LSTM, for the Chapter 4 comparison")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    series = discover(a.data)
    if not series:
        sys.exit("No slice series found. Check the folder layout described in training/dataset.py")
    series = cap_per_subject(series, a.max_series_per_subject)
    tr, va, te = split_by_subject(series, seed=a.seed)
    count = lambda ss: dict(Counter(CLASS_NAMES[s.label] for s in ss))  # noqa: E731
    print("series  train", count(tr), " val", count(va), " test", count(te), flush=True)

    mk = lambda ss, train: DataLoader(SeriesDataset(ss, train, a.cache), batch_size=a.batch, shuffle=train,  # noqa: E731
                                      num_workers=a.workers, drop_last=False)
    loaders = {"train": mk(tr, True), "val": mk(va, False), "test": mk(te, False)}

    freq = np.bincount([s.label for s in tr], minlength=len(CLASS_NAMES)).astype(np.float64)
    w = freq.sum() / (len(CLASS_NAMES) * np.maximum(freq, 1))
    class_w = torch.tensor(np.clip(w, None, 3.0), dtype=torch.float32, device=device)   # capped so rare classes do not dominate
    print("class weights", class_w.tolist())

    model = ViTLSTM(len(CLASS_NAMES), a.backbone, pretrained=not a.no_pretrained, use_lstm=not a.vit_only).to(device)
    out_pt = os.path.join(a.out, "memora_vit_lstm.pt")
    best = {"f1": -1.0, "val": None}

    model.freeze_vit()
    best = run_epochs(model, loaders, device, a.epochs_a, 1e-3, 0.0, class_w, best, out_pt, a.patience, "A frozen ViT")
    if a.epochs_b > 0:
        model.unfreeze_last_blocks(a.unfreeze)
        best = run_epochs(model, loaders, device, a.epochs_b, 2e-4, 2e-5, class_w, best, out_pt, a.patience, "B fine-tune")

    model.load_state_dict(torch.load(out_pt, map_location=device))
    test = evaluate(model, loaders["test"], device)
    print("TEST", json.dumps({k: v for k, v in test.items() if k not in ("per_class", "confusion_matrix")}, indent=1))

    subj = lambda ss: len({s.subject for s in ss})  # noqa: E731
    card = {
        "version": "vit-lstm-" + time.strftime("%Y%m%d"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "labels": CLASS_NAMES,
        "architecture": {"backbone": a.backbone, "pretrained": not a.no_pretrained, "sequence_length": SEQ_LEN,
                         "lstm": "bidirectional, 128 hidden units", "pooling": "attention over slices",
                         "parameters": int(sum(p.numel() for p in model.parameters()))},
        "dataset": {"name": a.dataset_name, "split": "subject level, stratified 70 / 15 / 15",
                    "subjects": {"train": subj(tr), "val": subj(va), "test": subj(te)},
                    "series": {"train": count(tr), "val": count(va), "test": count(te)}},
        "training": {"stage_a_epochs": a.epochs_a, "stage_b_epochs": a.epochs_b, "unfrozen_blocks": a.unfreeze,
                     "batch": a.batch, "optimizer": "AdamW, one-cycle schedule", "loss": "class-weighted cross entropy, label smoothing 0.05",
                     "device": str(device), "seed": a.seed},
        "validation": best["val"],
        "test": test,
    }
    with open(os.path.join(a.out, "model_card.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    if a.vit_only:
        os.replace(os.path.join(a.out, "model_card.json"), os.path.join(a.out, "baseline_vit_only.json"))
        print("Baseline (ViT only) metrics saved to baseline_vit_only.json. Compare them with the hybrid model in Chapter 4.")
        return
    export(model.cpu(), os.path.join(a.out, "memora_vit_lstm.onnx"))
    print("Saved model_card.json and memora_vit_lstm.onnx to", a.out)


if __name__ == "__main__":
    main()

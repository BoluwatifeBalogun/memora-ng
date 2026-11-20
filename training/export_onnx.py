"""Export the trained ViT-LSTM to ONNX and confirm ONNX Runtime agrees with PyTorch."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memora.ai.preprocess import IMG_SIZE, SEQ_LEN  # noqa: E402


def export(model, path: str, check: bool = True) -> None:
    model.eval()
    for blk in model.vit.blocks:            # plain attention maths exports more portably
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False
    dummy = torch.randn(1, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE)
    kw = dict(input_names=["slices"], output_names=["logits", "slice_attention", "patch_attention"], opset_version=17)
    try:
        torch.onnx.export(model, (dummy,), path, dynamo=False, **kw)
    except TypeError:                        # older torch without the dynamo argument
        torch.onnx.export(model, (dummy,), path, **kw)
    if not check:
        return
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in model(dummy)]
    got = sess.run(None, {"slices": dummy.numpy()})
    for name, a, b in zip(kw["output_names"], ref, got):
        diff = float(np.abs(a - b.reshape(a.shape)).max())
        print(f"onnx check  {name:16s} max abs diff {diff:.2e}")
        if diff > 1e-3:
            raise RuntimeError(f"ONNX output {name} differs from PyTorch by {diff}")


if __name__ == "__main__":
    import argparse

    from training.model import ViTLSTM

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="model_store/memora_vit_lstm.onnx")
    ap.add_argument("--backbone", default="vit_tiny_patch16_224")
    a = ap.parse_args()
    m = ViTLSTM(3, a.backbone, pretrained=False)
    m.load_state_dict(torch.load(a.weights, map_location="cpu"))
    export(m, a.out)

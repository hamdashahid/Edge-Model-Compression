"""
inference.py - Q1.4 / Q2.5: deployment inference pipeline for ALL FIVE models
(fp32, ptq, qat, unstructured, structured). Everything runs on CPU.

Examples
  python inference.py --model qat --image cat.png          # classify one or more image files
  python inference.py --model all --test-samples 5         # compare all models on CIFAR-10 test images
  python inference.py --model structured --test-samples 20
"""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import *

ALL = list(MODEL_FILES)


def preprocess_image(path):
    """Any image file -> normalised 1x3x32x32 tensor (resized to CIFAR resolution)."""
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


@torch.no_grad()
def predict(model, x):
    """x: Bx3x32x32 -> (class_idx, confidence, latency_ms_per_batch)."""
    t0 = time.perf_counter()
    logits = model(x)
    ms = (time.perf_counter() - t0) * 1000
    prob = F.softmax(logits.float(), 1)
    conf, idx = prob.max(1)
    return idx, conf, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all", choices=ALL + ["all"])
    ap.add_argument("--image", nargs="*", default=None, help="image file(s) to classify")
    ap.add_argument("--test-samples", type=int, default=5,
                    help="number of CIFAR-10 test images to classify when --image is not given")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    names = ALL if args.model == "all" else [args.model]
    models = {}
    for n in names:
        try:
            models[n] = load_model(n)
        except FileNotFoundError as e:
            print(f"[skip] {n}: {e}")
    if not models:
        raise SystemExit("no models available")

    if args.image:
        items = [(p, preprocess_image(p), None) for p in args.image]
    else:
        _, _, test = get_datasets(args.synthetic)
        items = [(f"test[{i}]", test[i][0].unsqueeze(0), int(test[i][1]))
                 for i in range(args.test_samples)]

    for tag, x, label in items:
        head = f"\n{tag}" + (f"   (true: {CLASSES[label]})" if label is not None else "")
        print(head)
        for n, m in models.items():
            idx, conf, ms = predict(m, x)
            print(f"  {MODEL_LABELS[n]:<20s} -> {CLASSES[idx.item()]:<11s} "
                  f"conf {conf.item()*100:5.1f}%   {ms:6.1f} ms")


if __name__ == "__main__":
    main()

"""
onnx_inference.py - Deliverable 8: CPU inference + benchmark of the exported ONNX models
with ONNX Runtime (no PyTorch needed at inference time, except for loading CIFAR-10).

Examples
  python onnx_inference.py --model qat --image cat.png
  python onnx_inference.py --benchmark            # accuracy + latency of every ONNX model -> results/onnx_benchmark.csv
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from common import ONNX_DIR, RESULT_DIR, CLASSES, CIFAR_MEAN, CIFAR_STD, MODEL_LABELS, get_datasets

NAMES = ["fp32", "ptq", "qat", "unstructured", "structured"]


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"resnet18_{name}.onnx")


def session(name, threads=0):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path(name), so, providers=["CPUExecutionProvider"])


def preprocess(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(CIFAR_MEAN, np.float32)) / np.array(CIFAR_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run_benchmark(synthetic, threads, lat_n=200):
    _, _, test = get_datasets(synthetic)
    X = np.stack([test[i][0].numpy() for i in range(len(test))]).astype(np.float32)
    y = np.array([int(test[i][1]) for i in range(len(test))])
    rows = []
    for n in NAMES:
        if not os.path.exists(onnx_path(n)):
            print(f"[skip] {n}: {onnx_path(n)} missing")
            continue
        s = session(n, threads)
        inp = s.get_inputs()[0].name
        preds = np.concatenate([s.run(None, {inp: X[i:i + 256]})[0].argmax(1)
                                for i in range(0, len(X), 256)])
        acc = 100.0 * (preds == y).mean()
        for _ in range(20):
            s.run(None, {inp: X[:1]})
        ts = []
        for i in range(lat_n):
            t0 = time.perf_counter()
            s.run(None, {inp: X[i % len(X): i % len(X) + 1]})
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"model": n, "description": MODEL_LABELS[n], "onnx_size_mb": round(os.path.getsize(onnx_path(n)) / 1e6, 3),
                     "accuracy_pct": round(acc, 2), "latency_bs1_mean_ms": round(float(np.mean(ts)), 3),
                     "latency_bs1_p95_ms": round(float(np.percentile(ts, 95)), 3)})
        print(f"  {MODEL_LABELS[n]:<20s} acc {acc:6.2f}%  {np.mean(ts):7.3f} ms/img  {rows[-1]['onnx_size_mb']:.2f} MB")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "onnx_benchmark.csv"), index=False)
    print(f"saved {os.path.join(RESULT_DIR, 'onnx_benchmark.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fp32", choices=NAMES)
    ap.add_argument("--image", nargs="*", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.benchmark:
        return run_benchmark(args.synthetic, args.threads)
    s = session(args.model, args.threads)
    inp = s.get_inputs()[0].name
    if args.image:
        for p in args.image:
            prob = softmax(s.run(None, {inp: preprocess(p)})[0])
            print(f"{p}: {CLASSES[prob.argmax()]} ({prob.max()*100:.1f}%)  [{MODEL_LABELS[args.model]}, ONNX]")
    else:
        _, _, test = get_datasets(args.synthetic)
        for i in range(5):
            x, lab = test[i]
            prob = softmax(s.run(None, {inp: x.numpy()[None]})[0])
            print(f"test[{i}] true={CLASSES[int(lab)]:<10s} pred={CLASSES[prob.argmax()]:<10s} ({prob.max()*100:.1f}%)")


if __name__ == "__main__":
    main()

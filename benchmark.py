"""
benchmark.py - Q1.5 / Q2.6: automatic benchmark of all deployable models on the CIFAR-10 test set.

For every available model (fp32, ptq, qat, unstructured, structured) it measures
  accuracy, per-class accuracy, model file size, compressed size, parameter count,
  non-zero weights / sparsity, MACs, batch-1 latency (mean / p50 / p95) and throughput.

Outputs (results/):
  benchmark_results.csv       one row per model (accuracy + inference performance)
  benchmark_quantization.csv  subset: FP32 vs PTQ vs QAT            (Question 1)
  benchmark_pruning.csv       subset: FP32 vs unstructured vs structured (Question 2)
  predictions.csv             per-image test predictions + confidence for every model
  benchmark_summary.png       bar charts used in the report

All measurements run on CPU (INT8 kernels are CPU kernels) for a fair comparison.
"""
import argparse
import os
import time
import zlib
import numpy as np
import pandas as pd
import torch
from common import *


def latency_batch1(model, test_ds, n=200, warmup=20):
    xs = [test_ds[i % len(test_ds)][0].unsqueeze(0) for i in range(n + warmup)]
    times = []
    with torch.no_grad():
        for i, x in enumerate(xs):
            t0 = time.perf_counter()
            model(x)
            dt = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                times.append(dt)
    t = np.array(times)
    return t.mean(), np.percentile(t, 50), np.percentile(t, 95)


def throughput(model, test_ds, bs=128, reps=5):
    x = torch.stack([test_ds[i % len(test_ds)][0] for i in range(bs)])
    with torch.no_grad():
        model(x)                                    # warm-up
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        dt = time.perf_counter() - t0
    return bs * reps / dt


def compressed_size_mb(path):
    with open(path, "rb") as f:
        return len(zlib.compress(f.read(), 6)) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(MODEL_FILES), choices=list(MODEL_FILES))
    ap.add_argument("--lat-samples", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(0)

    _, _, test_ds = get_datasets(args.synthetic)
    loader = torch.utils.data.DataLoader(test_ds, 256, shuffle=False, num_workers=0)
    labels = np.array([int(test_ds[i][1]) for i in range(len(test_ds))])
    ref_macs = None
    rows, preds = [], {"image_idx": np.arange(len(labels)), "true_label": labels,
                       "true_class": [CLASSES[i] for i in labels]}
    fp32_size = fp32_lat = None

    for name in args.models:
        path = model_path(name)
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing")
            continue
        print(f"[bench] {MODEL_LABELS[name]} ...")
        model = load_model(name)
        acc, p, conf = evaluate(model, loader, "cpu", return_preds=True)
        per_class = [100.0 * (p[labels == c] == c).mean() for c in range(10)]
        lat_mean, lat_p50, lat_p95 = latency_batch1(model, test_ds, args.lat_samples)
        thr = throughput(model, test_ds)
        total_w, nz_w = weight_stats(model)
        size = file_size_mb(path)
        if name in ("ptq", "qat"):                    # same architecture as FP32 baseline
            macs = ref_macs or count_macs(ResNet18())
        else:
            macs = count_macs(model)
            if name == "fp32":
                ref_macs = macs
        row = {
            "model": name, "description": MODEL_LABELS[name],
            "test_samples": len(labels), "accuracy_pct": round(acc, 2),
            "size_mb": round(size, 3), "compressed_size_mb": round(compressed_size_mb(path), 3),
            "total_weights": total_w, "nonzero_weights": nz_w,
            "sparsity_pct": round(100 * (1 - nz_w / total_w), 2),
            "macs_millions": round(macs / 1e6, 1),
            "latency_bs1_mean_ms": round(lat_mean, 3), "latency_bs1_p50_ms": round(lat_p50, 3),
            "latency_bs1_p95_ms": round(lat_p95, 3), "throughput_img_per_s": round(thr, 1),
        }
        for c, a in enumerate(per_class):
            row[f"acc_{CLASSES[c]}"] = round(a, 2)
        rows.append(row)
        preds[f"pred_{name}"] = p
        preds[f"conf_{name}"] = np.round(conf, 4)
        if name == "fp32":
            fp32_size, fp32_lat = size, lat_mean

    if not rows:
        raise SystemExit("no models found - train/quantize/prune first")
    df = pd.DataFrame(rows)
    base = df[df.model == "fp32"]
    if len(base):
        df.insert(5, "acc_delta_vs_fp32", (df.accuracy_pct - base.accuracy_pct.iloc[0]).round(2))
        df["size_reduction_x"] = (base.size_mb.iloc[0] / df.size_mb).round(2)
        df["speedup_x"] = (base.latency_bs1_mean_ms.iloc[0] / df.latency_bs1_mean_ms).round(2)

    df.to_csv(os.path.join(RESULT_DIR, "benchmark_results.csv"), index=False)
    df[df.model.isin(["fp32", "ptq", "qat"])].to_csv(
        os.path.join(RESULT_DIR, "benchmark_quantization.csv"), index=False)
    df[df.model.isin(["fp32", "unstructured", "structured"])].to_csv(
        os.path.join(RESULT_DIR, "benchmark_pruning.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(RESULT_DIR, "predictions.csv"), index=False)

    cols = ["description", "accuracy_pct", "size_mb", "compressed_size_mb", "sparsity_pct",
            "macs_millions", "latency_bs1_mean_ms", "throughput_img_per_s"]
    print("\n" + df[cols].to_string(index=False))
    print(f"\nSaved CSVs to {RESULT_DIR}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
        names = df.description.tolist()
        for a, col, title in zip(ax, ["accuracy_pct", "size_mb", "latency_bs1_mean_ms", "macs_millions"],
                                 ["Test accuracy (%)", "Model file size (MB)",
                                  "Latency, batch 1 (ms)", "MACs (millions)"]):
            a.bar(range(len(df)), df[col], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"][:len(df)])
            a.set_xticks(range(len(df)))
            a.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
            a.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "benchmark_summary.png"), dpi=150)
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()

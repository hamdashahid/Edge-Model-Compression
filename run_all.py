"""
run_all.py - one-command reproducible pipeline for Assignment 03.

  python run_all.py                # full run on CIFAR-10 (GPU strongly recommended)
  python run_all.py --quick        # 2-minute smoke test on a tiny synthetic dataset (NOT real results)

Order:  baseline -> PTQ -> QAT -> unstructured pruning -> structured pruning
        -> ONNX export -> benchmark (PyTorch) -> benchmark (ONNX) -> technical report
"""
import argparse
import subprocess
import sys
import time

PY = sys.executable


def run(cmd):
    print("\n" + "=" * 78 + f"\n$ {' '.join(cmd)}\n" + "=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + cmd)
    if r.returncode != 0:
        sys.exit(f"STEP FAILED: {' '.join(cmd)}")
    print(f"-- done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="synthetic data + 1-2 epochs (smoke test only)")
    ap.add_argument("--baseline-epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=5)
    ap.add_argument("--prune-iters", type=int, default=5)
    ap.add_argument("--prune-ft-epochs", type=int, default=2)
    ap.add_argument("--struct-ft-epochs", type=int, default=10)
    ap.add_argument("--sparsity", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    syn = ["--synthetic"] if args.quick else []
    if args.quick:
        args.baseline_epochs, args.qat_epochs = 6, 3
        args.prune_iters, args.prune_ft_epochs, args.struct_ft_epochs = 2, 1, 2
    dev = ["--device", args.device]
    w = ["--workers", "0"] if args.quick else []

    run(["train_baseline.py", "--epochs", str(args.baseline_epochs)] + dev + w + syn)
    run(["ptq_quantization.py"] + syn)
    run(["qat_quantization.py", "--epochs", str(args.qat_epochs)] + dev + w + syn)
    run(["prune_unstructured.py", "--target-sparsity", str(args.sparsity), "--iterations", str(args.prune_iters),
         "--finetune-epochs", str(args.prune_ft_epochs)] + dev + w + syn)
    run(["prune_structured.py", "--ratio", str(args.ratio), "--finetune-epochs", str(args.struct_ft_epochs)] + dev + w + syn)
    run(["export_onnx.py"] + syn)
    run(["benchmark.py"] + syn)
    run(["onnx_inference.py", "--benchmark"] + syn)
    run(["inference.py", "--model", "all", "--test-samples", "3"] + syn)
    run(["generate_report.py"])
    print("\nAll steps finished. Deliverables: models/, onnx_models/, results/, docs/")


if __name__ == "__main__":
    main()

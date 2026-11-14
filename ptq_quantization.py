"""
ptq_quantization.py - Q1.2: Post-Training Static Quantization (PTQ) of ResNet-18.

Steps
  1. load the trained FP32 checkpoint
  2. fuse Conv+BN(+ReLU) layers
  3. configure the quantization backend (fbgemm on x86 / qnnpack on ARM)
  4. insert observers (prepare) and run a CALIBRATION pass over a subset of the
     training data (no gradients, no weight updates)
  5. convert to a real INT8 model
  6. save state_dict checkpoint (.pth) and a self-contained TorchScript file (.pt)

Usage:  python ptq_quantization.py --calib-samples 2000
"""
import argparse
import copy
import os
import torch
from common import *


def calibrate(model, loader):
    """Feed representative data so observers record activation ranges."""
    model.eval()
    n = 0
    with torch.no_grad():
        for x, _ in loader:
            model(x)
            n += x.size(0)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=model_path("fp32"))
    ap.add_argument("--out", default=model_path("ptq"))
    ap.add_argument("--calib-samples", type=int, default=2000)
    ap.add_argument("--calib-batch-size", type=int, default=100)
    ap.add_argument("--backend", default=None, help="fbgemm | qnnpack | x86 (auto if omitted)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    backend = set_quant_backend(args.backend)
    print(f"[ptq] quantization backend = {backend}")
    _, test_loader = get_loaders(synthetic=args.synthetic, workers=0)

    fp32, _ = load_fp32_ckpt(args.fp32)                      # (1) load FP32
    acc_fp32 = evaluate(fp32, test_loader)
    print(f"[ptq] FP32 accuracy: {acc_fp32:.2f}%")

    model = copy.deepcopy(fp32).eval()
    model.fuse_model()                                       # (2) layer fusion
    acc_fused = evaluate(model, test_loader)
    print(f"[ptq] fused FP32 accuracy: {acc_fused:.2f}%  (must equal FP32 - fusion is exact)")

    model.qconfig = tq.get_default_qconfig(backend)          # (3) backend config
    tq.prepare(model, inplace=True)                          # (4) insert observers
    calib = get_calibration_loader(args.calib_samples, args.calib_batch_size, args.synthetic)
    n = calibrate(model, calib)
    print(f"[ptq] calibrated on {n} training images")

    tq.convert(model, inplace=True)                          # (5) -> INT8
    acc_int8 = evaluate(model, test_loader)
    print(f"[ptq] INT8 accuracy: {acc_int8:.2f}%  (drop {acc_fp32-acc_int8:+.2f} pts)")

    save_int8_ckpt(model, args.out, "ptq", backend, fp32.mid_channels)   # (6) save
    ts_path = args.out.replace(".pth", ".pt")
    try:
        torch.jit.script(model).save(ts_path)
    except Exception as e:  # tracing fallback
        print(f"[ptq] scripting failed ({type(e).__name__}); using trace")
        torch.jit.trace(model, torch.randn(1, 3, 32, 32)).save(ts_path)
    print(f"[ptq] saved {args.out} ({file_size_mb(args.out):.2f} MB) and {ts_path}")

    # sanity: the saved checkpoint must reproduce the same accuracy when re-loaded
    reloaded, _ = load_int8_ckpt(args.out)
    acc_re = evaluate(reloaded, test_loader)
    print(f"[ptq] reload check: {acc_re:.2f}% ({'OK' if abs(acc_re-acc_int8)<1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()

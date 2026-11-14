"""
train_baseline.py - Q1.1 / Q2.1: train the FP32 ResNet-18 baseline on CIFAR-10
and save the checkpoint (models/resnet18_fp32.pth).

Usage:  python train_baseline.py --epochs 30
"""
import argparse
import csv
import os
import time
import torch
from common import *


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true",
                    help="tiny fake dataset - only for smoke-testing the pipeline")
    ap.add_argument("--out", default=model_path("fp32"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[baseline] device={device}  epochs={args.epochs}  synthetic={args.synthetic}")
    train_loader, test_loader = get_loaders(args.batch_size, args.synthetic, args.workers)

    model = ResNet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    log_path = os.path.join(RESULT_DIR, "training_log_baseline.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "test_acc", "seconds"])
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            loss, tr_acc = train_one_epoch(model, train_loader, opt, device, sched)
            te_acc = evaluate(model, test_loader, device)
            sec = time.time() - t0
            print(f"  epoch {ep:3d}/{args.epochs}  loss {loss:.4f}  train {tr_acc:.2f}%  "
                  f"test {te_acc:.2f}%  ({sec:.0f}s)")
            w.writerow([ep, f"{loss:.5f}", f"{tr_acc:.3f}", f"{te_acc:.3f}", f"{sec:.1f}"])
            f.flush()

    model.cpu().eval()
    save_fp32_ckpt(model, args.out, test_acc=te_acc, epochs=args.epochs)
    print(f"[baseline] final test accuracy {te_acc:.2f}%  -> saved {args.out}")


if __name__ == "__main__":
    main()

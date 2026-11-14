"""
prune_unstructured.py - Q2.2 + Q2.4: iterative magnitude-based UNSTRUCTURED pruning.

  * L1-magnitude pruning applied GLOBALLY over every Conv2d and Linear weight
  * iterative: `--iterations` rounds of  prune -> fine-tune  following a cubic
    sparsity schedule (Zhu & Gupta, 2017) up to `--target-sparsity`
  * pruning uses weight masks (torch.nn.utils.prune); fine-tuning keeps the mask
    active so pruned weights stay zero
  * at the end the mask is made PERMANENT with prune.remove()

Usage:  python prune_unstructured.py --target-sparsity 0.8 --iterations 5 --finetune-epochs 2
"""
import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from common import *


def prunable_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_schedule(target, n):
    """Cumulative sparsity after each of n iterations (cubic schedule)."""
    return [target * (1 - (1 - (i + 1) / n) ** 3) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=model_path("fp32"))
    ap.add_argument("--out", default=model_path("unstructured"))
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=2, help="epochs after EACH pruning step")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    set_seed(2)
    device = get_device(args.device)
    train_loader, test_loader = get_loaders(args.batch_size, args.synthetic, args.workers)
    model, _ = load_fp32_ckpt(args.fp32)
    base_acc = evaluate(model, test_loader, "cpu")
    print(f"[unstructured] baseline accuracy {base_acc:.2f}%  device={device}")
    model.to(device)

    params = prunable_params(model)
    schedule = sparsity_schedule(args.target_sparsity, args.iterations)
    log = []
    prev = 0.0
    for it, target in enumerate(schedule, 1):
        # torch's `amount` (float) is a fraction of the weights that are STILL unpruned
        amount = (target - prev) / (1.0 - prev)
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        prev = target
        _, sp = sparsity_report(model)
        acc_pruned = evaluate(model, test_loader, device)
        print(f"  iter {it}/{args.iterations}: sparsity {sp:.2f}% (target {100*target:.1f}%)  "
              f"acc right after pruning {acc_pruned:.2f}%")

        # fine-tune with masks active (weight = weight_orig * mask)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        steps = args.finetune_epochs * len(train_loader)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for ep in range(args.finetune_epochs):
            loss, tr = train_one_epoch(model, train_loader, opt, device, sched)
        acc_ft = evaluate(model, test_loader, device)
        print(f"           after fine-tune: loss {loss:.4f}  test acc {acc_ft:.2f}%")
        log.append((it, round(100 * target, 2), round(sp, 2), round(acc_pruned, 2), round(acc_ft, 2)))

    # make pruning permanent: weight = weight_orig * mask, then drop mask & reparametrisation
    for m, name in params:
        prune.remove(m, name)
    assert not any(hasattr(m, "weight_mask") for m in model.modules()), "mask not removed"
    model.cpu().eval()
    rows, sp = sparsity_report(model)
    final_acc = evaluate(model, test_loader, "cpu")
    print("[unstructured] per-layer sparsity (%):")
    for name, n, z, s in rows:
        print(f"    {name:28s} {s:6.2f}   ({z}/{n})")
    print(f"[unstructured] FINAL global sparsity {sp:.2f}%  accuracy {final_acc:.2f}% "
          f"(baseline {base_acc:.2f}%)")

    save_fp32_ckpt(model, args.out, sparsity=sp, test_acc=final_acc, method="global L1 unstructured")
    with open(os.path.join(RESULT_DIR, "pruning_log_unstructured.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "target_sparsity_pct", "measured_sparsity_pct",
                    "acc_after_pruning", "acc_after_finetune"])
        w.writerows(log)
    print(f"[unstructured] saved {args.out}")


if __name__ == "__main__":
    main()

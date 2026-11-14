"""
prune_structured.py - Q2.3 + Q2.4: structured CHANNEL pruning with architecture reconstruction.

How it keeps tensor dimensions valid
  Every residual block is   x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+ skip) -> relu.
  The block's input/output width must match the skip path, so those widths are never
  touched. Only the INNER channels (the output channels of conv1, which are the input
  channels of conv2) are pruned:
      conv1.weight[keep]      (rows)        bn1.{weight,bias,mean,var}[keep]
      conv2.weight[:, keep]   (input cols)
  The three tensors shrink together, so shapes stay consistent across the interconnected
  layers. A NEW, smaller ResNet18(mid_channels=[...]) is then built and the surviving
  weights are copied in - no zero-masked channels remain at inference time.

Channel importance = L1 norm of each conv1 filter (sum |w|); lowest-norm channels are removed.

Usage:  python prune_structured.py --ratio 0.5 --finetune-epochs 10
"""
import argparse
import copy
import csv
import os
import torch
import torch.nn as nn
from common import *


@torch.no_grad()
def select_channels(block, ratio, min_channels=8):
    """Indices (sorted) of the conv1 output channels to KEEP."""
    imp = block.conv1.weight.abs().sum(dim=(1, 2, 3))          # L1 norm per filter
    n = imp.numel()
    keep_n = max(min_channels, int(round(n * (1.0 - ratio))))
    return torch.sort(torch.topk(imp, keep_n).indices).values


@torch.no_grad()
def build_pruned_model(model, ratio):
    """Return (new_small_model, keep_index_list) with weights copied from `model`."""
    keeps = [select_channels(b, ratio) for b in model.blocks()]
    new = ResNet18(10, [len(k) for k in keeps])
    new_sd, old_sd = new.state_dict(), model.state_dict()

    # 1) copy every tensor whose shape did not change
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
    # 2) slice the tensors that depend on the pruned inner channels
    bi = 0
    for si, stage in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
        for bj in range(len(stage)):
            p, keep = f"layer{si}.{bj}.", keeps[bi]
            new_sd[p + "conv1.weight"] = old_sd[p + "conv1.weight"][keep].clone()
            for t in ("weight", "bias", "running_mean", "running_var"):
                new_sd[p + "bn1." + t] = old_sd[p + "bn1." + t][keep].clone()
            new_sd[p + "conv2.weight"] = old_sd[p + "conv2.weight"][:, keep].clone()
            bi += 1
    new.load_state_dict(new_sd)
    return new.eval(), keeps


@torch.no_grad()
def verify_reconstruction(model, small, keeps):
    """The physically smaller model must be numerically identical to the original model
    with the pruned channels masked to zero (proves the surgery is correct)."""
    masked = copy.deepcopy(model).eval()
    for blk, keep in zip(masked.blocks(), keeps):
        drop = torch.ones(blk.conv1.out_channels, dtype=torch.bool)
        drop[keep] = False
        blk.conv1.weight[drop] = 0
        for t in (blk.bn1.weight, blk.bn1.bias, blk.bn1.running_mean):
            t[drop] = 0
    x = torch.randn(16, 3, 32, 32)
    diff = (masked(x) - small(x)).abs().max().item()
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=model_path("fp32"))
    ap.add_argument("--out", default=model_path("structured"))
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of inner channels removed per block")
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    set_seed(3)
    device = get_device(args.device)
    train_loader, test_loader = get_loaders(args.batch_size, args.synthetic, args.workers)
    model, _ = load_fp32_ckpt(args.fp32)
    base_acc = evaluate(model, test_loader)
    base_params = sum(p.numel() for p in model.parameters())
    base_macs = count_macs(model)
    print(f"[structured] baseline acc {base_acc:.2f}%  params {base_params/1e6:.2f}M  MACs {base_macs/1e6:.1f}M")

    small, keeps = build_pruned_model(model, args.ratio)
    diff = verify_reconstruction(model, small, keeps)
    print(f"[structured] inner channels per block: {small.mid_channels}")
    print(f"[structured] reconstruction check: max|masked-original - reconstructed| = {diff:.2e}"
          f" ({'OK' if diff < 1e-3 else 'FAIL'})")
    assert diff < 1e-3, "pruned architecture does not match masked model"

    params = sum(p.numel() for p in small.parameters())
    macs = count_macs(small)
    acc_pruned = evaluate(small, test_loader)
    print(f"[structured] after pruning (no fine-tune): acc {acc_pruned:.2f}%  "
          f"params {params/1e6:.2f}M ({100*(1-params/base_params):.1f}% fewer)  "
          f"MACs {macs/1e6:.1f}M ({100*(1-macs/base_macs):.1f}% fewer)")

    small.to(device)
    opt = torch.optim.SGD(small.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=max(args.finetune_epochs, 1),
                                                steps_per_epoch=len(train_loader))
    log = []
    for ep in range(1, args.finetune_epochs + 1):
        loss, tr = train_one_epoch(small, train_loader, opt, device, sched)
        te = evaluate(small, test_loader, device)
        print(f"  fine-tune epoch {ep}/{args.finetune_epochs}  loss {loss:.4f}  train {tr:.2f}%  test {te:.2f}%")
        log.append((ep, round(loss, 4), round(tr, 2), round(te, 2)))

    small.cpu().eval()
    final_acc = evaluate(small, test_loader)
    print(f"[structured] FINAL acc {final_acc:.2f}% (baseline {base_acc:.2f}%, "
          f"after-prune {acc_pruned:.2f}%)")
    save_fp32_ckpt(small, args.out, ratio=args.ratio, test_acc=final_acc, method="L1 channel pruning")
    with open(os.path.join(RESULT_DIR, "pruning_log_structured.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
        w.writerows(log)
    print(f"[structured] saved {args.out}")


if __name__ == "__main__":
    main()

"""
qat_quantization.py - Q1.3: Quantization-Aware Training (QAT) of ResNet-18.

Steps
  1. initialise from the pretrained FP32 model
  2. fuse Conv+BN+ReLU (QAT variant) and insert FakeQuantize modules (prepare_qat)
  3. fine-tune a few epochs so weights adapt to INT8 rounding noise
     (observers are frozen and BN statistics are frozen in the last epochs)
  4. convert the QAT model to a real INT8 model
  5. save the INT8 checkpoint (.pth + TorchScript .pt) and the fake-quant
     checkpoint (used for the ONNX export)

Usage:  python qat_quantization.py --epochs 5
"""
import argparse
import copy
import torch
from common import *


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=model_path("fp32"))
    ap.add_argument("--out", default=model_path("qat"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--freeze-observer-epoch", type=int, default=None,
                    help="epoch after which activation observers are frozen (default: epochs-2)")
    ap.add_argument("--freeze-bn-epoch", type=int, default=None,
                    help="epoch after which BN statistics are frozen (default: epochs-1)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    set_seed(1)
    backend = set_quant_backend(args.backend)
    device = get_device(args.device)
    obs_ep = args.freeze_observer_epoch if args.freeze_observer_epoch is not None else max(args.epochs - 2, 1)
    bn_ep = args.freeze_bn_epoch if args.freeze_bn_epoch is not None else max(args.epochs - 1, 1)
    print(f"[qat] backend={backend} device={device} epochs={args.epochs}")
    train_loader, test_loader = get_loaders(args.batch_size, args.synthetic, args.workers)

    fp32, _ = load_fp32_ckpt(args.fp32)                       # (1) pretrained FP32
    acc_fp32 = evaluate(fp32, test_loader)
    print(f"[qat] FP32 accuracy: {acc_fp32:.2f}%")

    model = copy.deepcopy(fp32)
    model.train()
    model.fuse_model(qat=True)                                # (2) fuse (train-mode fusion)
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)                       #     insert fake-quant modules
    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))

    for ep in range(1, args.epochs + 1):                      # (3) QAT fine-tuning
        if ep > obs_ep:
            model.apply(tq.disable_observer)
        if ep > bn_ep:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        loss, tr = train_one_epoch(model, train_loader, opt, device, sched)
        acc_fq = evaluate(model, test_loader, device)         # accuracy with fake-quant ops
        print(f"  epoch {ep}/{args.epochs}  loss {loss:.4f}  train {tr:.2f}%  "
              f"fake-quant test {acc_fq:.2f}%")

    model.cpu().eval()
    fq_path = args.out.replace("_int8.pth", "_fakequant.pth")
    torch.save({"state_dict": model.state_dict(), "backend": backend,
                "mid_channels": list(fp32.mid_channels)}, fq_path)

    int8 = tq.convert(model, inplace=False)                   # (4) -> real INT8
    acc_int8 = evaluate(int8, test_loader)
    print(f"[qat] INT8 accuracy: {acc_int8:.2f}%  (drop vs FP32 {acc_fp32-acc_int8:+.2f} pts)")

    save_int8_ckpt(int8, args.out, "qat", backend, fp32.mid_channels)    # (5) export
    ts_path = args.out.replace(".pth", ".pt")
    try:
        torch.jit.script(int8).save(ts_path)
    except Exception as e:
        print(f"[qat] scripting failed ({type(e).__name__}); using trace")
        torch.jit.trace(int8, torch.randn(1, 3, 32, 32)).save(ts_path)
    print(f"[qat] saved {args.out} ({file_size_mb(args.out):.2f} MB), {ts_path}, {fq_path}")

    reloaded, _ = load_int8_ckpt(args.out)
    acc_re = evaluate(reloaded, test_loader)
    print(f"[qat] reload check: {acc_re:.2f}% ({'OK' if abs(acc_re-acc_int8)<1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()

"""
common.py - shared utilities for the Edge Computing Assignment 03 pipeline.

Contains:
  * ResNet-18 (CIFAR-10 variant) that is quantization-friendly and supports
    custom per-block channel widths (needed to rebuild structurally pruned models)
  * CIFAR-10 data loading (+ a tiny synthetic dataset used ONLY for smoke tests)
  * training / evaluation helpers, MAC & parameter counters
  * checkpoint save / load helpers for all five model variants
"""
import os
import warnings
import time
import random
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)   # quantization / TypedStorage notices

# torch.ao.quantization is the modern namespace; fall back for old versions
try:
    import torch.ao.quantization as tq
    from torch.ao.nn.quantized import FloatFunctional
except ImportError:  # pragma: no cover
    import torch.quantization as tq
    from torch.nn.quantized import FloatFunctional

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
for _d in (MODEL_DIR, RESULT_DIR, DATA_DIR, ONNX_DIR):
    os.makedirs(_d, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# name -> checkpoint file. These five names are used by every script.
MODEL_FILES = {
    "fp32":         "resnet18_fp32.pth",
    "ptq":          "resnet18_ptq_int8.pth",
    "qat":          "resnet18_qat_int8.pth",
    "unstructured": "resnet18_unstructured_pruned.pth",
    "structured":   "resnet18_structured_pruned.pth",
}
MODEL_LABELS = {
    "fp32": "FP32 baseline", "ptq": "PTQ INT8", "qat": "QAT INT8",
    "unstructured": "Unstructured pruned", "structured": "Structured pruned",
}
DEFAULT_MID = [64, 64, 128, 128, 256, 256, 512, 512]


# --------------------------------------------------------------------------
# Reproducibility / environment
# --------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def get_quant_backend():
    """fbgemm for x86 CPUs, qnnpack for ARM (Apple silicon / Raspberry Pi)."""
    engines = torch.backends.quantized.supported_engines
    machine = platform.machine().lower()
    if ("arm" in machine or "aarch64" in machine) and "qnnpack" in engines:
        return "qnnpack"
    if "fbgemm" in engines:
        return "fbgemm"
    if "x86" in engines:
        return "x86"
    return "qnnpack"


def set_quant_backend(backend=None):
    backend = backend or get_quant_backend()
    torch.backends.quantized.engine = backend
    return backend


# --------------------------------------------------------------------------
# Model: ResNet-18 for CIFAR-10 (3x3 stem, no max-pool)
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """in -> conv1(mid) -> conv2(out) + skip.
    `mid` (inner width) can be smaller than `out` after structured pruning;
    the block's input/output width stays unchanged so skip connections remain valid."""

    def __init__(self, in_c, mid_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
        # FloatFunctional lets the residual add be quantized (plain `+` cannot)
        self.add = FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.add.add_relu(out, identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        mid_channels = list(mid_channels or DEFAULT_MID)
        assert len(mid_channels) == 8, "ResNet-18 has 8 basic blocks"
        self.mid_channels = mid_channels
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        widths, strides = [64, 128, 256, 512], [1, 2, 2, 2]
        in_c, k = 64, 0
        stages = []
        for w, s in zip(widths, strides):
            blocks = []
            for b in range(2):
                blocks.append(BasicBlock(in_c, mid_channels[k], w, s if b == 0 else 1))
                in_c, k = w, k + 1
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def blocks(self):
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                yield blk

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self, qat=False):
        """Fuse Conv+BN(+ReLU) so they quantize as one op (required for good INT8)."""
        fuse = tq.fuse_modules_qat if qat else tq.fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for blk in self.blocks():
            fuse(blk, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if blk.downsample is not None:
                fuse(blk.downsample, [["0", "1"]], inplace=True)
        return self


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _synthetic_dataset(n, seed):
    """Tiny learnable fake dataset (class-dependent colour/pattern + noise).
    ONLY used for offline smoke-testing the pipeline (--synthetic)."""
    labels = torch.arange(n) % 10
    base = torch.randn(10, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * 0.8  # shared class patterns
    g = torch.Generator().manual_seed(seed)                                              # per-split noise
    x = base[labels] + 0.5 * torch.randn(n, 3, 32, 32, generator=g)
    return torch.utils.data.TensorDataset(x, labels)


def get_datasets(synthetic=False, data_dir=DATA_DIR):
    """Returns (train_aug, train_plain, test) datasets.
    train_plain = training images WITHOUT augmentation (used for calibration)."""
    if synthetic:
        tr = _synthetic_dataset(512, 1)
        te = _synthetic_dataset(256, 2)
        return tr, tr, te
    import torchvision
    import torchvision.transforms as T
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    tf_test = T.Compose([T.ToTensor(), norm])
    train_aug = torchvision.datasets.CIFAR10(data_dir, True, tf_train, download=True)
    train_plain = torchvision.datasets.CIFAR10(data_dir, True, tf_test, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, tf_test, download=True)
    return train_aug, train_plain, test


def get_loaders(batch_size=128, synthetic=False, workers=2, test_bs=256):
    tr, _, te = get_datasets(synthetic)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        te, test_bs, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_loader, test_loader


def get_calibration_loader(num_samples=2000, batch_size=100, synthetic=False, seed=0):
    """Random subset of the TRAINING set (no augmentation) for PTQ calibration."""
    _, plain, _ = get_datasets(synthetic)
    g = np.random.RandomState(seed)
    idx = g.choice(len(plain), min(num_samples, len(plain)), replace=False)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(plain, idx.tolist()),
                                       batch_size, shuffle=False)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None, log_every=0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    batch {i+1}/{len(loader)}  loss {loss_sum/total:.4f}")
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device="cpu", return_preds=False):
    model.eval()
    correct, total = 0, 0
    preds, confs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(p.cpu())
            confs.append(F.softmax(out.float(), 1).max(1).values.cpu())
    acc = 100.0 * correct / total
    if return_preds:
        return acc, torch.cat(preds).numpy(), torch.cat(confs).numpy()
    return acc


# --------------------------------------------------------------------------
# Model statistics
# --------------------------------------------------------------------------
def count_macs(model, input_size=(1, 3, 32, 32)):
    """Multiply-accumulates of conv + linear layers (FP32 model only)."""
    macs = [0]
    hooks = []

    def conv_hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0]


def weight_stats(model):
    """(total, nonzero) element count of conv/linear weights.
    Works for FP32 and for quantized (INT8) modules."""
    total, nonzero = 0, 0
    for m in model.modules():
        w = None
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
        elif hasattr(m, "weight") and callable(m.weight):
            try:
                qw = m.weight()
                w = qw.int_repr() if qw.is_quantized else qw
            except Exception:
                w = None
        elif type(m).__module__.startswith("torch.ao.nn.intrinsic.qat") or \
                type(m).__module__.startswith("torch.ao.nn.qat"):
            w = m.weight.detach()
        if w is not None:
            total += w.numel()
            nonzero += int((w != 0).sum().item())
    return total, nonzero


def sparsity_report(model):
    """Per-layer + global sparsity for Conv2d/Linear weights of an FP32 model."""
    rows, tot, zeros = [], 0, 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            z = int((w == 0).sum().item())
            rows.append((name, w.numel(), z, 100.0 * z / w.numel()))
            tot += w.numel()
            zeros += z
    return rows, 100.0 * zeros / max(tot, 1)


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_fp32_ckpt(model, path, **meta):
    torch.save({"state_dict": model.state_dict(),
                "mid_channels": list(model.mid_channels),
                "meta": meta}, path)


def load_fp32_ckpt(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = ResNet18(10, ck.get("mid_channels"))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck.get("meta", {})


def build_int8_skeleton(mid_channels=None, backend=None, mode="ptq"):
    """Re-create the *structure* of a converted INT8 model (fuse -> prepare -> convert),
    so that a saved INT8 state_dict can be loaded into it."""
    backend = set_quant_backend(backend)
    m = ResNet18(10, mid_channels).eval()
    if mode == "qat":
        m.train()
        m.fuse_model(qat=True)
        m.qconfig = tq.get_default_qat_qconfig(backend)
        tq.prepare_qat(m, inplace=True)
    else:
        m.fuse_model()
        m.qconfig = tq.get_default_qconfig(backend)
        tq.prepare(m, inplace=True)
    m.eval()
    tq.convert(m, inplace=True)
    return m


def save_int8_ckpt(model, path, mode, backend, mid_channels):
    torch.save({"state_dict": model.state_dict(), "mode": mode, "backend": backend,
                "mid_channels": list(mid_channels)}, path)


def load_int8_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_int8_skeleton(ck["mid_channels"], ck["backend"], ck["mode"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def load_model(name, model_dir=MODEL_DIR):
    """Load any of the five deployable models, ready for CPU inference."""
    path = os.path.join(model_dir, MODEL_FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the pipeline first (see README).")
    if name in ("ptq", "qat"):
        model, _ = load_int8_ckpt(path)
    else:
        model, _ = load_fp32_ckpt(path)
    return model.eval()


def model_path(name):
    return os.path.join(MODEL_DIR, MODEL_FILES[name])

"""
export_onnx.py - Deliverable 8: export all five models to ONNX (onnx_models/).

  fp32 / unstructured / structured : torch.onnx.export of the FP32-format models
  qat INT8  : export of the QAT model with FakeQuantize ops -> ONNX QuantizeLinear/DequantizeLinear (QDQ),
              i.e. the exact INT8 scales learned during QAT
  ptq INT8  : ONNX Runtime static quantization (QDQ, per-channel weights) of the FP32 ONNX graph,
              calibrated on the same training subset used for the PyTorch PTQ model

(PyTorch eager-mode INT8 modules use fbgemm/qnnpack-specific packed ops that have no ONNX
equivalent, therefore the INT8 ONNX models are produced via QDQ graphs as above.)

Usage:  python export_onnx.py
"""
import argparse
import inspect
import os
import numpy as np
import torch
from common import *


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"resnet18_{name}.onnx")


def torch_export(model, path, opset=17):
    model = model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False                      # stable TorchScript-based exporter
    torch.onnx.export(model, dummy, path, **kw)


def load_qat_fakequant():
    ck = torch.load(model_path("qat").replace("_int8.pth", "_fakequant.pth"),
                    map_location="cpu", weights_only=False)
    backend = set_quant_backend(ck["backend"])
    m = ResNet18(10, ck["mid_channels"])
    m.train()
    m.fuse_model(qat=True)
    m.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(m, inplace=True)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.apply(tq.disable_observer)
    _make_exportable(m)
    return m


def _make_exportable(model):
    """The fused moving-average fake-quant op has no ONNX mapping. Replace each FakeQuantize's
    forward with the plain fake_quantize_* op using the SAME learned scale / zero-point
    (numerically identical once observers are frozen)."""
    import types

    def fwd(self, x):
        if getattr(self, "is_per_channel", False):
            return torch.fake_quantize_per_channel_affine(
                x, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max)
        return torch.fake_quantize_per_tensor_affine(      # 0-d scale / zp -> scalar ONNX initialisers
            x, self.scale.reshape(()), self.zero_point.reshape(()), self.quant_min, self.quant_max)

    for mod in model.modules():
        if isinstance(mod, tq.FakeQuantizeBase):
            mod.forward = types.MethodType(fwd, mod)


def ort_static_quantize(fp32_onnx, out_path, synthetic):
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                          QuantFormat, QuantType)

    class Reader(CalibrationDataReader):
        def __init__(self):
            loader = get_calibration_loader(1000, 50, synthetic)
            self.it = iter([{"input": x.numpy()} for x, _ in loader])

        def get_next(self):
            return next(self.it, None)

    quantize_static(fp32_onnx, out_path, Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    set_seed(0)

    for name in ("fp32", "unstructured", "structured"):
        if os.path.exists(model_path(name)):
            torch_export(load_model(name), onnx_path(name))
            print(f"[onnx] {name:13s} -> {onnx_path(name)}  ({file_size_mb(onnx_path(name)):.2f} MB)")

    if os.path.exists(onnx_path("fp32")):
        try:
            ort_static_quantize(onnx_path("fp32"), onnx_path("ptq"), args.synthetic)
            print(f"[onnx] {'ptq':13s} -> {onnx_path('ptq')}  ({file_size_mb(onnx_path('ptq')):.2f} MB)")
        except Exception as e:
            print(f"[onnx] PTQ export failed: {type(e).__name__}: {e}")

    fq = model_path("qat").replace("_int8.pth", "_fakequant.pth")
    if os.path.exists(fq):
        try:
            torch_export(load_qat_fakequant(), onnx_path("qat"), opset=13)
            print(f"[onnx] {'qat':13s} -> {onnx_path('qat')}  ({file_size_mb(onnx_path('qat')):.2f} MB)")
        except Exception as e:
            print(f"[onnx] QAT export failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

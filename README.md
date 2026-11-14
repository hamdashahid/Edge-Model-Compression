# Edge Model Compression - Quantization & Pruning of ResNet-18

Post-training quantization (PTQ), quantization-aware training (QAT), iterative unstructured pruning and structured
channel pruning of a CIFAR-10 ResNet-18, with a unified inference pipeline, automated benchmarking and ONNX Runtime deployment.

*Edge Computing & Intelligent Systems - Assignment 03 - Hamda (23I-0069, Section B, AI)*

**Full documentation (PDF):** [`docs/Documentation.pdf`](docs/Documentation.pdf)

## Results (CIFAR-10 test set, 10,000 images, CPU)

| Model | Top-1 acc. | File size | Latency (batch 1) | Notes |
|---|---|---|---|---|
| FP32 baseline | 93.36% | 44.8 MB | 13.9 ms | reference |
| PTQ INT8 | 93.48% | 11.3 MB | 7.1 ms | 4.0x smaller, 2.0x faster |
| QAT INT8 | 93.52% | 11.3 MB | 6.7 ms | 4.0x smaller, 2.1x faster |
| Unstructured pruned | 93.64% | 44.8 MB | 13.7 ms | 80% zero weights (dense storage, so no speed-up) |
| Structured pruned | 93.34% | 22.8 MB | 9.2 ms | 49% fewer MACs, 2.0x smaller, 1.5x faster |

Differences of a few tenths of a point are within test-set noise (about +/-0.25). Raw numbers: [`results/benchmark_results.csv`](results/benchmark_results.csv),
per-image predictions: [`results/predictions.csv`](results/predictions.csv), ONNX Runtime numbers: [`results/onnx_benchmark.csv`](results/onnx_benchmark.csv).

## What is in the repo
| Path | Contents |
|---|---|
| `train_baseline.py` | trains the FP32 ResNet-18 baseline |
| `ptq_quantization.py` | Conv-BN-ReLU fusion, backend config, calibration, INT8 conversion |
| `qat_quantization.py` | fake-quant insertion, QAT fine-tuning, INT8 conversion |
| `prune_unstructured.py` | global L1 magnitude pruning, 5 iterations, masks made permanent |
| `prune_structured.py` | channel pruning + rebuilt smaller architecture (with equivalence check) |
| `inference.py` | one inference pipeline for all 5 models (CIFAR-10 images or any image file) |
| `benchmark.py` | automatic accuracy / size / sparsity / MACs / latency benchmark -> CSV |
| `export_onnx.py`, `onnx_inference.py` | ONNX export and ONNX Runtime CPU inference |
| `run_all.py` | the whole pipeline in one command |
| `models/` | trained checkpoints (FP32, PTQ-INT8, QAT-INT8, unstructured, structured) |
| `onnx_models/` | exported ONNX models |
| `results/` | training logs, pruning logs, benchmark CSVs, charts |
| `docs/` | documentation PDF and figures |

## Quick start
```bash
pip install -r requirements.txt

# use the trained models that are already in the repo
python inference.py --model all --test-samples 5
python inference.py --model qat --image my_photo.jpg
python benchmark.py                      # downloads CIFAR-10 once, re-creates results/*.csv
python onnx_inference.py --benchmark

# or retrain everything from scratch (GPU recommended, roughly 40 min on a Kaggle GPU)
python run_all.py
```
Model names: `fp32`, `ptq`, `qat`, `unstructured`, `structured`.

Step by step:
```bash
python train_baseline.py --epochs 30
python ptq_quantization.py
python qat_quantization.py --epochs 5
python prune_unstructured.py --target-sparsity 0.8 --iterations 5
python prune_structured.py --ratio 0.5
python export_onnx.py
python benchmark.py
python generate_report.py --repo-url <this repo's URL>   # rebuilds docs/Documentation.pdf from results/
```

## Notes
- INT8 models run on CPU (`fbgemm` on x86, `qnnpack` on ARM; chosen automatically). The saved INT8 checkpoints store the backend they were built with.
- Unstructured pruning keeps dense tensors (zeros are stored), so file size and speed do not change on dense kernels - see the zip-compressed size column in the benchmark.
- Structured pruning removes the *inner* channels of every residual block, so skip-connection shapes stay valid.
- INT8 ONNX: PTQ via ONNX Runtime static QDQ quantization; QAT via exported fake-quant (QDQ) nodes.
- Python 3.10+, PyTorch 2.x (eager-mode quantization API).

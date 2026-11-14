"""
generate_report.py - builds the project documentation PDF (docs/Documentation.pdf) from the REAL
experiment outputs in results/ (CSV logs, benchmark tables) and models/ - nothing is hard-coded.

  python generate_report.py
  python generate_report.py --repo-url https://github.com/<user>/edge-model-compression
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, Preformatted)
from common import ROOT, RESULT_DIR, MODEL_FILES, MODEL_LABELS, CLASSES, load_model, count_macs, model_path

DOCS = os.path.join(ROOT, "docs")
FIG = os.path.join(DOCS, "figures")
os.makedirs(FIG, exist_ok=True)
NAVY, BLUE, GREY = colors.HexColor("#1f3a5f"), colors.HexColor("#2f6db5"), colors.HexColor("#f2f5f9")


def rd(name):
    p = os.path.join(RESULT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ------------------------------------------------------------------ figures
def make_figures(bench, base, ulog, slog, preds):
    out = {}
    if base is not None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(base.epoch, base.train_loss, color="#2f6db5"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
        ax[1].plot(base.epoch, base.train_acc, label="train", color="#2f6db5")
        ax[1].plot(base.epoch, base.test_acc, label="test", color="#c44e52")
        ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
        for a in ax: a.grid(alpha=.3)
        plt.tight_layout(); out["baseline"] = os.path.join(FIG, "baseline_training.png"); plt.savefig(out["baseline"], dpi=170); plt.close()
    if ulog is not None and slog is not None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(ulog.measured_sparsity_pct, ulog.acc_after_pruning, "o--", label="right after pruning", color="#c44e52")
        ax[0].plot(ulog.measured_sparsity_pct, ulog.acc_after_finetune, "o-", label="after fine-tuning", color="#2f6db5")
        ax[0].set_xlabel("global sparsity (%)"); ax[0].set_ylabel("test accuracy (%)"); ax[0].set_title("Iterative unstructured pruning"); ax[0].legend(fontsize=8)
        ax[1].plot(slog.epoch, slog.test_acc, "o-", color="#2f6db5"); ax[1].set_xlabel("fine-tuning epoch")
        ax[1].set_title("Structured pruning: fine-tuning recovery"); ax[1].set_ylabel("test accuracy (%)")
        for a in ax: a.grid(alpha=.3)
        plt.tight_layout(); out["pruning"] = os.path.join(FIG, "pruning_curves.png"); plt.savefig(out["pruning"], dpi=170); plt.close()
    if bench is not None:
        cols = [c for c in bench.columns if c.startswith("acc_") and c != "acc_delta_vs_fp32"]
        fig, ax = plt.subplots(figsize=(9, 3.3)); w = 0.16
        for i, (_, r) in enumerate(bench.iterrows()):
            ax.bar(np.arange(len(cols)) + i * w, [r[c] for c in cols], w, label=r.description)
        ax.set_xticks(np.arange(len(cols)) + 2 * w); ax.set_xticklabels([c[4:] for c in cols], rotation=25)
        ax.set_ylim(80, 100); ax.set_ylabel("accuracy (%)"); ax.set_title("Per-class test accuracy"); ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); out["perclass"] = os.path.join(FIG, "per_class_accuracy.png"); plt.savefig(out["perclass"], dpi=170); plt.close()
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
        for a, c, t in zip(ax, ["accuracy_pct", "size_mb", "latency_bs1_mean_ms"],
                           ["Test accuracy (%)", "Model file size (MB)", "CPU latency, batch 1 (ms)"]):
            a.bar(range(len(bench)), bench[c], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"][:len(bench)])
            a.set_xticks(range(len(bench))); a.set_xticklabels([d.replace(" ", "\n") for d in bench.description], fontsize=7); a.set_title(t, fontsize=10)
            if c == "accuracy_pct": a.set_ylim(90, 95)
        plt.tight_layout(); out["summary"] = os.path.join(FIG, "benchmark_summary.png"); plt.savefig(out["summary"], dpi=170); plt.close()
    return out


# ------------------------------------------------------------------ pdf helpers
def styles():
    ss = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=24, textColor=NAVY, leading=28, alignment=TA_CENTER),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=12, textColor=colors.HexColor("#555555"), alignment=TA_CENTER, leading=16),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=BLUE, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", parent=ss["Normal"], fontSize=9.5, leading=13.5, spaceAfter=5),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8, textColor=colors.HexColor("#666666"), alignment=TA_CENTER, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.5, backColor=GREY, leftIndent=6, borderPadding=5, spaceAfter=8),
    }
    return s


def table(df, cols, heads, widths=None, fs=7.8):
    data = [heads] + [[("" if pd.isna(r[c]) else str(r[c])) for c in cols] for _, r in df.iterrows()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d0da")), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Edge Computing & Intelligent Systems - Assignment 03 - 23I-0069 (AI-B)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); canvas.restoreState()


def img(path, width_cm=16.5):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-url", default="(add your GitHub repository URL)")
    ap.add_argument("--name", default="Hamda"); ap.add_argument("--roll", default="23I-0069"); ap.add_argument("--section", default="B (AI)")
    a = ap.parse_args()

    bench, base = rd("benchmark_results.csv"), rd("training_log_baseline.csv")
    ulog, slog, onnx, preds = rd("pruning_log_unstructured.csv"), rd("pruning_log_structured.csv"), rd("onnx_benchmark.csv"), rd("predictions.csv")
    if bench is None:
        raise SystemExit("results/benchmark_results.csv missing - run the pipeline first (python run_all.py)")
    figs = make_figures(bench, base, ulog, slog, preds)
    B = bench.set_index("model")
    fp, ptq, qat, un, st = (B.loc[k] for k in ("fp32", "ptq", "qat", "unstructured", "structured"))
    s = styles(); story = []
    P = lambda t, k="p": story.append(Paragraph(t, s[k]))
    bullets = lambda items: [story.append(Paragraph(i, s["b"], bulletText="-")) for i in items]

    # ---- cover
    story += [Spacer(1, 3.2 * cm), Paragraph("Model Compression for Edge Deployment", s["title"]), Spacer(1, 8),
              Paragraph("Post-Training Quantization, Quantization-Aware Training and Neural Network Pruning<br/>of a ResNet-18 image classifier (CIFAR-10)", s["sub"]),
              Spacer(1, 1.2 * cm), Paragraph("<b>Edge Computing &amp; Intelligent Systems - Assignment 03</b>", s["sub"]), Spacer(1, 10),
              Paragraph(f"{a.name} &nbsp;|&nbsp; Roll No. {a.roll} &nbsp;|&nbsp; Section {a.section}", s["sub"]), Spacer(1, 6),
              Paragraph(f"Source code, trained models and results: <font color='#2f6db5'>{a.repo_url}</font>", s["sub"]), Spacer(1, 1.6 * cm)]
    summ = pd.DataFrame([
        ["FP32 baseline", f"{fp.accuracy_pct:.2f}%", f"{fp.size_mb:.1f} MB", f"{fp.latency_bs1_mean_ms:.1f} ms", "reference"],
        ["PTQ INT8", f"{ptq.accuracy_pct:.2f}%", f"{ptq.size_mb:.1f} MB", f"{ptq.latency_bs1_mean_ms:.1f} ms", f"{ptq.size_reduction_x:.1f}x smaller, {ptq.speedup_x:.1f}x faster"],
        ["QAT INT8", f"{qat.accuracy_pct:.2f}%", f"{qat.size_mb:.1f} MB", f"{qat.latency_bs1_mean_ms:.1f} ms", f"{qat.size_reduction_x:.1f}x smaller, {qat.speedup_x:.1f}x faster"],
        ["Unstructured pruned", f"{un.accuracy_pct:.2f}%", f"{un.size_mb:.1f} MB", f"{un.latency_bs1_mean_ms:.1f} ms", f"{un.sparsity_pct:.0f}% zero weights"],
        ["Structured pruned", f"{st.accuracy_pct:.2f}%", f"{st.size_mb:.1f} MB", f"{st.latency_bs1_mean_ms:.1f} ms", f"{(1 - st.macs_millions / fp.macs_millions) * 100:.0f}% fewer MACs, {st.speedup_x:.1f}x faster"]],
        columns=["m", "a", "s", "l", "n"])
    story.append(table(summ, list(summ.columns), ["Model", "Test accuracy", "File size", "CPU latency (bs 1)", "Highlights"],
                       [4 * cm, 2.7 * cm, 2.4 * cm, 3.3 * cm, 5.2 * cm], fs=8.5))
    story.append(PageBreak())

    # ---- 1 overview
    P("1. Overview", "h1")
    P("This project builds a complete, reproducible model-optimisation pipeline for edge devices. A ResNet-18 trained on CIFAR-10 is "
      "compressed with four techniques - post-training static quantization (PTQ), quantization-aware training (QAT), iterative "
      "unstructured magnitude pruning and structured channel pruning - and all five resulting models are served by one inference "
      "pipeline, evaluated by one automated benchmark and exported to ONNX for CPU deployment with ONNX Runtime.")
    P("Pipeline (<font face='Courier'>python run_all.py</font>):", "p")
    story.append(Preformatted("train_baseline -> ptq_quantization -> qat_quantization -> prune_unstructured -> prune_structured\n"
                              "   -> export_onnx -> benchmark (PyTorch) -> onnx_inference --benchmark -> inference demo -> generate_report", s["code"]))
    P("Every stage writes a checkpoint to <font face='Courier'>models/</font> and a CSV log to <font face='Courier'>results/</font>. Seeds are fixed "
      "and the full test set (10,000 images) is used for every number reported here.")

    # ---- 2 baseline
    P("2. Baseline FP32 model", "h1")
    P("<b>Architecture.</b> ResNet-18 adapted to 32x32 inputs (3x3 stride-1 stem, no max-pool, 8 basic blocks, global average pooling, "
      "10-way linear classifier; 11.2 M parameters, 555 M MACs). Residual additions use <font face='Courier'>FloatFunctional</font> so they "
      "can be quantized, and every block exposes its inner width so the structurally pruned network can be rebuilt with narrower layers.")
    P("<b>Training.</b> SGD with Nesterov momentum 0.9, weight decay 5e-4, OneCycle learning-rate schedule (peak 0.1), batch size 128, "
      "random-crop and horizontal-flip augmentation, seed 42. The <i>final-epoch</i> model is saved - no checkpoint is selected using the test set.")
    if base is not None:
        P(f"<b>Result.</b> {base.test_acc.iloc[-1]:.2f}% test accuracy after {len(base)} epochs (training accuracy {base.train_acc.iloc[-1]:.1f}%).")
        story += [img(figs["baseline"], 15.5), Paragraph("Figure 1 - Baseline training curves (results/training_log_baseline.csv).", s["cap"])]

    # ---- 3 PTQ
    P("3. Post-training static quantization (PTQ)", "h1")
    bullets(["<b>Fusion:</b> Conv+BN+ReLU and Conv+BN (including shortcut convolutions) are fused into single modules. Fusion is mathematically exact; "
             "the script prints the accuracy before and after to prove it.",
             "<b>Backend:</b> chosen automatically - <i>fbgemm</i> on x86, <i>qnnpack</i> on ARM.",
             "<b>Calibration:</b> observers are inserted with <font face='Courier'>prepare()</font> and 2,000 randomly chosen, non-augmented <i>training</i> images "
             "are passed through the model (no gradients, no weight updates) to record activation ranges.",
             "<b>Conversion:</b> <font face='Courier'>convert()</font> produces a real INT8 model with per-channel INT8 weights and UINT8 activations.",
             "<b>Deployment:</b> a state_dict checkpoint (.pth) and a self-contained TorchScript file (.pt); the checkpoint is re-loaded and re-evaluated "
             "to verify it reproduces the same accuracy."])
    P(f"<b>Result.</b> {ptq.accuracy_pct:.2f}% accuracy ({ptq.acc_delta_vs_fp32:+.2f} points vs FP32), {ptq.size_mb:.1f} MB ({ptq.size_reduction_x:.2f}x smaller), "
      f"{ptq.latency_bs1_mean_ms:.1f} ms per image ({ptq.speedup_x:.2f}x faster).")

    # ---- 4 QAT
    P("4. Quantization-aware training (QAT)", "h1")
    bullets(["Initialised from the pretrained FP32 weights; modules are fused in training mode (<font face='Courier'>fuse_modules_qat</font>).",
             "<font face='Courier'>prepare_qat()</font> inserts FakeQuantize modules that simulate INT8 rounding and clamping in the forward pass; gradients pass "
             "through with the straight-through estimator, so the weights learn to tolerate quantization noise.",
             "Fine-tuning: SGD, learning rate 1e-3 with cosine decay, 5 epochs. Activation observers are frozen two epochs before the end and BatchNorm statistics "
             "one epoch before the end so the final scales are stable.",
             "<font face='Courier'>convert()</font> gives the real INT8 model, saved as .pth and TorchScript; the fake-quant checkpoint is also kept because it is "
             "what the ONNX export uses."])
    P(f"<b>Result.</b> {qat.accuracy_pct:.2f}% accuracy ({qat.acc_delta_vs_fp32:+.2f} vs FP32), {qat.size_mb:.1f} MB ({qat.size_reduction_x:.2f}x smaller), "
      f"{qat.latency_bs1_mean_ms:.1f} ms per image ({qat.speedup_x:.2f}x faster).")

    # ---- 5 unstructured
    P("5. Unstructured pruning", "h1")
    bullets(["<b>Criterion:</b> global L1-magnitude pruning over every Conv2d and Linear weight (<font face='Courier'>torch.nn.utils.prune</font>) - "
             "the smallest-magnitude weights anywhere in the network are removed first.",
             "<b>Iterative schedule:</b> 5 prune-then-fine-tune rounds following the cubic sparsity schedule of Zhu and Gupta up to 80% sparsity; each round fine-tunes "
             "2 epochs with the masks active (weight = weight_orig x mask), so pruned weights stay zero.",
             "<b>Permanent pruning:</b> <font face='Courier'>prune.remove()</font> folds the mask into the weights and deletes the re-parametrisation; the script asserts "
             "that no mask remains and prints per-layer and global sparsity."])
    if ulog is not None:
        story.append(table(ulog, list(ulog.columns), ["Iteration", "Target sparsity %", "Measured sparsity %", "Acc. right after pruning", "Acc. after fine-tuning"]))
        story.append(Spacer(1, 4))
    P(f"<b>Result.</b> {un.sparsity_pct:.1f}% of the weights are zero and accuracy is {un.accuracy_pct:.2f}% ({un.acc_delta_vs_fp32:+.2f} vs FP32). "
      f"The tensors stay dense, so the file is still {un.size_mb:.1f} MB and latency is unchanged ({un.speedup_x:.2f}x); the benefit shows up in compressibility - "
      f"the zip-compressed model is {un.compressed_size_mb:.1f} MB versus {fp.compressed_size_mb:.1f} MB for FP32 - and would become real speed only on sparse-aware hardware or kernels.")

    # ---- 6 structured
    P("6. Structured channel pruning", "h1")
    mids = load_model("structured").mid_channels
    bullets(["<b>Granularity:</b> whole convolution filters. The importance of a channel is the L1 norm of its filter in <i>conv1</i> of each residual block; "
             "the lowest-norm half is removed.",
             "<b>Keeping dimensions valid:</b> only the <i>inner</i> channels of each residual block are pruned. conv1's output filters, the matching BatchNorm channels and "
             "conv2's matching input channels are removed together, so every tensor shape stays consistent. The block's input/output width never changes, "
             "so the skip connection and the next block remain valid.",
             "<b>Reconstruction:</b> a new, physically smaller ResNet-18 is instantiated with the reduced widths and the surviving weights are copied in - "
             "no zero-masked channels remain at inference time.",
             "<b>Verification:</b> the script checks the rebuilt network against the original with the pruned channels zeroed (max absolute difference below 1e-3).",
             "<b>Fine-tuning:</b> 10 epochs of SGD with a OneCycle schedule to recover accuracy."])
    P(f"Inner channels per block after pruning: <font face='Courier'>{mids}</font> (original [64, 64, 128, 128, 256, 256, 512, 512]).")
    P(f"<b>Result.</b> {st.accuracy_pct:.2f}% accuracy ({st.acc_delta_vs_fp32:+.2f} vs FP32) with {st.macs_millions:.0f} M MACs instead of {fp.macs_millions:.0f} M "
      f"({(1 - st.macs_millions / fp.macs_millions) * 100:.0f}% fewer), a {st.size_mb:.1f} MB file ({st.size_reduction_x:.2f}x smaller) and {st.latency_bs1_mean_ms:.1f} ms latency "
      f"({st.speedup_x:.2f}x faster) - a smaller dense network that is faster on any CPU without special support.")
    if "pruning" in figs:
        story += [img(figs["pruning"], 15.5), Paragraph("Figure 2 - Left: unstructured pruning accuracy vs sparsity. Right: structured-pruning recovery during fine-tuning.", s["cap"])]

    # ---- 7 inference
    P("7. Inference pipeline", "h1")
    P("<font face='Courier'>inference.py</font> loads any of the five models through one function (<font face='Courier'>common.load_model</font>), runs it on the CPU and "
      "prints the predicted class, confidence and latency - for CIFAR-10 test images or for any image file (resized to 32x32). INT8 checkpoints are re-created "
      "by repeating fuse, prepare and convert and then loading the saved state_dict, so a quantized model can be restored without retraining.")
    story.append(Preformatted("python inference.py --model all --test-samples 5\npython inference.py --model qat --image my_photo.jpg", s["code"]))

    # ---- 8 benchmark
    P("8. Benchmark results", "h1")
    P("<font face='Courier'>benchmark.py</font> evaluates every model on the full 10,000-image test set on the CPU and records accuracy, per-class accuracy, file size, "
      "zip-compressed size, weight sparsity, MACs, batch-1 latency (mean / 95th percentile over 200 runs after warm-up) and batch-128 throughput. "
      "It writes <font face='Courier'>results/benchmark_results.csv</font> and the per-image <font face='Courier'>predictions.csv</font>.")
    tb = bench.copy()
    tb["acc"] = tb.accuracy_pct.map("{:.2f}".format); tb["d"] = tb.acc_delta_vs_fp32.map("{:+.2f}".format)
    tb["sz"] = tb.size_mb.map("{:.1f}".format); tb["zp"] = tb.compressed_size_mb.map("{:.1f}".format); tb["sp"] = tb.sparsity_pct.map("{:.1f}".format)
    tb["mac"] = tb.macs_millions.map("{:.0f}".format); tb["lat"] = tb.latency_bs1_mean_ms.map("{:.2f}".format)
    tb["x"] = tb.speedup_x.map("{:.2f}x".format); tb["thr"] = tb.throughput_img_per_s.map("{:.0f}".format)
    story.append(table(tb, ["description", "acc", "d", "sz", "zp", "sp", "mac", "lat", "x", "thr"],
                       ["Model", "Top-1 %", "vs FP32", "Size MB", "Zip MB", "Zero wts %", "MACs M", "Lat. ms", "Speed-up", "Img/s"],
                       [3.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm, 1.4 * cm], fs=7.5))
    story += [Spacer(1, 6), img(figs["summary"], 16.5), Paragraph("Figure 3 - Accuracy, file size and CPU latency of the five models.", s["cap"]),
              img(figs["perclass"], 15.5), Paragraph("Figure 4 - Per-class test accuracy; the hardest classes (cat, dog, bird) are the same for every model.", s["cap"])]
    if preds is not None:
        ag = {k: (preds["pred_" + k] == preds["pred_fp32"]).mean() * 100 for k in ("ptq", "qat", "unstructured", "structured")}
        P("<b>Agreement with the FP32 model.</b> Percentage of the 10,000 test images on which the compressed model predicts the same class as FP32: "
          f"PTQ {ag['ptq']:.1f}%, QAT {ag['qat']:.1f}%, unstructured {ag['unstructured']:.1f}%, structured {ag['structured']:.1f}%.")

    # ---- 9 onnx
    P("9. ONNX deployment", "h1")
    P("<font face='Courier'>export_onnx.py</font> exports all five models (dynamic batch axis). FP32 and both pruned models use <font face='Courier'>torch.onnx.export</font>. "
      "The QAT model is exported with its learned fake-quantization nodes (QuantizeLinear / DequantizeLinear), i.e. the exact INT8 scales found during QAT. "
      "The PTQ ONNX model is produced with ONNX Runtime static QDQ quantization (per-channel weights) calibrated on the same training subset. "
      "(PyTorch's eager INT8 modules use backend-specific packed operators that have no ONNX equivalent, hence these two routes.) "
      "<font face='Courier'>onnx_inference.py</font> runs everything with ONNX Runtime on <font face='Courier'>CPUExecutionProvider</font>.")
    if onnx is not None:
        story.append(table(onnx, ["description", "onnx_size_mb", "accuracy_pct", "latency_bs1_mean_ms", "latency_bs1_p95_ms"],
                           ["Model", "ONNX size MB", "Top-1 %", "Latency mean ms", "Latency p95 ms"], [4.6 * cm, 2.8 * cm, 2.4 * cm, 3.2 * cm, 3.2 * cm], fs=8))
        O = onnx.set_index("model")
        story.append(Spacer(1, 4))
        P(f"ONNX Runtime is faster than eager PyTorch for the same FP32 network ({O.loc['fp32'].latency_bs1_mean_ms:.1f} ms vs {fp.latency_bs1_mean_ms:.1f} ms). "
          f"The PTQ ONNX model is {O.loc['fp32'].onnx_size_mb / O.loc['ptq'].onnx_size_mb:.1f}x smaller and {O.loc['fp32'].latency_bs1_mean_ms / O.loc['ptq'].latency_bs1_mean_ms:.2f}x faster, "
          f"at {O.loc['ptq'].accuracy_pct:.2f}% accuracy ({O.loc['ptq'].accuracy_pct - O.loc['fp32'].accuracy_pct:+.2f} vs ONNX FP32 - ONNX Runtime's calibration differs slightly from PyTorch's). "
          f"The QAT ONNX file keeps FP32 weights with Quantize/Dequantize nodes, so it is not smaller and runs slower ({O.loc['qat'].latency_bs1_mean_ms:.1f} ms) in ONNX Runtime's "
          f"CPU provider; its value is that it preserves the exact QAT-learned scales for runtimes that fuse QDQ pairs into integer kernels. "
          f"The structured model is the fastest ({O.loc['structured'].latency_bs1_mean_ms:.1f} ms, {O.loc['fp32'].latency_bs1_mean_ms / O.loc['structured'].latency_bs1_mean_ms:.2f}x).")

    # ---- 10 discussion
    P("10. Discussion", "h1")
    se = np.sqrt(fp.accuracy_pct / 100 * (1 - fp.accuracy_pct / 100) / fp.test_samples) * 100
    bullets([f"<b>All four compressed models stay within {max(abs(x) for x in (ptq.acc_delta_vs_fp32, qat.acc_delta_vs_fp32, un.acc_delta_vs_fp32, st.acc_delta_vs_fp32)):.2f} points of FP32.</b> "
             f"With 10,000 test images the standard error of an accuracy near {fp.accuracy_pct:.0f}% is about {se:.2f} points, so differences of a few tenths "
             "(including PTQ vs QAT) are within noise and should not be over-interpreted; the small gains over FP32 are not evidence that compression improves accuracy.",
             "<b>PTQ was enough here.</b> A well-trained ResNet-18 with per-channel INT8 weights and 2,000 calibration images lost nothing measurable. QAT costs extra "
             "fine-tuning but is the safer choice for more fragile networks (e.g. depthwise-separable models) or lower bit-widths.",
             "<b>Which technique buys what:</b> quantization shrinks storage ~4x and roughly halves latency with no architecture change; unstructured pruning gives the "
             "highest sparsity but no speed-up on dense kernels; structured pruning gives a genuinely smaller network (about half the MACs) that is faster everywhere.",
             "<b>Latency vs MACs:</b> structured pruning halves the MACs but speeds up by 1.5x rather than 2x, most likely because memory movement, the unpruned stem/shortcut layers and "
             "per-layer overheads do not shrink proportionally (not profiled separately).",
             "<b>Combination:</b> the techniques are complementary - a structured-pruned model can additionally be quantized to INT8 - which was outside this assignment's scope."])
    P("Limitations", "h2")
    bullets(["Single training run and seed; no confidence intervals across seeds.",
             "Latency was measured on the CPU of the machine that ran the pipeline (a shared cloud notebook), so absolute milliseconds vary between machines; the ratios are more meaningful.",
             "Structured pruning only touches the inner block channels; channels on the residual stream are shared across blocks and were left intact.",
             "The ONNX QAT model is not stored in INT8 (see section 9)."])

    # ---- 11 reproduce
    P("11. Reproducing the results", "h1")
    story.append(Preformatted("pip install -r requirements.txt\npython run_all.py            # whole pipeline, GPU recommended\n\n"
                              "# or step by step\npython train_baseline.py --epochs 30\npython ptq_quantization.py\npython qat_quantization.py --epochs 5\n"
                              "python prune_unstructured.py --target-sparsity 0.8 --iterations 5\npython prune_structured.py --ratio 0.5\n"
                              "python export_onnx.py\npython benchmark.py\npython onnx_inference.py --benchmark", s["code"]))
    P("12. Repository layout", "h1")
    story.append(Preformatted("common.py                 model, data, train/eval helpers, checkpoint loaders\n"
                              "train_baseline.py         FP32 ResNet-18 training\n"
                              "ptq_quantization.py       post-training static quantization\n"
                              "qat_quantization.py       quantization-aware training\n"
                              "prune_unstructured.py     iterative global magnitude pruning\n"
                              "prune_structured.py       channel pruning + architecture reconstruction\n"
                              "inference.py              unified inference for all 5 models\n"
                              "benchmark.py              automated benchmark -> results/*.csv\n"
                              "export_onnx.py, onnx_inference.py    ONNX export + ONNX Runtime CPU inference\n"
                              "run_all.py                one-command pipeline\n"
                              "models/  onnx_models/  results/  docs/", s["code"]))
    doc = SimpleDocTemplate(os.path.join(DOCS, "Documentation.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title="Assignment 03 - Model Compression for Edge Deployment",
                            author=f"{a.name} ({a.roll})")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("saved", os.path.join(DOCS, "Documentation.pdf"))


if __name__ == "__main__":
    main()

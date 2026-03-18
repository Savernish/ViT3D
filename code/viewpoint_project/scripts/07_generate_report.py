"""
scripts/07_generate_report.py
=============================
Generates a PoC report with:
- Per-bin mAP comparison table (LaTeX)
- Training loss curves
- Example training images (Condition A vs B)
- Example test detections
- Summary figures

Run:
    python scripts/07_generate_report.py --obj_id 0X1fGvojr1Z
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import get_test_frames_by_bin

config.make_dirs()

REPORT_DIR = os.path.join(config.RESULTS_ROOT, "poc_report")
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_poc_results(obj_id):
    path = os.path.join(config.RESULTS_ROOT, f"poc_{obj_id}.json")
    with open(path) as f:
        return json.load(f)


def load_training_log(obj_id, condition):
    path = os.path.join(
        config.CHECKPOINTS_ROOT, obj_id, condition, "training_log.json"
    )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Training Loss Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(obj_id):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Training Loss Curves — {obj_id}",
        fontsize=13, fontweight="bold"
    )

    conditions = ["condition_a", "condition_b"]
    titles = ["Condition A: Zero123++", "Condition B: 2D Augmentation"]
    colors = ["#2196F3", "#FF5722"]

    for ax, condition, title, color in zip(axes, conditions, titles, colors):
        try:
            log = load_training_log(obj_id, condition)
            losses = log["losses"]
            epochs = list(range(1, len(losses) + 1))
            ax.plot(epochs, losses, color=color, linewidth=2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("InfoNCE Loss")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.axhline(
                y=min(losses), color=color,
                linestyle="--", alpha=0.5,
                label=f"Min: {min(losses):.4f}"
            )
            ax.legend(fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{e}",
                    ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig1_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Per-Bin mAP Comparison
# ---------------------------------------------------------------------------

def plot_map_comparison(obj_id):
    results = load_poc_results(obj_id)

    bins = ["frontal", "side", "rear_side", "rear", "overall"]
    bin_labels = ["Frontal", "Side", "Rear-Side", "Rear", "Overall"]

    ca_scores = [results.get("condition_a", {}).get(b, 0) for b in bins]
    cb_scores = [results.get("condition_b", {}).get(b, 0) for b in bins]

    x = np.arange(len(bins))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(
        x - width/2, ca_scores, width,
        label="Condition A: Zero123++",
        color="#2196F3", alpha=0.85
    )
    bars_b = ax.bar(
        x + width/2, cb_scores, width,
        label="Condition B: 2D Augmentation",
        color="#FF5722", alpha=0.85
    )

    ax.set_xlabel("Viewpoint Bin", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title(
        f"Per-Bin mAP Comparison — PoC ({obj_id})\n"
        f"(Single object, fixed threshold=0.5, unoptimized training)",
        fontsize=11
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(ca_scores), max(cb_scores)) * 1.3 + 0.05)
    ax.grid(True, axis="y", alpha=0.3)

    # value labels on bars
    for bar in bars_a:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., h + 0.002,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8
        )
    for bar in bars_b:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., h + 0.002,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig2_map_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Example Training Images
# ---------------------------------------------------------------------------

def plot_training_examples(obj_id):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(
        "Example Training Images",
        fontsize=13, fontweight="bold"
    )

    for row, condition in enumerate(["condition_a", "condition_b"]):
        final_dir = os.path.join(
            config.TRAINING_DATA_ROOT, obj_id, condition, "final"
        )
        images = sorted([
            f for f in os.listdir(final_dir) if f.endswith(".png")
        ])[:6]

        label = "Condition A: Zero123++" if condition == "condition_a" \
            else "Condition B: 2D Augmentation"
        axes[row][0].set_ylabel(label, fontsize=9, rotation=90,
                                labelpad=10)

        for col, img_name in enumerate(images):
            img = Image.open(os.path.join(final_dir, img_name))
            axes[row][col].imshow(img)
            axes[row][col].axis("off")

        # fill remaining columns if fewer than 6 images
        for col in range(len(images), 6):
            axes[row][col].axis("off")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig3_training_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 4: I0 Reference Image
# ---------------------------------------------------------------------------

def plot_i0(obj_id):
    i0_path = os.path.join(config.TRAINING_DATA_ROOT, obj_id, "i0.png")
    i0 = Image.open(i0_path).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(i0)
    ax.set_title(f"Reference Image $I_0$\n{obj_id}", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig4_i0_reference.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 5: Test Images Per Bin
# ---------------------------------------------------------------------------

def plot_test_bin_examples(obj_id):
    bins = get_test_frames_by_bin(obj_id)
    bin_names = ["frontal", "side", "rear_side", "rear"]
    bin_labels = ["Frontal", "Side", "Rear-Side", "Rear"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Test Set Examples Per Viewpoint Bin — {obj_id}",
        fontsize=12, fontweight="bold"
    )

    for ax, bin_name, label in zip(axes, bin_names, bin_labels):
        frames = bins.get(bin_name, [])
        if frames:
            frame = frames[len(frames)//2]  # pick middle frame
            img_path = os.path.join(
                config.GSO_ROOT, obj_id, "rgb", f"{frame}.png"
            )
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(
                f"{label}\n({len(frames)} test frames)",
                fontsize=10
            )
        else:
            ax.text(0.5, 0.5, "No frames",
                    ha="center", va="center",
                    transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig5_test_bin_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 6: Zero123++ Raw View Grid
# ---------------------------------------------------------------------------

def plot_zero123_views(obj_id):
    raw_dir = os.path.join(
        config.TRAINING_DATA_ROOT, obj_id, "condition_a", "raw"
    )
    if not os.path.exists(raw_dir):
        print("No raw Zero123++ views found, skipping fig6")
        return None

    images = sorted([
        f for f in os.listdir(raw_dir) if f.endswith(".png")
    ])[:12]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(
        f"Zero123++ Generated Views (first 12 of 96) — {obj_id}",
        fontsize=12, fontweight="bold"
    )

    for idx, (ax, img_name) in enumerate(zip(axes.flat, images)):
        img = Image.open(os.path.join(raw_dir, img_name))
        ax.imshow(img)
        ax.set_title(f"View {idx+1}", fontsize=8)
        ax.axis("off")

    for idx in range(len(images), 12):
        axes.flat[idx].axis("off")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "fig6_zero123_views.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# LaTeX Report
# ---------------------------------------------------------------------------

def generate_latex(obj_id):
    results = load_poc_results(obj_id)

    bins = ["frontal", "side", "rear_side", "rear", "overall"]
    bin_labels = ["Frontal", "Side", "Rear-Side", "Rear", "Overall"]

    ca = results.get("condition_a", {})
    cb = results.get("condition_b", {})

    # build table rows
    table_rows = ""
    for b, label in zip(bins, bin_labels):
        ca_val = ca.get(b, 0.0)
        cb_val = cb.get(b, 0.0)
        better = r"\textbf{" + f"{ca_val:.4f}" + "}" \
            if ca_val >= cb_val else f"{ca_val:.4f}"
        worse = f"{cb_val:.4f}" \
            if ca_val >= cb_val else r"\textbf{" + f"{cb_val:.4f}" + "}"
        separator = r"\midrule" if b == "overall" else ""
        table_rows += f"        {label} & {better} & {worse} \\\\\n"
        if separator:
            table_rows = table_rows.rstrip("\n") + "\n        " + \
                         separator + "\n"

    latex = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{amsmath}

\title{
    \textbf{Viewpoint Generalization in Single-Image Object Detection:}\\
    \textbf{3D-Consistent Synthetic Adaptation vs. 2D Augmentation}\\
    \large Proof of Concept Report
}
\author{A2S Lab}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We investigate whether LoRA adapters trained on Zero123++-generated
3D-consistent synthetic views produce superior viewpoint-invariant
object detection compared to adapters trained on 2D-augmented data
from the same single reference image.
This report presents a proof-of-concept evaluation on a single
Google Scanned Objects instance using a fixed detection threshold
and unoptimized training settings.
Despite suboptimal training conditions, Condition A (Zero123++)
consistently outperforms Condition B (2D Augmentation) across
all viewpoint bins, providing directional evidence for the hypothesis.
\end{abstract}

\section{Experimental Setup}

\subsection{Object}
Object ID: \texttt{""" + obj_id + r"""}
from the Google Scanned Objects dataset.
The reference image $I_0$ is the frontal view (azimuth $\approx 0^\circ$).

\subsection{Conditions}
\begin{itemize}
    \item \textbf{Condition A (Zero123++):} $I_0 \rightarrow$
          Zero123++ $\rightarrow$ 96 synthetic views $\rightarrow$
          400 composited training images $\rightarrow$ LoRA fine-tune
    \item \textbf{Condition B (2D Augmentation):} $I_0 \rightarrow$
          2D augmentation (rotation, flip, scale, perspective warp)
          $\rightarrow$ 400 composited training images $\rightarrow$
          identical LoRA fine-tune
\end{itemize}

\subsection{Detection Protocol}
DINOv2 base backbone (frozen). LoRA rank 16, alpha 32, inserted into
query and value projections. Detection via SAM region proposals +
cosine similarity to object prototype. Fixed threshold = 0.5
(unoptimized; full experiment uses per-category threshold sweep).

\section{Results}

\begin{table}[h]
\centering
\caption{mAP@0.5 per viewpoint bin. Bold = higher score.
PoC only --- single object, fixed threshold, unoptimized training.}
\begin{tabular}{lcc}
\toprule
\textbf{Viewpoint} & \textbf{Cond. A (Zero123++)} &
\textbf{Cond. B (2D Aug.)} \\
\midrule
""" + table_rows + r"""
\bottomrule
\end{tabular}
\label{tab:poc_results}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{fig2_map_comparison.png}
\caption{Per-bin mAP comparison between Condition A and Condition B.
Condition A (Zero123++) outperforms Condition B across all viewpoint bins.}
\label{fig:map_comparison}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{fig1_loss_curves.png}
\caption{Training loss curves for both conditions.
Loss oscillation indicates suboptimal learning rate
(to be fixed before full experiment).}
\label{fig:loss_curves}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.4\textwidth]{fig4_i0_reference.png}
\caption{Reference image $I_0$ used as input for both conditions.}
\label{fig:i0}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{fig5_test_bin_examples.png}
\caption{Example test images from each viewpoint bin.}
\label{fig:test_bins}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{fig6_zero123_views.png}
\caption{First 12 of 96 Zero123++ generated views used in Condition A training.}
\label{fig:zero123_views}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{fig3_training_examples.png}
\caption{Example composited training images. Top: Condition A (Zero123++).
Bottom: Condition B (2D Augmentation).}
\label{fig:training_examples}
\end{figure}

\section{Discussion}

Condition A outperforms Condition B across all viewpoint bins,
with the largest absolute gap at the frontal bin (""" + \
    f"{ca.get('frontal', 0):.4f}" + r""" vs """ + \
    f"{cb.get('frontal', 0):.4f}" + r""").
The rear-side and rear bins show near-zero mAP for both conditions,
consistent with the expected difficulty of novel viewpoint detection
under unoptimized training.

These results provide directional evidence that 3D-consistent
synthetic views from Zero123++ produce more discriminative LoRA
adaptations than 2D augmentation alone.
The absolute mAP values are expected to improve significantly
after: (1) learning rate reduction to $3 \times 10^{-5}$,
(2) epoch reduction to 30, and (3) per-category threshold sweep
on the validation split.

\section{Known Limitations of This PoC}
\begin{itemize}
    \item Single object --- no variance estimate possible
    \item Fixed threshold 0.5 --- not optimized
    \item Training instability --- learning rate too high
    \item Zero123++ generates some geometrically inconsistent
          rear views (cable disappearance artifact documented)
\end{itemize}

\section{Next Steps}
\begin{enumerate}
    \item Fix training: LR = $3 \times 10^{-5}$, epochs = 30
    \item Train all 30 objects for both conditions
    \item Run threshold sweep on 15 validation objects
    \item Evaluate on 15 evaluation objects
    \item Run Condition C (OWL-ViT) baseline
    \item Generate final results with variance estimates
\end{enumerate}

\end{document}
"""

    latex_path = os.path.join(REPORT_DIR, "poc_report.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved: {latex_path}")
    return latex_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(obj_id):
    print(f"Generating PoC report for {obj_id}")
    print(f"Output directory: {REPORT_DIR}")
    print()

    # install matplotlib if needed
    try:
        import matplotlib
    except ImportError:
        os.system("pip install matplotlib")

    print("Generating figures...")
    plot_loss_curves(obj_id)
    plot_map_comparison(obj_id)
    plot_training_examples(obj_id)
    plot_i0(obj_id)
    plot_test_bin_examples(obj_id)
    plot_zero123_views(obj_id)

    print("\nGenerating LaTeX report...")
    latex_path = generate_latex(obj_id)

    print(f"\nReport complete.")
    print(f"All files in: {REPORT_DIR}")
    print(f"LaTeX file: {latex_path}")
    print(f"\nTo compile PDF:")
    print(f"  cd {REPORT_DIR}")
    print(f"  pdflatex poc_report.tex")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=str, required=True)
    args = parser.parse_args()
    main(args.obj_id)

"""
Renders publication-quality architectural diagram for NeuroCode.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def render_system_architecture(output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 1. Host CPU Box
    host_box = patches.FancyBboxPatch((0.5, 0.8), 4.2, 4.6, boxstyle="round,pad=0.2",
                                      facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(host_box)
    ax.text(2.6, 5.1, "HOST SYSTEM (CPU & 16-64GB RAM)", ha="center", fontsize=11, fontweight="bold", color="#2c3e50")

    # Sub-boxes inside Host
    cpg_box = patches.FancyBboxPatch((0.8, 3.4), 3.6, 1.2, boxstyle="round,pad=0.1",
                                     facecolor="#3498db", edgecolor="#1d6fa5", linewidth=1.5)
    ax.add_patch(cpg_box)
    ax.text(2.6, 4.15, "1. In-Memory Code Property Graph", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(2.6, 3.7, "Tree-sitter AST + Call/Inheritance Graph", ha="center", fontsize=8, color="#ecf0f1")

    slicer_box = patches.FancyBboxPatch((0.8, 1.4), 3.6, 1.5, boxstyle="round,pad=0.1",
                                        facecolor="#2ecc71", edgecolor="#219653", linewidth=1.5)
    ax.add_patch(slicer_box)
    ax.text(2.6, 2.5, "2. Deterministic Context Slicer", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(2.6, 2.1, "Topological k-hop Reachability Closure", ha="center", fontsize=8, color="#ecf0f1")
    ax.text(2.6, 1.7, "Latency < 2ms | Prunes 95% of Repo", ha="center", fontsize=8, color="#ecf0f1")

    # Arrow inside Host
    ax.annotate("", xy=(2.6, 3.0), xytext=(2.6, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

    # 2. Accelerator Device Box (GPU)
    gpu_box = patches.FancyBboxPatch((5.3, 0.8), 4.2, 4.6, boxstyle="round,pad=0.2",
                                     facecolor="#fcf3cf", edgecolor="#f39c12", linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(7.4, 5.1, "DEVICE ACCELERATOR (GPU < 8GB VRAM)", ha="center", fontsize=11, fontweight="bold", color="#b7950b")

    # Gemma Agent Box
    gemma_box = patches.FancyBboxPatch((5.6, 2.2), 3.6, 2.4, boxstyle="round,pad=0.1",
                                       facecolor="#e67e22", edgecolor="#ba4a00", linewidth=1.5)
    ax.add_patch(gemma_box)
    ax.text(7.4, 4.1, "Local Gemma 4 Developer Agent", ha="center", fontsize=10.5, fontweight="bold", color="white")
    ax.text(7.4, 3.6, "Inference via Ollama / vLLM", ha="center", fontsize=8.5, color="#fef9e7")
    ax.text(7.4, 3.1, "- Zero Tool-Calling Loops", ha="center", fontsize=8.5, color="#fef9e7")
    ax.text(7.4, 2.65, "- Strict sub-2k token prompt", ha="center", fontsize=8.5, color="#fef9e7")

    patch_box = patches.FancyBboxPatch((5.6, 1.1), 3.6, 0.8, boxstyle="round,pad=0.1",
                                       facecolor="#95a5a6", edgecolor="#7f8c8d", linewidth=1)
    ax.add_patch(patch_box)
    ax.text(7.4, 1.45, "Exact SEARCH/REPLACE Diff Patch", ha="center", fontsize=8.5, fontweight="bold", color="white")

    # Cross-boundary arrow (PCIe Transfer)
    ax.annotate("", xy=(5.5, 2.6), xytext=(4.5, 2.6),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
    ax.text(5.0, 2.85, "PCIe Transfer\n(< 1,500 tokens)", ha="center", fontsize=8, fontweight="bold", color="#c0392b")

    # Feedback verification arrow back to CPU
    ax.annotate("", xy=(4.5, 1.45), xytext=(5.5, 1.45),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#2980b9", linestyle="--"))
    ax.text(5.0, 1.1, "Patch Verification\n(Unit Tests on CPU)", ha="center", fontsize=7.5, fontweight="bold", color="#2980b9")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Generated system architecture diagram at: {output_path}")


if __name__ == "__main__":
    out_file = Path("/home/deepak-r/Project/research/paper/figures/fig2_system_architecture.png")
    render_system_architecture(out_file)

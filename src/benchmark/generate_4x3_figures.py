"""
Generates publication-quality 4:3 aspect ratio figures for Kaggle Writeup & Paper.
Renders:
1. paper/figures/fig1_empirical_comparison_4x3.png (figsize=(8, 6))
2. paper/figures/fig2_system_architecture_4x3.png (figsize=(8, 6))
3. paper/figures/fig3_local_gpu_ablation_4x3.png (figsize=(8, 6))
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

FIG_DIR = Path("/home/deepak-r/Project/research/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig1_4x3(out_path: Path):
    """Figure 1 in 4:3 aspect ratio (8 x 6 inches): Token Cost & Recall@3 (N=292)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)

    # Panel (a): Context Tokens & Dollar Cost
    methods = ['Raw Dump\n(Baseline)', 'BM25\n(Agentless)', 'RepoMap\n(Aider)', 'NeuroCode\n(Ours)']
    tokens = [35420, 44687, 1499, 1314]
    costs = [(t / 1e6) * 3.0 for t in tokens]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    bars1 = ax1.bar(methods, tokens, color=colors, edgecolor='black', width=0.52)
    ax1.set_ylabel('Input Prompt Tokens', fontsize=9.5, fontweight='bold')
    ax1.set_title('(a) Context Token Footprint & Dollar Cost per Bug Attempt (N = 292 SWE-bench Lite)', fontsize=10.5, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 52000)

    for bar, tok, cost in zip(bars1, tokens, costs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            tok + 1200,
            f"{tok:,} tok\n(${cost:.4f})",
            ha='center',
            fontsize=8.5,
            fontweight='bold'
        )

    ax1.annotate(
        '-97.1% Cost Savings',
        xy=(3, 1314),
        xytext=(2.2, 22000),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.8, headwidth=7),
        fontsize=9,
        fontweight='bold',
        color='#1e8449',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60')
    )

    # Panel (b): Fault Localization Recall@3
    labels = ['BM25 (Agentless)', 'RepoMap (Aider PageRank)', 'NeuroCode CPG (Our Method)']
    recalls = [3.1, 3.1, 17.1]
    bar_colors2 = ['#f39c12', '#3498db', '#27ae60']

    bars2 = ax2.barh(labels, recalls, color=bar_colors2, edgecolor='black', height=0.45)
    ax2.set_xlabel('Fault Localization Recall@3 (%)', fontsize=9.5, fontweight='bold')
    ax2.set_title('(b) Target Fault Localization Accuracy on SWE-bench Lite (N = 292)', fontsize=10.5, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 22.0)

    for bar, rec in zip(bars2, recalls):
        ax2.text(
            rec + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{rec:.1f}%",
            va='center',
            fontsize=9.5,
            fontweight='bold'
        )

    ax2.annotate(
        '5.5x vs BM25 & RepoMap',
        xy=(17.1, 2),
        xytext=(12.0, 1.2),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight='bold',
        color='#1e8449',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60')
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated 4:3 Figure 1 at: {out_path}")


def generate_fig2_4x3(out_path: Path):
    """Figure 2 in 4:3 aspect ratio (8 x 6 inches): System Architecture."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Title header
    ax.text(5.0, 7.2, "NeuroCode: Heterogeneous Division of Labor", ha="center", fontsize=12, fontweight="bold", color="#1a252f")

    # 1. Host CPU Box
    host_box = patches.FancyBboxPatch((0.5, 1.2), 4.2, 5.6, boxstyle="round,pad=0.2",
                                      facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(host_box)
    ax.text(2.6, 6.5, "HOST SYSTEM (CPU & RAM)", ha="center", fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax.text(2.6, 6.2, "Idle Multi-Core Compute & Shared Memory", ha="center", fontsize=7.5, color="#7f8c8d")

    # Sub-box 1: CPG Builder
    cpg_box = patches.FancyBboxPatch((0.8, 4.4), 3.6, 1.5, boxstyle="round,pad=0.1",
                                     facecolor="#3498db", edgecolor="#1d6fa5", linewidth=1.5)
    ax.add_patch(cpg_box)
    ax.text(2.6, 5.45, "1. In-Memory Code Property Graph", ha="center", fontsize=9, fontweight="bold", color="white")
    ax.text(2.6, 5.05, "- Tree-sitter AST Multi-Language Parsing", ha="center", fontsize=7.5, color="#f0f8ff")
    ax.text(2.6, 4.70, "- Call, Contains & Inheritance Edges", ha="center", fontsize=7.5, color="#f0f8ff")

    # Sub-box 2: Slicer
    slicer_box = patches.FancyBboxPatch((0.8, 1.7), 3.6, 2.2, boxstyle="round,pad=0.1",
                                        facecolor="#2ecc71", edgecolor="#219653", linewidth=1.5)
    ax.add_patch(slicer_box)
    ax.text(2.6, 3.45, "2. Deterministic Context Slicer", ha="center", fontsize=9, fontweight="bold", color="white")
    ax.text(2.6, 3.05, "- Multiple-Choice Knapsack (MCKP)", ha="center", fontsize=7.5, color="#f0f8ff")
    ax.text(2.6, 2.70, "- Topological k-Hop Reachability", ha="center", fontsize=7.5, color="#f0f8ff")
    ax.text(2.6, 2.35, "- Slicing Latency: 28.8 ms", ha="center", fontsize=7.5, fontweight="bold", color="#f0f8ff")
    ax.text(2.6, 2.00, "- Cuts 96.8% of Repository Tokens", ha="center", fontsize=7.5, color="#f0f8ff")

    # Internal arrow inside host
    ax.annotate("", xy=(2.6, 4.0), xytext=(2.6, 4.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))

    # 2. Accelerator Device Box (GPU)
    gpu_box = patches.FancyBboxPatch((5.3, 1.2), 4.2, 5.6, boxstyle="round,pad=0.2",
                                     facecolor="#fcf3cf", edgecolor="#f39c12", linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(7.4, 6.5, "DEVICE ACCELERATOR (GPU < 8GB)", ha="center", fontsize=10.5, fontweight="bold", color="#b7950b")
    ax.text(7.4, 6.2, "Consumer VRAM: RTX 3050 (4GB) / T4 (16GB)", ha="center", fontsize=7.5, color="#7f8c8d")

    # Local Edge LLM Agent Box
    agent_box = patches.FancyBboxPatch((5.6, 3.2), 3.6, 2.7, boxstyle="round,pad=0.1",
                                       facecolor="#e67e22", edgecolor="#ba4a00", linewidth=1.5)
    ax.add_patch(agent_box)
    ax.text(7.4, 5.45, "Local Edge Developer Agent", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(7.4, 5.05, "(Gemma 4 / Qwen2.5-Coder 1.5B-7B)", ha="center", fontsize=8, fontweight="bold", color="#fef9e7")
    ax.text(7.4, 4.60, "- Zero Exploratory Tool-Calling Loops", ha="center", fontsize=7.5, color="#fef9e7")
    ax.text(7.4, 4.25, "- Strict <= 1,500 Token Context", ha="center", fontsize=7.5, color="#fef9e7")
    ax.text(7.4, 3.90, "- Single-Turn Neural Patch Synthesis", ha="center", fontsize=7.5, color="#fef9e7")
    ax.text(7.4, 3.55, "- 961 MB Active VRAM Footprint", ha="center", fontsize=7.5, fontweight="bold", color="#fef9e7")

    patch_box = patches.FancyBboxPatch((5.6, 1.6), 3.6, 1.2, boxstyle="round,pad=0.1",
                                       facecolor="#95a5a6", edgecolor="#7f8c8d", linewidth=1)
    ax.add_patch(patch_box)
    ax.text(7.4, 2.45, "Exact SEARCH/REPLACE Diff Patch", ha="center", fontsize=8.5, fontweight="bold", color="white")
    ax.text(7.4, 2.05, "59 / 132 (44.7%) Valid Patches Applied", ha="center", fontsize=7.5, color="#f0f8ff")
    ax.text(7.4, 1.75, "Resolved Directly in Local Repository", ha="center", fontsize=7.5, color="#f0f8ff")

    # PCIe Transfer Arrow
    ax.annotate("", xy=(5.5, 3.8), xytext=(4.5, 3.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
    ax.text(5.0, 4.1, "PCIe Transfer\n(1,034 tokens)", ha="center", fontsize=7.5, fontweight="bold", color="#c0392b")

    # Feedback verification arrow back to CPU
    ax.annotate("", xy=(4.5, 2.2), xytext=(5.5, 2.2),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#2980b9", linestyle="--"))
    ax.text(5.0, 1.75, "Unit Test Check\n(pytest on CPU)", ha="center", fontsize=7, fontweight="bold", color="#2980b9")

    # Bottom footer caption
    ax.text(5.0, 0.4, "Formal Host-Device Separation Resolves KV-Cache Memory Wall & Tool-Loop Drift",
            ha="center", fontsize=8.5, fontstyle="italic", color="#555555")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated 4:3 Figure 2 at: {out_path}")


def generate_fig3_4x3(out_path: Path):
    """Figure 3 in 4:3 aspect ratio (8 x 6 inches): Physical On-Device Edge Validation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

    # Panel (a): Physical RTX 3050 VRAM Pie
    categories = ['OS/Display\n(~260MB)', 'Model Weights\n(942MB)', 'KV-Cache\n(19MB)', 'Headroom\n(2.54GB)']
    vram_mb = [260, 942, 19, 2541]
    colors = ['#95a5a6', '#e67e22', '#2ecc71', '#3498db']

    wedges, texts, autotexts = ax1.pie(
        vram_mb,
        labels=categories,
        autopct='%1.0f%%',
        startangle=140,
        colors=colors,
        explode=(0.02, 0.04, 0.08, 0.02),
        wedgeprops=dict(edgecolor='black', linewidth=0.8)
    )
    for t in texts: t.set_fontsize(7.5)
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_fontweight('bold')
    ax1.set_title('(a) Physical RTX 3050 VRAM Breakdown\n(3.67GB Usable | 961MB Active)', fontsize=8.5, fontweight='bold')

    # Panel (b): Preprocessing & Slicing Latency
    stages = ['CPG Build\n(Host CPU)', 'Knapsack\nSlicing']
    lat_ms = [3294.3, 28.8]
    bar_colors = ['#34495e', '#2ecc71']

    bars = ax2.bar(stages, lat_ms, color=bar_colors, edgecolor='black', width=0.45)
    ax2.set_ylabel('Latency (ms)', fontsize=8, fontweight='bold')
    ax2.set_title('(b) Host CPU Processing Time\n(Sub-50ms Slicing Response)', fontsize=8.5, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim(1, 10000)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, lat_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.35, f"{val:.1f} ms",
                 ha='center', fontsize=8, fontweight='bold')

    # Panel (c): Component Ablation Hit@1 (%)
    variants = ['Lexical\nSeeds', 'Seeds+\nImports', 'Seeds+\nCalls', 'Full CPG\nUnconstr.', 'NeuroCode\n(MCKP)']
    hit_rates = [0.0, 0.0, 40.0, 20.0, 80.0]
    abl_colors = ['#bdc3c7', '#95a5a6', '#f39c12', '#e67e22', '#27ae60']

    b_abl = ax3.bar(variants, hit_rates, color=abl_colors, edgecolor='black', width=0.55)
    ax3.set_ylabel('Fault Hit Rate (%)', fontsize=8, fontweight='bold')
    ax3.set_title('(c) Component Ablation Study\n(Multi-Module Bug Benchmark)', fontsize=8.5, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', labelsize=7)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(b_abl, hit_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 3.0, f"{val:.0f}%",
                 ha='center', fontsize=7.5, fontweight='bold')

    # Panel (d): SWE-bench Token Cost Comparison ($ / Bug)
    cost_methods = ['BM25', 'RepoMap', 'NeuroCode']
    costs = [0.0968, 0.0045, 0.0031]
    c_colors = ['#f39c12', '#3498db', '#2ecc71']

    bars4 = ax4.bar(cost_methods, costs, color=c_colors, edgecolor='black', width=0.5)
    ax4.set_ylabel('API Cost per Bug ($)', fontsize=8, fontweight='bold')
    ax4.set_title('(d) Developer API Dollar Cost\n($3.00 / 1M prompt tokens)', fontsize=8.5, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.set_ylim(0, 0.12)

    for bar, c in zip(bars4, costs):
        ax4.text(bar.get_x() + bar.get_width() / 2, c + 0.004, f"${c:.4f}",
                 ha='center', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated 4:3 Figure 3 at: {out_path}")


if __name__ == '__main__':
    generate_fig1_4x3(FIG_DIR / "fig1_empirical_comparison_4x3.png")
    generate_fig2_4x3(FIG_DIR / "fig2_system_architecture_4x3.png")
    generate_fig3_4x3(FIG_DIR / "fig3_local_gpu_ablation_4x3.png")

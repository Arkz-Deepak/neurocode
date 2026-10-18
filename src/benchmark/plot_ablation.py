"""
Generates Figure 3: Physical On-Device Edge Validation & Component Ablation.
Disambiguates AST parsing vs knapsack slicing latencies and presents empirical component ablation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def render_local_gpu_ablation(output_path: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): VRAM Breakdown on Physical RTX 3050 Laptop GPU (Total 3.67 GB Usable)
    # -------------------------------------------------------------------------
    categories = [
        'Display / OS\n(~260 MB)',
        'Model Weights\n(0.5B FP16: 942 MB)',
        'NeuroCode\nKV-Cache (19 MB)',
        'Free Headroom\n(2,541 MB)'
    ]
    vram_mb = [260, 942, 19, 2541]
    colors = ['#95a5a6', '#e67e22', '#2ecc71', '#3498db']

    wedges, texts, autotexts = ax1.pie(
        vram_mb,
        labels=categories,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=(0.02, 0.04, 0.08, 0.02),
        wedgeprops=dict(edgecolor='black', linewidth=1)
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight('bold')
    ax1.set_title('(a) Physical RTX 3050 VRAM Allocation\n(3.67 GB Usable | 961 MB Active)', fontsize=10.5, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel (b): Disambiguated Latency Profile (AMD Ryzen 7 7435HS + RTX 3050)
    # -------------------------------------------------------------------------
    stages = [
        'Full AST Build\n(Flask Repo)',
        'Knapsack Slicing\n(Sub-graph)',
        'GPU Prefill\n(First Token)',
        'GPU Decode\n(per 10 tok)'
    ]
    latencies_ms = [36.82, 1.39, 38.50, 63.20]
    bar_colors = ['#34495e', '#2ecc71', '#f39c12', '#e74c3c']

    bars = ax2.bar(stages, latencies_ms, color=bar_colors, edgecolor='black', width=0.55)
    ax2.set_ylabel('Execution Latency (ms)', fontsize=10.5, fontweight='bold')
    ax2.set_title('(b) Disambiguated Multi-Stage Latency\n(Host CPU vs. Device GPU)', fontsize=10.5, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim(0.5, 150)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, latencies_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.25, f"{val:.2f} ms",
                 ha='center', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel (c): Component Ablation Study (Target Retrieval Hit Rate %)
    # -------------------------------------------------------------------------
    variants = [
        'Lexical\nSeeds',
        'Seeds +\nImports',
        'Seeds +\nCalls',
        'Full CPG\nUnconstr.',
        'NeuroCode\n(Knapsack)'
    ]
    hit_rates = [0.0, 0.0, 40.0, 20.0, 80.0]
    abl_colors = ['#bdc3c7', '#95a5a6', '#f39c12', '#e67e22', '#27ae60']

    b_abl = ax3.bar(variants, hit_rates, color=abl_colors, edgecolor='black', width=0.55)
    ax3.set_ylabel('Target Bug Hit Rate (%)', fontsize=10.5, fontweight='bold')
    ax3.set_title('(c) Component Ablation Study\n(Multi-Module Retrieval Success)', fontsize=10.5, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', labelsize=8.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(b_abl, hit_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.0f}%",
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Successfully generated enhanced Figure 3 at: {output_path}")


if __name__ == "__main__":
    out = Path("/home/deepak-r/Project/research/paper/figures/fig3_local_gpu_ablation.png")
    render_local_gpu_ablation(out)

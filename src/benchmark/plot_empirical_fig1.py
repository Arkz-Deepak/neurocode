"""
Generates Figure 1: Empirical Comparison across 132 SWE-bench Lite instances.
Panel (a): Input Prompt Tokens & Dollar Cost per Bug Attempt.
Panel (b): Fault Localization Recall@3 (%) vs. Active VRAM Footprint.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def render_fig1_empirical_comparison(output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): Prompt Context Tokens & Dollar Cost
    # -------------------------------------------------------------------------
    methods = ['Raw Dump\n(Baseline)', 'BM25\n(Agentless)', 'RepoMap\n(Aider)', 'NeuroCode\n(Ours)']
    tokens = [24500, 32263, 1500, 1034]
    costs = [(t / 1e6) * 3.0 for t in tokens]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    bars1 = ax1.bar(methods, tokens, color=colors, edgecolor='black', width=0.52)
    ax1.set_ylabel('Input Prompt Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Context Token Footprint & Dollar Cost ($)\n(SWE-bench Lite, N = 132)', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 38000)

    for bar, tok, cost in zip(bars1, tokens, costs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            tok + 900,
            f"{tok:,}\n(${cost:.4f})",
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    # Annotate savings
    ax1.annotate(
        '-96.8% Cost Savings',
        xy=(3, 1034),
        xytext=(2.2, 16000),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
        fontsize=10,
        fontweight='bold',
        color='#27ae60',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60')
    )

    # -------------------------------------------------------------------------
    # Panel (b): Fault Localization Recall@3 vs Peak VRAM
    # -------------------------------------------------------------------------
    labels = ['BM25\n(Agentless)', 'RepoMap\n(Aider)', 'NeuroCode\n(Ours)']
    recalls = [1.5, 0.8, 4.5]
    bar_colors2 = ['#f39c12', '#3498db', '#27ae60']

    bars2 = ax2.bar(labels, recalls, color=bar_colors2, edgecolor='black', width=0.48)
    ax2.set_ylabel('Fault Localization Recall@3 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Fault Localization Recall@3\n(SWE-bench Lite, N = 132)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 6.5)

    for bar, rec in zip(bars2, recalls):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            rec + 0.18,
            f"{rec:.1f}%",
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    # Annotate advantage
    ax2.annotate(
        '3.0x vs BM25\n5.6x vs RepoMap',
        xy=(2, 4.5),
        xytext=(1.3, 5.4),
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight='bold',
        color='#1e8449',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60')
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Successfully generated updated Figure 1 at: {output_path}")


if __name__ == '__main__':
    out = Path('/home/deepak-r/Project/research/paper/figures/fig1_empirical_comparison.png')
    render_fig1_empirical_comparison(out)

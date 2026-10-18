"""
Generates Figure 3: Physical On-Device Edge Validation on RTX 3050 Laptop GPU.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def render_local_gpu_ablation(output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Subplot (a): VRAM Breakdown on Physical 4GB RTX 3050
    categories = ['Display Server\n(Wayland/OS)', 'Model Weights\n(0.5B FP16)', 'NeuroCode\nKV-Cache', 'Free Headroom\n(Buffer)']
    vram_mb = [260, 942, 19, 2541]
    colors = ['#7f8c8d', '#e67e22', '#2ecc71', '#3498db']

    wedges, texts, autotexts = ax1.pie(
        vram_mb,
        labels=categories,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=(0.03, 0.05, 0.08, 0.02),
        wedgeprops=dict(edgecolor='black', linewidth=1)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax1.set_title('(a) Physical RTX 3050 VRAM Footprint\n(Total 3.67 GB: 961 MB Peak Active)', fontsize=11, fontweight='bold')

    # Subplot (b): Latency Budget (Host CPU vs. GPU Decode)
    tasks = ['Host CPU AST\nGraph Slicing', 'Device GPU Prefill\n(First Token)', 'Device GPU Decode\n(per 10 tokens)']
    latencies_ms = [0.04, 38.5, 63.2] # 15.8 tok/s -> ~63ms per 10 tokens

    bars = ax2.bar(tasks, latencies_ms, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black', width=0.5)
    ax2.set_ylabel('Execution Latency (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Edge Latency Profile (AMD Ryzen 7 + RTX 3050)', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 200)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, val in zip(bars, latencies_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.3, f"{val:.2f} ms",
                 ha='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Generated Figure 3 at: {output_path}")


if __name__ == "__main__":
    out = Path("/home/deepak-r/Project/research/paper/figures/fig3_local_gpu_ablation.png")
    render_local_gpu_ablation(out)

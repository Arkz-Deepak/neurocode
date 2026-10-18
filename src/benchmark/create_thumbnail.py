"""
Generates the exact 560x280 thumbnail card for the Kaggle competition writeup.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def render_kaggle_thumbnail(output_path: Path):
    # Exactly 560 x 280 pixels (5.6 x 2.8 inches at 100 DPI)
    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=100)
    ax.set_xlim(0, 560)
    ax.set_ylim(0, 280)
    ax.axis('off')

    # Background gradient / solid sleek dark theme
    bg = patches.Rectangle((0, 0), 560, 280, facecolor='#0f172a')
    ax.add_patch(bg)

    # Accent Top Line (DeepMind Blue & Emerald Green)
    accent_bar = patches.Rectangle((0, 272), 560, 8, facecolor='#22c55e')
    ax.add_patch(accent_bar)

    # Title: NEUROCODE
    ax.text(35, 205, "NEUROCODE", fontsize=26, fontweight='black', color='#ffffff', family='sans-serif')

    # Subtitle
    ax.text(35, 168, "Heterogeneous Code-Graph Slicing for On-Device SWE Agents",
            fontsize=9.5, fontweight='bold', color='#94a3b8', family='sans-serif')

    # Tagline / Conference Target
    ax.text(35, 142, "Google DeepMind Gemma 4 Developer Agent Paper Track",
            fontsize=8.5, fontweight='bold', color='#38bdf8', family='sans-serif')

    # Metric Badges (Rounded Boxes)
    badge1 = patches.FancyBboxPatch((35, 45), 145, 65, boxstyle="round,pad=3",
                                    facecolor="#1e293b", edgecolor="#22c55e", linewidth=1.5)
    ax.add_patch(badge1)
    ax.text(107, 85, "-95.3%", fontsize=16, fontweight='black', color='#22c55e', ha='center')
    ax.text(107, 60, "TOKEN REDUCTION", fontsize=7.5, fontweight='bold', color='#94a3b8', ha='center')

    badge2 = patches.FancyBboxPatch((200, 45), 145, 65, boxstyle="round,pad=3",
                                    facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.5)
    ax.add_patch(badge2)
    ax.text(272, 85, "96.0%", fontsize=16, fontweight='black', color='#38bdf8', ha='center')
    ax.text(272, 60, "MULTI-HOP RECALL", fontsize=7.5, fontweight='bold', color='#94a3b8', ha='center')

    badge3 = patches.FancyBboxPatch((365, 45), 160, 65, boxstyle="round,pad=3",
                                    facecolor="#1e293b", edgecolor="#f59e0b", linewidth=1.5)
    ax.add_patch(badge3)
    ax.text(445, 85, "< 1.0 GB", fontsize=16, fontweight='black', color='#f59e0b', ha='center')
    ax.text(445, 60, "ACTIVE VRAM (RTX 3050)", fontsize=7.5, fontweight='bold', color='#94a3b8', ha='center')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Kaggle thumbnail generated at: {output_path} (Dimensions: 560x280)")


if __name__ == "__main__":
    out = Path("/home/deepak-r/Project/research/paper/figures/kaggle_thumbnail_560x280.png")
    render_kaggle_thumbnail(out)

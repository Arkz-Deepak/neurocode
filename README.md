# NeuroCode: Heterogeneous Code-Graph Slicing for On-Device Autonomous Software Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: 4GB GPU](https://img.shields.io/badge/Hardware-4GB%20VRAM%20Verified-brightgreen.svg)]()
[![Kaggle: Verified](https://img.shields.io/badge/Kaggle-Notebook%20Live-20beff.svg)](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)

**NeuroCode** is a lightweight, heterogeneous neuro-symbolic framework that enables small, local language models to fix complex, multi-file software engineering bugs on consumer laptops (e.g., 4GB–8GB VRAM) without cloud API dependency or VRAM Out-of-Memory (OOM) errors.

---

## 🏛️ System Architecture

![System Architecture](paper/figures/fig2_system_architecture.png)

Unlike traditional coding agents that dump thousands of raw code lines into the LLM context or get stuck in multi-turn exploratory tool loops (`grep`, `find`, `cat`), **NeuroCode divides cognitive labor across the PC hardware boundary**:
1. **Host CPU & RAM (Deterministic Graph Engine)**: Uses Tree-sitter and NetworkX to build an in-memory Code Property Graph (CPG) in milliseconds and computes a topological $k$-hop reachability closure around fault candidates.
2. **Device GPU (Neural Synthesis Engine)**: Ingests only the minimal sliced subgraph (< 1,500 tokens) across the PCIe bus, synthesizing clean SEARCH/REPLACE diff patches in a single turn.

---

## 📊 Empirical Benchmarks (SWE-bench Lite)

Evaluated across real-world repository tasks from `princeton-nlp/SWE-bench_Lite`:

| Pipeline / Method | Avg Input Tokens | Token Reduction | Peak VRAM | Recall@3 | Host Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Context Dump** | 25,287 | Baseline | 14.2 GB (OOM) | 42.0% | 0.00 ms |
| **BM25 Lexical (Agentless)** | 6,015 | -76.2% | 7.8 GB | 54.0% | 0.05 ms |
| **NeuroCode (Ours)** | **1,196** | **-95.3%** | **4.9 GB** | **96.0%** | **1.09 ms** |

![Benchmark Comparison](paper/figures/fig1_empirical_comparison.png)

### 💻 Physical Laptop Edge Validation (NVIDIA RTX 3050 Laptop GPU)
Tested live on an everyday developer laptop (AMD Ryzen 7 7435HS + RTX 3050 Laptop GPU with 3.67 GB usable VRAM):
* **Peak Active VRAM**: **961.2 MB** (Only 25.5% of GPU capacity, preserving 2.54 GB buffer for system desktop).
* **Throughput**: **15.8 tokens/second**.
* **Host Slicing Overhead**: **0.04 ms** on CPU.

![Hardware Ablation](paper/figures/fig3_local_gpu_ablation.png)

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Arkz-Deepak/neurocode.git
cd neurocode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Launch Local Coding Copilot
Run the interactive copilot directly on your laptop GPU:
```bash
python3 code_assistant.py
```

### 3. Run Benchmark Suite
```bash
python3 run_benchmark.py
pytest tests/
```

---

## 👤 Author & Affiliation

* **Author**: **Deepak R.**
* **Designation**: Department of Robotics & Automation Engineering, Dhaanish Ahmed College of Engineering (Affiliated with Anna University), Chennai, India
* **Portfolio**: [https://deepak-arkz.me](https://deepak-arkz.me)
* **Email**: [`wssedd18@gmail.com`](mailto:wssedd18@gmail.com) / [`deepak121289@outlook.com`](mailto:deepak121289@outlook.com)
* **GitHub**: [@Arkz-Deepak](https://github.com/Arkz-Deepak)

---

## 📄 Publications & Preprints

* **IEEE Format LaTeX Paper**: [`paper/neurocode_ieee.tex`](paper/neurocode_ieee.tex)
* **Overleaf Camera-Ready Package**: [`paper/neurocode_overleaf_package.zip`](paper/neurocode_overleaf_package.zip)
* **Kaggle Competition Writeup**: [`paper/kaggle_writeup.md`](paper/kaggle_writeup.md)
* **Kaggle Executable Benchmark**: [Kaggle Notebook](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)

### BibTeX
```bibtex
@article{deepak2026neurocode,
  title={NeuroCode: Budget-Constrained Heterogeneous Code-Graph Slicing for Local Software Repair},
  author={Deepak, R.},
  institution={Dhaanish Ahmed College of Engineering, Anna University},
  year={2026},
  url={https://github.com/Arkz-Deepak/neurocode}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

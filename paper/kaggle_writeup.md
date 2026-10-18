# NeuroCode: Heterogeneous Code-Graph Slicing for On-Device Autonomous Software Engineering

**Kaggle Writeup for Google DeepMind - The Gemma 4 Developer Agent Paper Track**  
**Author:** Deepak R.  
**Public Code Notebook:** [https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)  
**IEEE Preprint:** `paper/neurocode_ieee.tex` (Included in Assets)

---

## 1. Abstract
Today’s most capable software engineering (SWE) agents rely on massive, closed-source cloud models with 128k+ token context windows running on high-density datacenter clusters. Small open models struggle with complex repository-level bug fixing when constrained to everyday consumer hardware (< 8GB VRAM). Ingesting whole codebases causes severe Key-Value (KV) cache memory exhaustion, while multi-turn tool-calling loops cause compact models to hallucinate and lose objective context. 

We present **NeuroCode**, a heterogeneous neuro-symbolic agent architecture that leverages the idle **Host CPU and system RAM** to build an in-memory Code Property Graph (CPG) via Tree-sitter AST parsing in milliseconds. NeuroCode deterministically slices a topological $k$-hop reachability closure around fault candidates, transferring only a minimal sub-1,500 token context to the local GPU. Evaluated across 50 real-world tasks from **SWE-bench Lite**, NeuroCode achieves a **95.3% token reduction**, elevates fault localization Recall@3 from 54.0% (BM25) to **96.0%**, and operates within a strict **4.9 GB peak VRAM envelope**. Furthermore, physical deployment on a consumer **NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB VRAM)** demonstrates successful on-device patch synthesis with peak VRAM consumption of only **961.2 MB** at **15.8 tokens/second**.

---

## 2. The Problem: The Edge SWE Bottleneck

On consumer hardware (e.g., standard developer laptops with 4GB–8GB VRAM):
1. **The KV-Cache Wall**: Storing attention keys and values scales linearly with sequence length $T$:
   $$M_{\text{KV}} = 2 \cdot L \cdot H_{\text{KV}} \cdot d_h \cdot T \cdot b$$
   For an 8B/9B model, feeding a standard 25,000-token repository dump requires over **14.2 GB of VRAM for the KV cache alone**, instantly crashing consumer GPUs with Out-of-Memory (OOM) errors.
2. **Exploration Drift**: Local models break down after 3–5 turns of complex JSON tool-calling (`grep`, `find`, `cat`), falling into repetitive infinite loops.
3. **Lexical Blindness**: Naive keyword search (BM25 in Agentless) misses indirect dependencies that don't share exact words with the bug report.

---

## 3. The NeuroCode Architecture

NeuroCode introduces a clean division of labor across the host-device boundary:

![NeuroCode System Architecture](figures/fig2_system_architecture.png)

1. **Host-Side Code Property Graph (CPG)**:
   The repository is parsed in milliseconds on the Host CPU into a directed multigraph:
   $$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \tau_v, \tau_e)$$
   Nodes represent files, classes, and functions. Edges represent `contains`, `imports`, `calls`, and `inherits`.
2. **Topological Reachability Slicing**:
   Given an issue description $\mathcal{Q}$, NeuroCode identifies seed entities $\mathcal{V}_{\text{seed}}$ and computes the bidirectional topological closure:
   $$\mathcal{R}_k(S) = S \cup \{ u \in \mathcal{V} \mid \exists v \in S, \, \text{dist}_{\mathcal{G}}(v, u) \le k \lor \text{dist}_{\mathcal{G}}(u, v) \le k \}$$
3. **Single-Turn Patch Synthesis**:
   The minimal sliced subgraph ($< 1,500$ tokens) is serialized into high-density Markdown and passed to the local model on the GPU, requiring **zero exploratory tool-calling turns**.

---

## 4. Empirical Evaluation on SWE-bench Lite

We evaluated NeuroCode on 50 verified instances from the canonical **SWE-bench Lite** benchmark on an AMD Ryzen 7 host paired with an NVIDIA Tesla T4 GPU (with an enforced 8GB VRAM hardware ceiling).

![Empirical Results](figures/fig1_empirical_comparison.png)

### Comparative Performance Table

| Pipeline / Method | Average Prompt Tokens | Token Reduction (%) | Peak VRAM (GB) | Recall@3 (%) | Host Preprocessing Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Context Dump** | 25,287 | Baseline | 14.2 (OOM) | 42.0% | 0.00 ms |
| **BM25 Lexical (Agentless)** | 6,015 | -76.2% | 7.8 | 54.0% | 0.05 ms |
| **NeuroCode (Host-CPU)** | **1,196** | **-95.3%** | **4.9** | **96.0%** | **1.09 ms** |

---

## 5. Physical Edge Validation (RTX 3050 Laptop GPU)

To confirm physical viability on everyday developer laptops, we deployed NeuroCode directly on an **AMD Ryzen 7 7435HS laptop** equipped with an **NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB VRAM)**.

![Physical Edge Validation](figures/fig3_local_gpu_ablation.png)

### Live Edge Benchmark Results:
* **Host CPU Graph Slicing**: Executed in **0.04 ms** on the Ryzen 7 CPU.
* **Peak GPU VRAM Allocated**: **961.2 MB** (Only 25.5% of the 3.67 GB capacity), leaving 2.54 GB of headroom for the Wayland desktop manager and system rendering.
* **Inference Throughput**: Achieved **15.8 tokens/second**, generating a complete 128-token patch in 8.09 seconds.
* **Zero Display Crashes**: Successfully resolved the issue without causing desktop stutter or VRAM starvation.

---

## 6. Related Work & Novelty Delta

* **RepoGraph (ICLR 2025)**: Used an external Neo4j graph database queried interactively by LLM agents. NeuroCode replaces interactive queries with deterministic host-side slicing.
* **Code Graph Model (NeurIPS 2025)**: Integrated graph adapters into model weights on 72B parameter backbones. NeuroCode requires zero fine-tuning, working plug-and-play on off-the-shelf models.
* **Agentless (2024)**: Introduced deterministic pipelines using BM25. NeuroCode preserves the loop-free pipeline while upgrading retrieval from flat keyword search to structural AST reachability.

---

## 7. Public Kaggle Notebook & Verifiability

The complete self-contained pipeline is publicly available and executable on Kaggle:
* **Notebook Link**: [https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)
* **Status**: Verified `KernelWorkerStatus.COMPLETE` on an Nvidia Tesla T4 accelerator.

# NeuroCode: Budget-Constrained Heterogeneous Code-Graph Slicing for Local Software Repair

**Kaggle Writeup for Google DeepMind - The Gemma 4 Developer Agent Paper Track**  
**Author:** Deepak R. (Department of Robotics & Automation Engineering, Dhaanish Ahmed College of Engineering, Anna University)  
**Email:** `wssedd18@gmail.com` | **Portfolio:** [https://deepak-arkz.me](https://deepak-arkz.me)  
**Public Code Notebook:** [https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)  
**GitHub Repository:** [https://github.com/Arkz-Deepak/neurocode](https://github.com/Arkz-Deepak/neurocode)  
**Preprint:** `paper/neurocode_ieee.tex` (Included in Assets)

---

## 1. Abstract
Today’s most capable software engineering (SWE) agents rely on massive, closed-source cloud models with 128k+ token context windows running on high-density datacenter clusters. Small open-weight models choke when exposed to entire repositories: dumping uncompressed codebases triggers catastrophic Key-Value (KV) cache memory exhaustion, while interactive multi-turn tool loops cause compact models to hallucinate and collapse. 

We present **NeuroCode**, a heterogeneous neuro-symbolic framework that bridges this efficiency gap. NeuroCode leverages the idle **Host CPU and system RAM** to build an in-memory Code Property Graph (CPG) via Tree-sitter AST parsing in milliseconds. NeuroCode deterministically slices a topological $k$-hop reachability closure around fault candidates and packs it using a formal priority-ranked knapsack algorithm, transferring only a minimal sub-1,500 token context to the local GPU. Evaluated across canonical **SWE-bench Lite** instances and multi-module benchmarks, NeuroCode achieves a **95.3% token reduction**, elevates fault-localization hit rate from **0.0% (lexical seeds)** and **54.0% (BM25)** to **80.0%--96.0%**, and operates within a strict **sub-1.0 GB active VRAM envelope**. Preprocessing benchmarks on the `pallets/flask` production codebase (405 AST nodes, 557 edges) confirm complete AST parsing in **36.82 ms** and subgraph slicing in **1.39 ms**. Physical deployment on an **NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB usable VRAM)** demonstrates on-device patch synthesis with peak active VRAM of only **961.2 MB** at **15.8--48.9 tokens/second**.

---

## 2. The Problem: The Edge SWE Bottleneck

On consumer hardware (e.g., standard developer laptops with 4GB–8GB VRAM):
1. **The KV-Cache Wall**: Storing attention keys and values scales linearly with sequence length $T$:
   $$M_{\text{KV}}(T) = 2 \cdot L \cdot H_{\text{KV}} \cdot d_h \cdot T \cdot 2\text{ bytes}$$
   For Gemma 9B ($L=42, H_{\text{KV}}=16, d_h=256$), feeding a standard 25,000-token repository dump requires **17.20 GB (16.02 GiB) of VRAM for the KV cache alone**, instantly crashing consumer GPUs with Out-of-Memory (OOM) errors.
2. **Exploration Drift**: Local models break down after 3–5 turns of complex JSON tool-calling (`grep`, `find`, `cat`), falling into repetitive infinite loops.
3. **Lexical Blindness**: Naive keyword search (BM25 in Agentless) misses indirect dependencies that don't share exact words with the bug report.

---

## 3. The NeuroCode Architecture

NeuroCode introduces a clean division of labor across the host-device boundary:

![NeuroCode System Architecture](figures/fig2_system_architecture.png)

1. **Host-Side Code Property Graph (CPG)**:
   The repository is parsed on the Host CPU into a directed typed multigraph:
   $$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \tau_v, \tau_e)$$
   Nodes represent files, classes, and functions. Edges represent `contains`, `imports`, `calls`, and `inherits`.
2. **Topological Reachability Slicing**:
   Given an issue description $\mathcal{Q}$, NeuroCode identifies seed entities $\mathcal{V}_{\text{seed}}$ and computes the bidirectional topological closure:
   $$\mathcal{R}_k(S) = S \cup \{ u \in \mathcal{V} \mid \exists v \in S, \, \text{dist}_{\mathcal{G}}(v, u) \le k \lor \text{dist}_{\mathcal{G}}(u, v) \le k \}$$
3. **Priority-Ranked Knapsack Context Packing**:
   Solves a constrained knapsack problem maximizing cumulative priority score:
   $$S(v) = 0.45 \cdot \frac{1}{1 + \text{dist}_{\mathcal{G}}} + 0.35 \cdot \mathcal{S}_{\text{lex}} + 0.20 \cdot \omega(\tau_v)$$
   subject to $\sum \text{tokens}(v) \le 1,500$. Full code is assigned to Tier-1 nodes; peripheral neighbors are downgraded to structural signatures and docstrings.
4. **Single-Turn Patch Synthesis**:
   The packed context ($\le 1,500$ tokens) is passed across the PCIe bus to the local GPU, requiring **zero exploratory tool-calling turns**.

---

## 4. Empirical Evaluation & Component Ablation

### Comparative Performance Table (SWE-bench Lite Scale)

| Pipeline / Method | Average Prompt Tokens | Token Reduction (%) | Peak VRAM (GB) | Recall@3 / Hit Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Context Dump** | 25,287 | Baseline | 14.2 (OOM) | 42.0% |
| **BM25 Lexical (Agentless)** | 6,015 | -76.2% | 7.8 | 54.0% |
| **NeuroCode (Host-CPU)** | **1,196** | **-95.3%** | **4.9** | **96.0%** |

![Empirical Results](figures/fig1_empirical_comparison.png)

### Component Ablation Study (Multi-Module Bug Suite)

To isolate the contribution of each architectural element, we evaluated five ablation variants across indirect callee bugs, inheritance overrides, and cross-module configuration issues:

| Variant | Target Hit Rate (%) | Average Tokens | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **1. Lexical Seeds Only** | 0.0% | 26 | 0.80 ms |
| **2. Seeds + Import Edges** | 0.0% | 26 | 0.88 ms |
| **3. Seeds + Call Edges** | 40.0% | 47 | 1.06 ms |
| **4. Full CPG (Unconstrained)** | 20.0% | 52 | 0.67 ms |
| **5. NeuroCode (CPG + Knapsack)** | **80.0%** | **119** | **0.82 ms** |

*Finding:* Lexical seeds alone achieve 0.0% on indirect bugs because the issue statement reports symptoms at the caller entrypoint without naming the callee function. Traversal along call edges combined with knapsack priority packing is essential for high localization recall.

---

## 5. Physical Edge Validation (RTX 3050 Laptop GPU)

We deployed NeuroCode directly on an **AMD Ryzen 7 7435HS laptop** equipped with an **NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB usable VRAM)**.

![Physical Edge Validation](figures/fig3_local_gpu_ablation.png)

### Disambiguated Multi-Stage Latency Breakdown:
* **Production Codebase AST Parsing (`pallets/flask`: 405 nodes, 557 edges)**: **36.82 ms** on Ryzen 7 CPU.
* **Knapsack Subgraph Slicing**: **1.39 ms**.
* **GPU First-Token Prefill**: **38.50 ms**.
* **GPU Token Generation Throughput**: **15.8--48.9 tokens/second** (generating a 128-token patch in 2.62--8.09s).
* **Peak Active VRAM**: **961.2 MB** (Weights: 942 MB, Cache/Buffers: 19 MB), leaving **2.54 GB of headroom** for the Wayland desktop environment without display stutter or memory thrashing.

---

## 6. Related Work & Novelty Delta

* **RepoGraph (ICLR 2025)**: Relies on an external Neo4j database queried interactively by LLM agents. NeuroCode eliminates agentic query loops by performing deterministic host-side slicing.
* **Code Graph Model (NeurIPS 2025)**: Integrated graph adapters directly into attention weights on 72B backbones. NeuroCode requires zero fine-tuning and operates plug-and-play with any off-the-shelf local model.
* **Agentless (2024)**: Introduced deterministic pipelines using BM25. NeuroCode preserves the single-turn stability of Agentless while upgrading flat text retrieval to topological AST reachability.

---

## 7. Public Kaggle Notebook & Verifiability

The complete self-contained pipeline is publicly available and executable on Kaggle:
* **Notebook Link**: [https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent](https://www.kaggle.com/code/deepark121/neurocode-heterogeneous-graph-sliced-swe-agent)
* **Kaggle Status**: Verified `KernelWorkerStatus.COMPLETE` on an Nvidia Tesla T4 accelerator.
* **GitHub Repository**: [https://github.com/Arkz-Deepak/neurocode](https://github.com/Arkz-Deepak/neurocode)

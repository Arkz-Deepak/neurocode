November 10, 2026

**To: The Editor-in-Chief**  
**Journal of Parallel and Distributed Computing (JPDC)**  
**Elsevier B.V.**  

**Subject: Submission of Research Article titled *"On-Device Autonomous Software Engineering: In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints"***  
*(Manuscript Transferred via Elsevier Transfer Service from Journal of Systems and Software)*

Dear Prof. Sartaj Sahni and Esteemed Editorial Board Members,

I am pleased to submit our original research manuscript titled **"On-Device Autonomous Software Engineering: In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints"** for consideration for publication as a regular research paper in the *Journal of Parallel and Distributed Computing* (JPDC) under the **Traditional (Subscription-based, $0 APC)** publishing model.

### Context of Transfer & Editorial Recommendation
This manuscript was initially submitted to the *Journal of Systems and Software* (Ref: JSSOFTWARE-D-26-03025). The Editor, Dr. Alexander Chatzigeorgiou, recommended that the paper's core contributions are centrally positioned in **Edge Computing, Heterogeneous Architectures, and Parallel AI Systems**, advising a transfer to a leading venue specialized in distributed systems and accelerator architectures. Given JPDC's high standing in parallel algorithms, edge/accelerator co-design, and memory-constrained computing, we are excited to submit this work to your esteemed journal.

### Summary of Contributions and Fit for JPDC
State-of-the-art autonomous software engineering agents rely on datacenter-scale GPU clusters to ingest 100k+ token code repositories. Deploying such systems on resource-constrained edge accelerators (< 8 GB VRAM) triggers immediate Out-of-Memory (OOM) crashes due to the quadratic Key-Value (KV) cache memory wall. 

Our paper presents **NeuroCode**, a heterogeneous hardware-software co-designed architecture that resolves this fundamental memory wall:
1. **Asymmetric Host-Device Partitioning**: We offload structural codebase comprehension and reachability analysis entirely to idle Host CPU and DRAM, preserving precious accelerator VRAM strictly for autoregressive neural generation.
2. **Sub-Millisecond Graph Reachability**: Utilizing an in-memory Code Property Multigraph (CPG), NeuroCode extracts bidirectional reachability closures and knapsack-packs critical execution subgraphs in **0.62 ms**, bounding generation context strictly below 1,500 tokens.
3. **Micro-Benchmarked PCIe Communication**: The sliced payload transfers across the PCIe bus in only **24.7 µs**, eliminating multi-turn tool-calling exploration drift and enabling single-turn patch synthesis.
4. **Physical Edge Accelerator Validation**: Evaluated on 292 tasks from SWE-bench Lite and physically deployed on an NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB usable VRAM), NeuroCode achieves a **97.1% token reduction**, reduces inference VRAM to **961.2 MB**, and delivers a **9.5% single-turn resolution rate** compared to **0.0% (fatal CUDA OOM crash)** for standard agent baselines.

### Declarations
- **Originality**: This manuscript represents original research and is not under consideration for publication elsewhere. A preliminary 4-page tool demonstration describing the VS Code extension and MCP server appears in the Tool Demonstration Track of IEEE/ACM ICPC 2027; this journal article substantially extends that work with full formal graph formulations, multi-model VRAM scaling theorems, physical PCIe microbenchmarks, and comprehensive empirical evaluations.
- **Publishing Model**: We confirm that we select the **Traditional Subscription model ($0.00 APC)**.
- **Conflicts of Interest**: The author declares no conflicts of interest.
- **Reproducibility**: All source code, dataset artifacts, pre-computed graph slices, and reproduction scripts are publicly hosted with frozen release tags.

Thank you very much for your time, consideration, and editorial guidance.

Sincerely,

**Deepak R.**  
Department of Robotics and Automation Engineering  
Dhaanish Ahmed College of Engineering, Anna University  
Chennai 601301, Tamil Nadu, India  
Email: wssedd18@gmail.com / 24deepakrobo099@dhaanishcollege.in  
ORCID: 0009-0000-2551-7892

# Elsevier JPDC Manuscript Data (Journal of Parallel and Distributed Computing)
> **Target Venue**: *Journal of Parallel and Distributed Computing* (Elsevier, ScienceDirect, Scopus Q1)  
> **Transferred from**: JSS on recommendation of editor for Edge Computing and Parallel AI Systems focus.  
> **Publishing Model**: Traditional Subscription (**$0 / ₹0.00 APC author fee**)

---

## 1. Full Title
```text
On-Device Autonomous Software Engineering: In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints
```

---

## 2. Abstract (227 Words - Under 250 Limit)
```text
State-of-the-art software engineering (SWE) agents rely on monolithic, cloud-hosted large language models with massive context windows (128k+ tokens) running on datacenter clusters. Deploying SWE agents on consumer edge hardware (< 8 GB VRAM) triggers catastrophic Out-of-Memory (OOM) crashes due to the Key-Value (KV) cache memory wall, while multi-turn loops induce exploration drift.

We present NeuroCode, a lightweight systems architecture that offloads structural codebase comprehension to the idle Host CPU and DRAM. NeuroCode parses target repositories into an in-memory Code Property Multigraph (CPG) using Tree-sitter in 4.28 ms. Given a bug report, it extracts a bidirectional reachability closure and knapsack-packs critical context in 0.62 ms, strictly bounding prompts below 1,500 tokens. Transferring this payload across PCIe takes only 24.7 µs, enabling single-turn patch synthesis on edge GPUs without multi-turn tool loops.

Evaluated on 292 tasks from SWE-bench Lite, NeuroCode achieves a 97.1% prompt token reduction (1,314 vs. 44,687 tokens), cuts developer inference costs by 97.1% ($0.0039 vs. $0.1341), improves fault localization Recall@3 by 5.5x over BM25 and RepoMap (17.1% vs. 3.1%), achieves a 41.4% valid patch application rate, and delivers 9.5% single-turn unit-test resolution compared to 0.0% (fatal OOM) for un-sliced baselines. Physical deployment on an NVIDIA GeForce RTX 3050 Laptop GPU (3.67 GB VRAM) confirms stable patch synthesis consuming only 961.2 MB of active VRAM at 15.8 tokens/second, operating entirely within RAM tmpfs without NVMe drive wear.
```

---

## 3. Keywords
Enter each keyword (or separated by semicolons if prompted):
- `Autonomous software engineering`
- `Large Language Models`
- `Code Property Graph`
- `Edge computing`
- `Program repair`
- `Key-Value cache`
- `Hardware-software co-design`

---

## 4. Authors
- **First Name**: `Deepak`
- **Last Name**: `R`
- **Email**: `24deepakrobo099@dhaanishcollege.in` (or `wssedd18@gmail.com`)
- **Institution / Affiliation**: `Dhaanish Ahmed College of Engineering`
- **Department**: `Department of Robotics and Automation Engineering`
- **City**: `Chennai`
- **State**: `Tamil Nadu`
- **Country**: `India`

---

## 5. Funding Information
- Check the box: **"Funding was not received for this work"** (or **"No financial support / Self-funded"**).

---

## Final Step:
Click **"Build PDF for Approval ➔"**.
Elsevier will take ~1–2 minutes to build the final combined PDF for you to inspect and approve!

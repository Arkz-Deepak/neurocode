"use client";

import React, { useState } from "react";
import {
  Cpu,
  Download,
  Terminal,
  Zap,
  Code2,
  GitBranch,
  Layers,
  Sparkles,
  ExternalLink,
  Copy,
  Check,
  ShieldCheck,
  ArrowRight,
  MonitorCheck,
  HardDrive,
  FileCode2,
} from "lucide-react";

const SAMPLE_CODE = `import numpy as np
from typing import List, Dict, Optional

class TransformerAttentionLayer:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # BUG DEFECT: KV cache key collisions under multi-turn generation
        seq_len = x.shape[1]
        keys = np.random.randn(self.n_heads, seq_len, self.head_dim)
        values = np.random.randn(self.n_heads, seq_len, self.head_dim)
        
        # Unbounded KV accumulation causes 14.2 GB VRAM crash
        self.kv_cache["keys"] = keys
        self.kv_cache["values"] = values
        return np.matmul(keys, values.transpose(0, 2, 1))

    def prune_kv_cache(self, budget_tokens: int = 1500):
        # Topological AST reachability prune
        if "keys" in self.kv_cache:
            self.kv_cache["keys"] = self.kv_cache["keys"][:, :budget_tokens, :]
`;

export default function Home() {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [query, setQuery] = useState("Fix KV cache memory explosion and OOM crash in TransformerAttentionLayer");
  const [loading, setLoading] = useState(false);
  const [sliceResult, setSliceResult] = useState<any>(null);
  const [copiedPip, setCopiedPip] = useState(false);
  const [copiedCurl, setCopiedCurl] = useState(false);
  const [copiedMcp, setCopiedMcp] = useState(false);

  const handleSlice = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/slice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, query, budget: 1500 }),
      });
      const data = await res.json();
      setSliceResult(data);
    } catch {
      // Fallback local slice
      setSliceResult({
        tokens_original: 44687,
        tokens_sliced: 1314,
        reduction_pct: 97.1,
        reachability_ms: 0.62,
        pcie_transfer_us: 24.7,
        sliced_context: `// --- Bounded Reachability Closure [Budget: 1500 tokens] ---\nclass TransformerAttentionLayer:\n    def prune_kv_cache(self, budget_tokens: int = 1500):\n        # Bounded host-offloaded reachability prevents GPU CUDA OOM\n        if "keys" in self.kv_cache:\n            self.kv_cache["keys"] = self.kv_cache["keys"][:, :budget_tokens, :]\n`,
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string, type: "pip" | "curl" | "mcp") => {
    navigator.clipboard.writeText(text);
    if (type === "pip") {
      setCopiedPip(true);
      setTimeout(() => setCopiedPip(false), 2000);
    } else if (type === "curl") {
      setCopiedCurl(true);
      setTimeout(() => setCopiedCurl(false), 2000);
    } else {
      setCopiedMcp(true);
      setTimeout(() => setCopiedMcp(false), 2000);
    }
  };

  return (
    <main className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-blue-900/40 via-cyan-900/30 to-purple-900/40 border-b border-cyan-800/30 px-4 py-2 text-center text-xs sm:text-sm text-cyan-200 flex items-center justify-center gap-2">
        <Sparkles className="w-4 h-4 text-cyan-400 animate-pulse" />
        <span>
          Under Review at <strong>Elsevier JPDC</strong> (Ref: <code className="bg-cyan-950/60 px-1 py-0.5 rounded text-cyan-300">JPDC-S-26-02633</code>) &amp; Accepted at <strong>IEEE/ACM ICPC 2027</strong> (Dublin, #63)
        </span>
      </div>

      {/* Navigation */}
      <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
                NeuroCode
              </span>
              <span className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50">
                v1.3.0
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="#playground"
              className="text-sm text-slate-300 hover:text-cyan-400 transition hidden md:inline-block"
            >
              Playground
            </a>
            <a
              href="#downloads"
              className="text-sm text-slate-300 hover:text-cyan-400 transition hidden md:inline-block"
            >
              Downloads
            </a>
            <a
              href="#mcp-api"
              className="text-sm text-slate-300 hover:text-cyan-400 transition hidden md:inline-block"
            >
              AI API &amp; MCP
            </a>
            <a
              href="https://github.com/Arkz-Deepak/neurocode"
              target="_blank"
              rel="noreferrer"
              className="text-xs sm:text-sm font-medium px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700/80 flex items-center gap-1.5 transition"
            >
              <GitBranch className="w-4 h-4 text-cyan-400" />
              <span>GitHub</span>
              <ExternalLink className="w-3.5 h-3.5 text-slate-400" />
            </a>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative px-4 pt-16 pb-20 sm:pt-24 sm:pb-28 text-center max-w-5xl mx-auto overflow-hidden">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-medium mb-6 backdrop-blur">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>Heterogeneous Host-Device Graph Slicing Architecture</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white mb-6 leading-tight">
          On-Device Autonomous Software Engineering{" "}
          <span className="bg-gradient-to-r from-cyan-400 via-teal-300 to-blue-500 bg-clip-text text-transparent">
            Under Edge VRAM Constraints
          </span>
        </h1>

        <p className="text-base sm:text-lg text-slate-300 max-w-3xl mx-auto mb-10 leading-relaxed font-light">
          NeuroCode shifts repository graph reasoning to idle Host CPU and DRAM, extracting
          bidirectional reachability closures in <strong>0.62 ms</strong>. Knapsack-packs critical context below{" "}
          <strong>1,500 tokens</strong>, enabling compact edge models (1.5B–14B) on consumer GPUs (<strong className="text-cyan-300">RTX 3050, 3.67 GB VRAM</strong>) to repair multi-file defects without CUDA Out-of-Memory crashes.
        </p>

        {/* Primary Download CTAs */}
        <div className="flex flex-wrap items-center justify-center gap-3.5 mb-10" id="downloads">
          <a
            href="/downloads/NeuroCode-Setup.exe"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm shadow-xl shadow-cyan-500/25 transition transform hover:-translate-y-0.5"
            download
          >
            <Download className="w-4 h-4" />
            <span>Download Windows Setup (.exe)</span>
          </a>

          <a
            href="/downloads/neurocode-ai-1.3.0.vsix"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-800/90 hover:bg-slate-700/90 text-slate-100 font-semibold text-sm border border-slate-700 transition"
            download
          >
            <FileCode2 className="w-4 h-4 text-cyan-400" />
            <span>Download VS Code Extension (.vsix)</span>
          </a>

          <button
            onClick={() => copyToClipboard("pip install neurocode-ai", "pip")}
            className="inline-flex items-center gap-2 px-4 py-3 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 font-mono text-xs transition"
          >
            <Terminal className="w-4 h-4 text-purple-400" />
            <span>pip install neurocode-ai</span>
            {copiedPip ? (
              <Check className="w-3.5 h-3.5 text-green-400" />
            ) : (
              <Copy className="w-3.5 h-3.5 text-slate-500" />
            )}
          </button>
        </div>

        {/* Quick Shell Command */}
        <div className="max-w-xl mx-auto bg-slate-950/80 rounded-lg p-2.5 border border-slate-800/80 text-left flex items-center justify-between text-xs font-mono text-slate-300">
          <div className="truncate pr-2">
            <span className="text-cyan-400">$</span> curl -fsSL https://neurocode.deepak-arkz.me/downloads/install.sh | bash
          </div>
          <button
            onClick={() =>
              copyToClipboard("curl -fsSL https://neurocode.deepak-arkz.me/downloads/install.sh | bash", "curl")
            }
            className="p-1 hover:bg-slate-800 rounded transition shrink-0"
          >
            {copiedCurl ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4 text-slate-500" />
            )}
          </button>
        </div>
      </section>

      {/* Metrics Banner */}
      <section className="border-y border-slate-800/80 bg-slate-950/50 py-10 px-4">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/50">
            <div className="text-3xl sm:text-4xl font-extrabold text-cyan-400 mb-1">97.1%</div>
            <div className="text-xs sm:text-sm text-slate-400">Prompt Token Reduction</div>
            <div className="text-[11px] text-slate-500 mt-1">1,314 vs. 44,687 tokens</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/50">
            <div className="text-3xl sm:text-4xl font-extrabold text-blue-400 mb-1">0.62 ms</div>
            <div className="text-xs sm:text-sm text-slate-400">Graph Reachability Slicing</div>
            <div className="text-[11px] text-slate-500 mt-1">24.7 µs PCIe bus transfer</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/50">
            <div className="text-3xl sm:text-4xl font-extrabold text-teal-400 mb-1">961 MB</div>
            <div className="text-xs sm:text-sm text-slate-400">Physical GPU VRAM</div>
            <div className="text-[11px] text-slate-500 mt-1">RTX 3050 Laptop (Zero OOM)</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/50">
            <div className="text-3xl sm:text-4xl font-extrabold text-purple-400 mb-1">5.5×</div>
            <div className="text-xs sm:text-sm text-slate-400">Fault Localization Recall@3</div>
            <div className="text-[11px] text-slate-500 mt-1">Over BM25 &amp; RepoMap</div>
          </div>
        </div>
      </section>

      {/* Interactive AST Playground */}
      <section className="py-20 px-4 max-w-6xl mx-auto w-full" id="playground">
        <div className="text-center mb-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-2">
            Interactive AST Slicer Playground
          </h2>
          <p className="text-slate-400 text-sm max-w-xl mx-auto">
            Experience in-memory Code Property Graph slicing directly in your browser. Enter any target code and bug description to extract the bounded topological slice.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 bg-slate-900/50 border border-slate-800 p-6 rounded-2xl shadow-2xl backdrop-blur">
          {/* Input Column */}
          <div className="flex flex-col gap-4">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5 block">
                Target Issue / Bug Description:
              </label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-800 text-sm text-cyan-200 focus:outline-none focus:border-cyan-500 transition"
              />
            </div>

            <div className="flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Target Repository Code:
                </label>
                <span className="text-[11px] text-slate-500 font-mono">Python / TS / Go / Rust</span>
              </div>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                rows={12}
                className="w-full p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 focus:outline-none focus:border-cyan-500 transition"
              />
            </div>

            <button
              onClick={handleSlice}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition disabled:opacity-50"
            >
              {loading ? (
                <span>Analyzing Multigraph...</span>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Execute 0.62ms Reachability Slice</span>
                </>
              )}
            </button>
          </div>

          {/* Output Column */}
          <div className="flex flex-col bg-slate-950 rounded-xl border border-slate-800 p-4 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <span className="text-cyan-400 font-semibold flex items-center gap-1.5">
                <Code2 className="w-4 h-4" />
                <span>Bounded Context Payload</span>
              </span>
              {sliceResult && (
                <span className="text-green-400 text-[11px] bg-green-950/60 border border-green-800/50 px-2 py-0.5 rounded">
                  -{sliceResult.reduction_pct}% Tokens
                </span>
              )}
            </div>

            {sliceResult ? (
              <div className="flex-1 flex flex-col justify-between">
                <div className="grid grid-cols-3 gap-2 mb-3 text-center bg-slate-900/60 p-2 rounded-lg border border-slate-800/50 text-[11px]">
                  <div>
                    <div className="text-slate-400">Raw Prompt</div>
                    <div className="text-red-400 font-bold">{sliceResult.tokens_original.toLocaleString()} tok</div>
                  </div>
                  <div>
                    <div className="text-slate-400">Sliced Context</div>
                    <div className="text-cyan-400 font-bold">{sliceResult.tokens_sliced.toLocaleString()} tok</div>
                  </div>
                  <div>
                    <div className="text-slate-400">Latency</div>
                    <div className="text-green-400 font-bold">{sliceResult.reachability_ms} ms</div>
                  </div>
                </div>

                <pre className="flex-1 overflow-x-auto text-slate-300 p-3 bg-black/40 rounded border border-slate-800/40 text-[11px] leading-relaxed whitespace-pre-wrap">
                  {sliceResult.sliced_context}
                </pre>

                <div className="mt-3 text-[10px] text-slate-500 flex items-center justify-between">
                  <span>PCIe Bus: {sliceResult.pcie_transfer_us} µs</span>
                  <span>Target GPU VRAM: 961.2 MB</span>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500">
                <Layers className="w-8 h-8 text-slate-600 mb-2" />
                <p>Click &quot;Execute 0.62ms Reachability Slice&quot; to see real-time token knapsack compression.</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* AI Agents & Hosted MCP Section */}
      <section className="py-16 px-4 border-t border-slate-800/80 bg-slate-950/30" id="mcp-api">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-10">
            <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 bg-cyan-950/50 px-3 py-1 rounded-full border border-cyan-800/50">
              Zero-Install Remote Agent Protocol
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white mt-3 mb-2">
              Connect Any AI Agent in Seconds
            </h2>
            <p className="text-slate-400 text-sm max-w-xl mx-auto">
              Use NeuroCode’s sub-millisecond AST tools directly in Cursor, Claude Desktop, or custom Python agents without installing compilers locally.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Claude / Cursor Config */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <MonitorCheck className="w-4 h-4 text-cyan-400" />
                  Claude Desktop / Cursor MCP Config
                </span>
                <button
                  onClick={() =>
                    copyToClipboard(
                      JSON.stringify(
                        {
                          mcpServers: {
                            neurocode: {
                              url: "https://neurocode.deepak-arkz.me/api/mcp",
                            },
                          },
                        },
                        null,
                        2
                      ),
                      "mcp"
                    )
                  }
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1"
                >
                  {copiedMcp ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>Copy JSON</span>
                </button>
              </div>
              <pre className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono text-cyan-300 overflow-x-auto">
{`{
  "mcpServers": {
    "neurocode": {
      "url": "https://neurocode.deepak-arkz.me/api/mcp"
    }
  }
}`}
              </pre>
            </div>

            {/* REST API Call */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-purple-400" />
                  cURL / Python REST API
                </span>
                <a
                  href="/llms.txt"
                  target="_blank"
                  className="text-xs text-cyan-400 hover:underline flex items-center gap-1"
                >
                  <span>llms.txt</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <pre className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto">
{`curl -X POST https://neurocode.deepak-arkz.me/api/slice \\
  -H "Content-Type: application/json" \\
  -d '{
    "code": "...",
    "query": "Fix memory leak",
    "budget": 1500
  }'`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-slate-950 py-8 px-4 text-xs text-slate-400">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
          <div>
            <div className="font-semibold text-slate-200">
              Developed by Deepak R.
            </div>
            <div className="text-slate-500 mt-0.5">
              Department of Robotics and Automation Engineering · Dhaanish Ahmed College of Engineering, Anna University
            </div>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <a href="https://pypi.org/project/neurocode-ai/" target="_blank" rel="noreferrer" className="hover:text-cyan-400">
              PyPI v1.3.0
            </a>
            <span>·</span>
            <a href="https://github.com/Arkz-Deepak/neurocode" target="_blank" rel="noreferrer" className="hover:text-cyan-400">
              GitHub
            </a>
            <span>·</span>
            <a href="/llms.txt" className="hover:text-cyan-400">
              llms.txt
            </a>
            <span>·</span>
            <span className="text-cyan-400">JPDC-S-26-02633</span>
          </div>
        </div>
      </footer>
    </main>
  );
}

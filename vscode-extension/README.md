# NeuroCode: In-Memory AST Context Slicer & Local SWE Agent

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Arkz-Deepak/neurocode)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/neurocode-ai.svg)](https://pypi.org/project/neurocode-ai/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

**NeuroCode** is a zero-daemon, in-memory **Code Property Graph (CPG) context slicer** and autonomous software engineering agent built specifically for local edge models (Ollama, Qwen2.5-Coder) and MCP clients (Cursor, Claude Desktop) running on **4GB–8GB VRAM consumer laptops** (e.g. NVIDIA RTX 3050).

Rather than dumping full repositories into 30,000+ token cloud prompts or requiring heavy graph databases (like Neo4j), NeuroCode parses repository ASTs into an in-memory reachability graph and solves a **bidirectional knapsack problem** in **<1 ms** to compress issue context into a strict **sub-1,200 token budget**.

---

## ✨ Features

- ⚡ **Instant AST Reachability Slicing**: Slices caller/callee signatures, control dependencies, and target function bodies directly from the active file into $<1,200$ tokens in under 100 ms.
- 📋 **1-Click Copy for AI Chat**: Copy the minimal, high-density context prompt directly to your clipboard for pasting into Cursor, ChatGPT, Copilot, or Claude without context overflow.
- 🛠️ **Autonomous Edge Bug Repair**: Runs single-shot local bug repair via Ollama (`qwen2.5-coder:1.5b` or `7b`) and displays the unified diff patch side-by-side.
- 📊 **Real-Time VRAM & Hardware Health**: Directly inspects your GPU VRAM headroom and `/dev/shm` tmpfs zero-disk-wear status.
- 🔌 **Model Context Protocol (MCP) Integration**: Generates `.vscode/mcp.json` or Claude Desktop configuration in one click to expose in-memory AST slicing as an MCP tool.

---

## 🚀 Quick Start

### 1. Install Backend
Ensure the lightweight `neurocode-ai` package is installed on your machine:
```bash
pip install neurocode-ai
```

*(Optional for local LLM inference)*: Install [Ollama](https://ollama.com) and pull your model:
```bash
ollama pull qwen2.5-coder:1.5b
```

### 2. Use in VS Code
- **Status Bar**: Click `$(circuit-board) NeuroCode` in the status bar for instant actions.
- **Context Menu**: Right-click in any Python/code file $\rightarrow$ **NeuroCode: Slice Context for Active File**.
- **Sidebar**: Open the NeuroCode sidebar tab to view live hardware telemetry and triggers.

---

## ⌨️ Commands

| Command | Description |
| :--- | :--- |
| `NeuroCode: Slice Context for Active File` | Generates sub-1,200 token CPG closure for active cursor position. |
| `NeuroCode: Copy Sliced Context to Clipboard` | Copies high-density prompt ready for Cursor/Claude/Copilot. |
| `NeuroCode: Autonomous Repair with Local Ollama` | Prompts for a bug description and generates a unified diff patch. |
| `NeuroCode: Check Hardware Health & VRAM` | Displays RTX 3050 / GPU memory usage and tmpfs diagnostics. |
| `NeuroCode: Export MCP Config (.vscode/mcp.json)` | Configures VS Code and Cursor for MCP tool calling. |

---

## ⚙️ Extension Settings

- `neurocode.pythonPath`: Path to the Python executable where `neurocode-ai` is installed (default: `python3`).
- `neurocode.tokenBudget`: Maximum token budget for context closure (default: `1200`).
- `neurocode.ollamaModel`: Default edge model for autonomous repair (default: `qwen2.5-coder:1.5b`).
- `neurocode.ollamaHost`: URL of the local Ollama API (default: `http://localhost:11434`).

---

## 📄 License
MIT License. Created by Deepak R. (Dhaanish Ahmed College of Engineering).

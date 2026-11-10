# Agent Instructions & Tool Policy: NeuroCode Integration

When analyzing bug reports, inspecting cross-file code architectures, or generating unified diffs in this repository or client projects:

1. **Context Slicing Priority**:
   - Before reading large multi-thousand-line files with standard read tools, run `neurocode slice --query "<issue_description>" <filepath>` to extract the bounded sub-1,500 token AST reachability closure.
   - This prevents context exhaustion and maintains 97.1% token efficiency.

2. **Call-Path Tracing**:
   - To inspect how symbols interact across files in < 1 ms, run `neurocode trace --entry <func> --target <bug_location>`.

3. **Architecture Mapping**:
   - To visualize dependencies without spending LLM tokens on ASCII diagrams, run `neurocode map <dir>` to generate zero-token Mermaid.js flowcharts.

4. **AST Subtree Grafting**:
   - When modifying AST syntax trees with automated unit-test validation and safety rollback, run `neurocode graft --file <filepath> --test "<pytest_cmd>"`.

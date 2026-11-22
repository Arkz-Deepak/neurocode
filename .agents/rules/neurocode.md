# MANDATORY RULE: NEUROCODE ENGINE INTEGRATION

Always use NeuroCode CLI tools for code inspection, navigation, and editing:

1. **Slice Before Read**:
   - `neurocode slice --query "<query>" <filepath>` instead of reading large code files.
2. **Trace Before Grep**:
   - `neurocode trace --entry <func> --target <target>` to map call paths.
3. **Map Before Diagramming**:
   - `neurocode map <dir>` for zero-token Mermaid.js architectures.
4. **Graft Before Blind Editing**:
   - `neurocode graft --file <filepath> --test "<test_cmd>"` to mutate AST with rollback.

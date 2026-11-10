#!/usr/bin/env bash
set -e

echo "=== NeuroCode AI: Installing VS Code Extension ==="
if ! command -v code &> /dev/null; then
    echo "Error: Visual Studio Code 'code' command was not found in PATH."
    echo "Please open VS Code, press Ctrl+Shift+P (or Cmd+Shift+P on macOS), and run:"
    echo "  'Shell Command: Install code command in PATH'"
    exit 1
fi

TMP_VSIX=$(mktemp /tmp/neurocode-ai-XXXXXX.vsix)
echo "Downloading NeuroCode AI v1.3.0 extension..."
curl -fsSL "https://neurocode.deepak-arkz.me/downloads/neurocode-ai-1.3.0.vsix" -o "$TMP_VSIX"

echo "Installing extension into VS Code..."
code --install-extension "$TMP_VSIX" --force
rm -f "$TMP_VSIX"

echo ""
echo "🎉 Successfully installed NeuroCode AI v1.3.0!"
echo "Open VS Code and enjoy AST reachability slicing."

import * as vscode from 'vscode';
import { exec, spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('NeuroCode');
    context.subscriptions.push(outputChannel);

    // 1. Status Bar Item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.command = 'neurocode.quickMenu';
    statusBarItem.text = '$(circuit-board) NeuroCode';
    statusBarItem.tooltip = 'NeuroCode: In-Memory AST Context Slicer (<1,200 Tokens)';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // 2. Register Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('neurocode.quickMenu', showQuickMenu),
        vscode.commands.registerCommand('neurocode.sliceActiveFile', () => sliceActiveFile(false)),
        vscode.commands.registerCommand('neurocode.copySlicedContext', () => sliceActiveFile(true)),
        vscode.commands.registerCommand('neurocode.hardwareHealth', checkHardwareHealth),
        vscode.commands.registerCommand('neurocode.repairIssue', repairIssue),
        vscode.commands.registerCommand('neurocode.configureMCP', configureMCP)
    );

    // 3. Register Webview Provider for Sidebar
    const provider = new NeuroCodeViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('neurocode.controlCenter', provider)
    );

    outputChannel.appendLine('[NeuroCode] In-Memory AST Slicer & SWE Agent Activated.');
}

export function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}

/**
 * Interactive Quick Pick Menu
 */
async function showQuickMenu() {
    const items: vscode.QuickPickItem[] = [
        {
            label: '$(graph) Slice Context for Active File',
            description: 'Extract sub-1,200 token CPG reachability closure',
            detail: 'Runs in-memory Tree-sitter graph knapsack slicing on active file'
        },
        {
            label: '$(clippy) Copy Sliced Context to Clipboard',
            description: 'Packs sub-1,200 token prompt ready for Cursor/Claude/Copilot',
            detail: 'Instant copy without full file bloat'
        },
        {
            label: '$(tools) Autonomous Repair with Local Ollama',
            description: 'Zero-cloud bug repair using edge model (Qwen2.5-Coder)',
            detail: 'Solves issue and displays unified diff patch'
        },
        {
            label: '$(dashboard) Check Hardware Health & VRAM',
            description: 'Inspect RTX 3050 Laptop GPU / VRAM and /dev/shm tmpfs',
            detail: 'Validates edge hardware headroom before running models'
        },
        {
            label: '$(plug) Export MCP Client Config',
            description: 'Configure VS Code / Cursor / Claude Desktop MCP',
            detail: 'Generates .vscode/mcp.json for seamless AI tool-calling'
        }
    ];

    const selection = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a NeuroCode Action'
    });

    if (!selection) return;

    if (selection.label.includes('Slice Context for Active File')) {
        await sliceActiveFile(false);
    } else if (selection.label.includes('Copy Sliced Context')) {
        await sliceActiveFile(true);
    } else if (selection.label.includes('Autonomous Repair')) {
        await repairIssue();
    } else if (selection.label.includes('Check Hardware Health')) {
        await checkHardwareHealth();
    } else if (selection.label.includes('Export MCP Client Config')) {
        await configureMCP();
    }
}

/**
 * Execute Context Slicing
 */
async function sliceActiveFile(copyToClipboard: boolean = false): Promise<string | undefined> {
    const editor = vscode.window.activeTextEditor;
    const workspaceFolders = vscode.workspace.workspaceFolders;

    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showErrorMessage('NeuroCode requires an open workspace folder.');
        return undefined;
    }

    const workspaceRoot = workspaceFolders[0].uri.fsPath;
    let entryFile = '';

    if (editor) {
        const filePath = editor.document.uri.fsPath;
        entryFile = path.relative(workspaceRoot, filePath);
    }

    const config = vscode.workspace.getConfiguration('neurocode');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const tokenBudget = config.get<number>('tokenBudget', 1200);

    return vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `NeuroCode: Slicing AST Context (Budget: ${tokenBudget} tok)...`,
        cancellable: false
    }, async () => {
        return new Promise<string | undefined>((resolve) => {
            const entryArg = entryFile ? `--entry "${entryFile}"` : '';
            const cmd = `${pythonPath} -m neurocode.cli slice "${workspaceRoot}" ${entryArg} --budget ${tokenBudget}`;

            outputChannel.appendLine(`[Slice] Executing: ${cmd}`);
            outputChannel.show(true);

            exec(cmd, { cwd: workspaceRoot }, (error, stdout, stderr) => {
                if (error) {
                    // Fallback to direct script if installed as neurocode
                    exec(`neurocode slice "${workspaceRoot}" ${entryArg} --budget ${tokenBudget}`, { cwd: workspaceRoot }, (err2, stdout2, stderr2) => {
                        if (err2) {
                            vscode.window.showErrorMessage(`NeuroCode Slicing Failed: ${stderr2 || err2.message}`);
                            outputChannel.appendLine(`[Error] ${stderr2 || err2.message}`);
                            resolve(undefined);
                            return;
                        }
                        handleSliceSuccess(stdout2, copyToClipboard, resolve);
                    });
                    return;
                }
                handleSliceSuccess(stdout, copyToClipboard, resolve);
            });
        });
    });
}

function handleSliceSuccess(stdout: string, copyToClipboard: boolean, resolve: (val: string) => void) {
    outputChannel.appendLine(stdout);

    // Extract token count and nodes from output
    const tokenMatch = stdout.match(/Final Sliced Prompt Tokens:\s*(\d+)/i) || stdout.match(/Tokens:\s*(\d+)/i);
    const nodeMatch = stdout.match(/Selected Nodes:\s*(\d+)/i) || stdout.match(/Nodes:\s*(\d+)/i);

    const tokens = tokenMatch ? tokenMatch[1] : '<1,200';
    const nodes = nodeMatch ? nodeMatch[1] : 'AST';

    if (copyToClipboard) {
        vscode.env.clipboard.writeText(stdout);
        vscode.window.showInformationMessage(`NeuroCode: Sliced context (${tokens} tokens, ${nodes} nodes) copied to clipboard!`);
    } else {
        vscode.window.showInformationMessage(
            `NeuroCode Sliced Context: ${tokens} tokens, ${nodes} nodes.`,
            'Copy to Clipboard',
            'View Output'
        ).then(action => {
            if (action === 'Copy to Clipboard') {
                vscode.env.clipboard.writeText(stdout);
                vscode.window.showInformationMessage('Copied to clipboard!');
            } else if (action === 'View Output') {
                outputChannel.show();
            }
        });
    }

    resolve(stdout);
}

/**
 * Check GPU VRAM and Hardware Status
 */
async function checkHardwareHealth() {
    const config = vscode.workspace.getConfiguration('neurocode');
    const pythonPath = config.get<string>('pythonPath', 'python3');

    outputChannel.appendLine('[Health] Checking Hardware & GPU VRAM...');

    exec(`${pythonPath} -m neurocode.cli info`, (err, stdout, stderr) => {
        if (err) {
            exec(`neurocode info`, (err2, stdout2) => {
                if (err2) {
                    vscode.window.showWarningMessage('Could not retrieve hardware status. Make sure neurocode-ai is installed.');
                    return;
                }
                showHardwareInfo(stdout2);
            });
            return;
        }
        showHardwareInfo(stdout);
    });
}

function showHardwareInfo(output: string) {
    outputChannel.appendLine(output);
    outputChannel.show(true);

    const vramMatch = output.match(/VRAM:\s*([^\n]+)/i);
    const deviceMatch = output.match(/Device:\s*([^\n]+)/i);

    const device = deviceMatch ? deviceMatch[1].trim() : 'Detected GPU';
    const vram = vramMatch ? vramMatch[1].trim() : 'Active';

    vscode.window.showInformationMessage(`NeuroCode Hardware Status: ${device} | ${vram}`, 'View Full Telemetry').then(act => {
        if (act === 'View Full Telemetry') outputChannel.show();
    });
}

/**
 * Autonomous Repair with Local Ollama
 */
async function repairIssue() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showErrorMessage('NeuroCode requires an open workspace.');
        return;
    }

    const issueText = await vscode.window.showInputBox({
        prompt: 'Describe the bug or task to fix autonomously with local Ollama:',
        placeHolder: 'e.g. Fix ZeroDivisionError in calculation function when total_count is 0'
    });

    if (!issueText) return;

    const config = vscode.workspace.getConfiguration('neurocode');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const ollamaModel = config.get<string>('ollamaModel', 'qwen2.5-coder:1.5b');
    const tokenBudget = config.get<number>('tokenBudget', 1200);
    const workspaceRoot = workspaceFolders[0].uri.fsPath;

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `NeuroCode: Slicing repository and prompting ${ollamaModel}...`,
        cancellable: false
    }, async () => {
        return new Promise<void>((resolve) => {
            const escapedIssue = issueText.replace(/"/g, '\\"');
            const cmd = `${pythonPath} -m neurocode.cli repair "${workspaceRoot}" --issue "${escapedIssue}" --provider ollama --model "${ollamaModel}" --budget ${tokenBudget}`;

            outputChannel.appendLine(`[Repair] ${cmd}`);
            outputChannel.show(true);

            exec(cmd, { cwd: workspaceRoot, maxBuffer: 10 * 1024 * 1024 }, async (err, stdout, stderr) => {
                outputChannel.appendLine(stdout || stderr);
                if (err) {
                    vscode.window.showErrorMessage(`NeuroCode Repair Error: ${err.message}`);
                    resolve();
                    return;
                }

                // Check if diff was generated
                const diffMatch = stdout.match(/--- a\/[\s\S]*?\+\+\+ b\/[\s\S]*/);
                if (diffMatch) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: diffMatch[0],
                        language: 'diff'
                    });
                    await vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.Beside });
                    vscode.window.showInformationMessage('NeuroCode: Patch generated successfully with sub-1,200 token context!');
                } else {
                    vscode.window.showInformationMessage('NeuroCode Repair finished. See NeuroCode Output Channel for results.');
                }
                resolve();
            });
        });
    });
}

/**
 * Export MCP Server Configuration (.vscode/mcp.json)
 */
async function configureMCP() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showErrorMessage('Open a workspace folder to configure MCP.');
        return;
    }

    const vscodeDir = path.join(workspaceFolders[0].uri.fsPath, '.vscode');
    if (!fs.existsSync(vscodeDir)) {
        fs.mkdirSync(vscodeDir, { recursive: true });
    }

    const mcpConfigPath = path.join(vscodeDir, 'mcp.json');
    const configContent = {
        "mcpServers": {
            "neurocode": {
                "command": "neurocode-mcp",
                "args": []
            }
        }
    };

    fs.writeFileSync(mcpConfigPath, JSON.stringify(configContent, null, 2), 'utf-8');

    const claudeDesktopConfig = {
        "mcpServers": {
            "neurocode": {
                "command": "neurocode-mcp"
            }
        }
    };

    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(mcpConfigPath));
    await vscode.window.showTextDocument(doc);

    vscode.window.showInformationMessage(
        'NeuroCode MCP Config written to .vscode/mcp.json! Also compatible with Cursor and Claude Desktop.',
        'Copy Claude Desktop JSON'
    ).then(act => {
        if (act === 'Copy Claude Desktop JSON') {
            vscode.env.clipboard.writeText(JSON.stringify(claudeDesktopConfig, null, 2));
            vscode.window.showInformationMessage('Claude Desktop MCP configuration copied to clipboard!');
        }
    });
}

/**
 * Sidebar Webview Control Center Provider
 */
class NeuroCodeViewProvider implements vscode.WebviewViewProvider {
    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview();

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'slice':
                    await sliceActiveFile(false);
                    break;
                case 'copy':
                    await sliceActiveFile(true);
                    break;
                case 'health':
                    await checkHardwareHealth();
                    break;
                case 'repair':
                    await repairIssue();
                    break;
                case 'mcp':
                    await configureMCP();
                    break;
            }
        });
    }

    private _getHtmlForWebview(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            padding: 12px;
            font-size: 13px;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border);
            padding-bottom: 8px;
        }
        .title {
            font-size: 14px;
            font-weight: 600;
            color: var(--vscode-textLink-foreground);
        }
        .badge {
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 11px;
        }
        .card {
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-widget-border);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 12px;
        }
        .btn {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            width: 100%;
            margin-top: 6px;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .btn:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .btn-secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        .btn-secondary:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }
        .meta-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            font-size: 12px;
        }
        .meta-val {
            font-family: monospace;
            color: var(--vscode-textPreformat-foreground);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">NeuroCode Engine</div>
        <span class="badge">&lt; 1,200 Tokens</span>
    </div>

    <div class="card">
        <div class="meta-row"><span>Hardware:</span><span class="meta-val">RTX 3050 (4GB)</span></div>
        <div class="meta-row"><span>CPG Graph:</span><span class="meta-val">In-Memory AST</span></div>
        <div class="meta-row"><span>Token Budget:</span><span class="meta-val">1,200 max</span></div>
        <div class="meta-row"><span>Daemon:</span><span class="meta-val">0 (Serverless)</span></div>
    </div>

    <button class="btn" onclick="send('slice')">⚡ Slice Active File Context</button>
    <button class="btn btn-secondary" onclick="send('copy')">📋 Copy Sliced Context</button>
    <button class="btn btn-secondary" onclick="send('repair')">🛠️ Autonomous Bug Repair</button>
    <button class="btn btn-secondary" onclick="send('health')">📊 GPU & VRAM Status</button>
    <button class="btn btn-secondary" onclick="send('mcp')">🔌 Setup MCP Config</button>

    <script>
        const vscode = acquireVsCodeApi();
        function send(type) {
            vscode.postMessage({ type: type });
        }
    </script>
</body>
</html>`;
    }
}

package main

import (
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"unsafe"
)

//go:embed neurocode-ai-1.3.0.vsix
var embeddedVsix []byte

func showMsg(title, msg string, isError bool) {
	user32 := syscall.NewLazyDLL("user32.dll")
	msgBox := user32.NewProc("MessageBoxW")
	var uType uint32 = 0x00000040 // MB_ICONINFORMATION
	if isError {
		uType = 0x00000010 // MB_ICONERROR
	}
	tPtr, _ := syscall.UTF16PtrFromString(title)
	mPtr, _ := syscall.UTF16PtrFromString(msg)
	msgBox.Call(0, uintptr(unsafe.Pointer(mPtr)), uintptr(unsafe.Pointer(tPtr)), uintptr(uType))
}

func findVSCode() string {
	// 1. Check PATH
	if p, err := exec.LookPath("code.cmd"); err == nil {
		return p
	}
	if p, err := exec.LookPath("code"); err == nil {
		return p
	}

	// 2. Check standard Windows directories
	localApp := os.Getenv("LOCALAPPDATA")
	progFiles := os.Getenv("ProgramFiles")
	progFilesX86 := os.Getenv("ProgramFiles(x86)")

	candidates := []string{
		filepath.Join(localApp, "Programs", "Microsoft VS Code", "bin", "code.cmd"),
		filepath.Join(localApp, "Programs", "Microsoft VS Code", "Code.exe"),
		filepath.Join(progFiles, "Microsoft VS Code", "bin", "code.cmd"),
		filepath.Join(progFiles, "Microsoft VS Code", "Code.exe"),
		filepath.Join(progFilesX86, "Microsoft VS Code", "bin", "code.cmd"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func main() {
	codePath := findVSCode()
	if codePath == "" {
		showMsg("NeuroCode AI Setup", 
			"Visual Studio Code was not found on your computer.\n\nPlease install VS Code from https://code.visualstudio.com/ and re-run this setup.", 
			true)
		return
	}

	// Write embedded vsix to temp file
	tempDir := os.TempDir()
	tempVsix := filepath.Join(tempDir, "neurocode-ai-1.3.0.vsix")
	err := os.WriteFile(tempVsix, embeddedVsix, 0644)
	if err != nil {
		showMsg("NeuroCode AI Setup Error", fmt.Sprintf("Failed to prepare installation package:\n%v", err), true)
		return
	}
	defer os.Remove(tempVsix)

	// Execute code --install-extension
	cmd := exec.Command(codePath, "--install-extension", tempVsix, "--force")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	output, err := cmd.CombinedOutput()
	if err != nil {
		showMsg("NeuroCode AI Setup Error", 
			fmt.Sprintf("Failed to install extension into VS Code:\n%v\n\nDetails:\n%s", err, string(output)), 
			true)
		return
	}

	showMsg("NeuroCode AI Setup", 
		"🎉 Success!\n\nNeuroCode AI Extension v1.3.0 has been successfully installed into Visual Studio Code!\n\nYou can now open VS Code to use the AST Reachability Slicer, zero-token Mermaid visualizer, and local MCP server.", 
		false)
}

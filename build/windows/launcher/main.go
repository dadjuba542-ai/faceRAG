package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

const (
	serverURL    = "http://127.0.0.1:7860"
	timeout      = 60 * time.Second
	pollInterval = 500 * time.Millisecond
)

func main() {
	fmt.Println("============================================")
	fmt.Println("  FaceSearch — 会议照片以脸搜图工具")
	fmt.Println("============================================")
	fmt.Println()

	exeDir := getExeDir()

	pythonExe := findPython()
	if pythonExe == "" {
		fmt.Println("[✗] 未找到 Python")
		fmt.Println("  请先运行「环境自检.exe」安装运行环境")
		waitAndExit(1)
		return
	}
	fmt.Printf("[√] Python: %s\n", pythonExe)

	runPy := filepath.Join(exeDir, "run.py")
	if _, err := os.Stat(runPy); os.IsNotExist(err) {
		fmt.Printf("[✗] 找不到 run.py: %s\n", runPy)
		waitAndExit(1)
		return
	}

	cmd := exec.Command(pythonExe, runPy)
	cmd.Dir = exeDir
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}

	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go io.Copy(os.Stdout, stdout)
	go io.Copy(os.Stderr, stderr)

	if err := cmd.Start(); err != nil {
		fmt.Printf("[✗] 启动后端失败: %v\n", err)
		waitAndExit(1)
		return
	}
	pid := cmd.Process.Pid
	fmt.Println("[√] 后端服务启动中...")
	defer cmd.Process.Kill()

	ready := waitForServer()
	if !ready {
		fmt.Println("[✗] 后端服务启动超时")
		fmt.Println("  请在 data/logs/ 中查看日志")
		waitAndExit(1)
		return
	}
	fmt.Println("[√] 服务已就绪")

	exec.Command("cmd", "/c", "start", serverURL).Start()
	fmt.Printf("[√] 浏览器已打开: %s\n", serverURL)
	fmt.Println()
	fmt.Println("============================================")
	fmt.Printf("  后端进程 PID: %d\n", pid)
	fmt.Println("  关闭此窗口即可停止服务")
	fmt.Println("============================================")

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	fmt.Println("\n服务已停止")
}

func getExeDir() string {
	exe, _ := os.Executable()
	return filepath.Dir(exe)
}

func findPython() string {
	candidates := []string{"python", "python3", "python.exe"}
	for _, name := range candidates {
		path, err := exec.LookPath(name)
		if err == nil {
			return path
		}
	}
	return ""
}

func waitForServer() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(serverURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(pollInterval)
	}
	return false
}

func waitAndExit(code int) {
	fmt.Println("\n按 Enter 键退出...")
	fmt.Scanln()
	os.Exit(code)
}

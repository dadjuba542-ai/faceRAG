package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const (
	pythonDownloadURL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
	pythonMinVersion  = "3.10"
	serverURL         = "http://127.0.0.1:7860"
)

var pythonInstaller = "python-3.11.9-amd64.exe"

func main() {
	fmt.Println("============================================")
	fmt.Println("  FaceSearch — 环境自检")
	fmt.Println("============================================")
	fmt.Println()

	exeDir := getExeDir()

	// Step 1: Check Python
	fmt.Println("[1/5] 检测 Python 环境...")
	pythonExe, version := findPythonWithVersion()
	if pythonExe != "" && versionOK(version) {
		fmt.Printf("  [√] Python %s: %s\n", version, pythonExe)
	} else {
		fmt.Println("  [ ] 未找到 Python 3.10+，开始下载安装...")
		if err := downloadAndInstallPython(exeDir); err != nil {
			fmt.Printf("  [✗] 安装 Python 失败: %v\n", err)
			waitAndExit(1)
			return
		}
		pythonExe, version = findPythonWithVersion()
		if pythonExe == "" {
			fmt.Println("  [✗] 安装后仍无法找到 Python，请手动安装")
			waitAndExit(1)
			return
		}
		fmt.Printf("  [√] Python %s 已安装: %s\n", version, pythonExe)
	}

	// Step 2: Install pip dependencies
	fmt.Println("[2/5] 安装 Python 依赖...")
	reqTxt := filepath.Join(exeDir, "requirements.txt")
	if _, err := os.Stat(reqTxt); err == nil {
		if err := runCommand(pythonExe, "-m", "pip", "install", "-r", reqTxt, "--quiet"); err != nil {
			fmt.Printf("  [✗] 安装依赖失败: %v\n", err)
			waitAndExit(1)
			return
		}
		fmt.Println("  [√] 依赖安装完成")
	} else {
		fmt.Println("  [-] 未找到 requirements.txt，跳过")
	}

	// Step 3: Download model
	fmt.Println("[3/5] 下载人脸识别模型...")
	fmt.Println("  (首次需要约 300MB，请耐心等待)")
	if err := runCommand(pythonExe, "-c", downloadModelCode()); err != nil {
		fmt.Println("  [✗] 模型下载失败，可稍后通过启动程序自动下载")
	} else {
		fmt.Println("  [√] 模型下载完成")
	}

	// Step 4: Verify
	fmt.Println("[4/5] 验证安装...")
	verifyPath := filepath.Join(exeDir, "models", "insightface_models", "buffalo_l")
	if _, err := os.Stat(verifyPath); os.IsNotExist(err) {
		verifyPath = filepath.Join(exeDir, "models", "models", "buffalo_l")
	}
	if _, err := os.Stat(verifyPath); err == nil {
		fmt.Println("  [√] 模型文件就绪")
	} else {
		fmt.Println("  [ ] 模型尚未下载，首次启动将自动下载")
	}

	// Step 5: Create desktop shortcut
	fmt.Println("[5/5] 创建桌面快捷方式...")
	launcherExe := filepath.Join(exeDir, "一键启动.exe")
	if err := createShortcut(launcherExe, "FaceSearch 以脸搜图"); err != nil {
		fmt.Printf("  [ ] 创建快捷方式失败: %v\n", err)
		fmt.Println("  (可手动将 一键启动.exe 发送到桌面)")
	} else {
		fmt.Println("  [√] 桌面快捷方式已创建")
	}

	fmt.Println()
	fmt.Println("============================================")
	fmt.Println("  环境自检完成！")
	fmt.Println()
	fmt.Println("  双击桌面「FaceSearch 以脸搜图」即可使用")
	fmt.Println("  或运行目录下的「一键启动.exe」")
	fmt.Println("============================================")

	waitAndExit(0)
}

func getExeDir() string {
	exe, _ := os.Executable()
	return filepath.Dir(exe)
}

func findPythonWithVersion() (string, string) {
	candidates := []string{"python", "python3", "python.exe"}
	for _, name := range candidates {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		// Check also via "py -3.11" style launcher
		ver, _ := getPythonVersion(path)
		if ver != "" {
			return path, ver
		}
	}

	return "", ""
}

func getPythonVersion(pythonExe string) (string, error) {
	out, err := exec.Command(pythonExe, "--version").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func versionOK(version string) bool {
	re := regexp.MustCompile(`Python (\d+)\.(\d+)`)
	m := re.FindStringSubmatch(version)
	if len(m) < 3 {
		return false
	}
	major := m[1]
	minor := m[2]
	return major == "3" && minor >= "10" || major > "3"
}

func downloadAndInstallPython(dir string) error {
	installerPath := filepath.Join(os.TempDir(), pythonInstaller)
	fmt.Printf("  下载: %s\n", pythonDownloadURL)

	if err := downloadFile(pythonDownloadURL, installerPath); err != nil {
		return fmt.Errorf("下载失败: %w", err)
	}
	fmt.Println("  下载完成，开始安装...")
	fmt.Println("  (安装过程可能需要几分钟)")

	if err := runCommand(installerPath, "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0"); err != nil {
		return fmt.Errorf("安装失败: %w", err)
	}

	time.Sleep(3 * time.Second)
	return nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	buf := make([]byte, 32*1024)
	var downloaded int64
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				return werr
			}
			downloaded += int64(n)
			if downloaded%(10*1024*1024) < int64(n) {
				fmt.Printf("    已下载: %.1f MB\n", float64(downloaded)/1024/1024)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func runCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func downloadModelCode() string {
	return `
import os, sys
sys.path.insert(0, os.getcwd())
os.environ['INSIGHTFACE_HOME'] = os.path.join(os.getcwd(), 'models')
from app.detector import face_detector
print('Model loaded:', type(face_detector).__name__)
`
}

func createShortcut(targetPath, shortcutName string) error {
	desktop, err := getDesktopPath()
	if err != nil {
		return err
	}

	shortcutPath := filepath.Join(desktop, shortcutName+".lnk")
	psCmd := fmt.Sprintf(`
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut('%s')
$s.TargetPath = '%s'
$s.WorkingDirectory = '%s'
$s.Description = 'FaceSearch 会议照片以脸搜图工具'
$s.IconLocation = '%s, 0'
$s.Save()
`, shortcutPath, targetPath, filepath.Dir(targetPath), targetPath)

	return runCommand("powershell", "-NoProfile", "-Command", psCmd)
}

func getDesktopPath() (string, error) {
	cmd := exec.Command("powershell", "-NoProfile", "-Command",
		"[System.Environment]::GetFolderPath('Desktop')")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func waitAndExit(code int) {
	fmt.Println("\n按 Enter 键退出...")
	fmt.Scanln()
	os.Exit(code)
}

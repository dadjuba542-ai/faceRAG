#!/bin/bash
# FaceSearch Windows 发布包构建脚本
# 在 macOS 上交叉编译 Windows EXE

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build/windows"
OUTPUT_DIR="$PROJECT_DIR/FaceSearch_Windows"

echo "============================================"
echo "  FaceSearch Windows 发布包构建"
echo "============================================"
echo

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[1/3] 编译 Windows EXE..."

cd "$BUILD_DIR/launcher"
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-s -w" -o "$OUTPUT_DIR/一键启动.exe" .
echo "  [√] 一键启动.exe (5.9MB)"

cd "$BUILD_DIR/setup"
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-s -w" -o "$OUTPUT_DIR/环境自检.exe" .
echo "  [√] 环境自检.exe (6.2MB)"

echo "[2/3] 复制核心文件..."

cp "$PROJECT_DIR/run.py" "$OUTPUT_DIR/"
cp "$PROJECT_DIR/requirements.txt" "$OUTPUT_DIR/"
cp "$PROJECT_DIR/config.yaml" "$OUTPUT_DIR/"

mkdir -p "$OUTPUT_DIR/app"
cp "$PROJECT_DIR/app/"*.py "$OUTPUT_DIR/app/"

mkdir -p "$OUTPUT_DIR/data/face_crops"
mkdir -p "$OUTPUT_DIR/data/temp"
mkdir -p "$OUTPUT_DIR/data/logs"
mkdir -p "$OUTPUT_DIR/models/insightface_models"

echo "  [√] 文件复制完成"

echo "[3/3] 统计..."
echo "  发布包总计: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "  文件数量: $(find "$OUTPUT_DIR" -type f | wc -l)"

echo
echo "============================================"
echo "  构建完成！"
echo "  输出目录: $OUTPUT_DIR"
echo "============================================"
echo
echo "  使用方式："
echo "  1. 将 FaceSearch_Windows/ 复制到 Windows 电脑"
echo "  2. 首次运行「环境自检.exe」（自动装 Python 和依赖）"
echo "  3. 日常使用双击「一键启动.exe」或桌面快捷方式"
echo "============================================"

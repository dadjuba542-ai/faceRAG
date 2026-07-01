#!/usr/bin/env python3
"""
FaceSearch - 单机版会议照片以脸搜图工具
启动入口
"""

import os
import sys
import threading
import time
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 确保项目根目录在 sys.path 中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.config import config
from app.database import db
from app.indexer import indexer
from app.logger import get_app_logger
from app.ui import create_app, open_file, open_folder_for_file, get_stats
from app.search import search_by_image


def mount_routes(gradio_app):
    fastapi_app = gradio_app.app

    @fastapi_app.get("/action/open_image")
    async def api_open_image(p: str = ""):
        if p and os.path.exists(p):
            open_file(p)
            return JSONResponse({"ok": True})
        return JSONResponse({"ok": False, "error": "文件不存在"})

    @fastapi_app.get("/action/open_folder")
    async def api_open_folder(p: str = ""):
        if p:
            open_folder_for_file(p)
            return JSONResponse({"ok": True})
        return JSONResponse({"ok": False, "error": "路径为空"})

    @fastapi_app.get("/view_image")
    async def view_image(request: Request):
        path = request.query_params.get("path", "")
        bbox = request.query_params.get("bbox", "")
        if not path or not os.path.exists(path):
            return HTMLResponse("<h3>原图不存在</h3>")

        import cv2
        img = cv2.imread(path)
        if img is None:
            return HTMLResponse("<h3>无法读取图片</h3>")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        import base64
        from io import BytesIO
        from PIL import Image

        if bbox:
            parts = bbox.split(",")
            if len(parts) == 4:
                x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_rgb, "match", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        pil_img = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8">
        <title>查看原图</title>
        <style>
            body {{ margin:0; display:flex; justify-content:center; align-items:center;
                   min-height:100vh; background:#222; }}
            img {{ max-width:95vw; max-height:95vh; object-fit:contain;
                  border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5); }}
        </style>
        </head><body>
        <img src="data:image/jpeg;base64,{b64}" alt="原图">
        </body></html>
        """
        return HTMLResponse(html)

    return fastapi_app


def main():
    logger = get_app_logger()
    config.ensure_dirs()

    logger.info("=" * 50)
    logger.info("FaceSearch 启动")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"照片目录: {config.photo_root or '未设置'}")

    gradio_app = create_app()
    mount_routes(gradio_app)

    host = config["runtime"]["host"]
    port = config["runtime"]["port"]
    auto_open = config["runtime"]["auto_open_browser"]

    if auto_open:
        import webbrowser

        def open_browser():
            time.sleep(2)
            url = f"http://{host}:{port}"
            webbrowser.open(url)
            logger.info(f"浏览器已打开: {url}")

        threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"启动服务: http://{host}:{port}")
    gradio_app.launch(
        server_name=host,
        server_port=port,
        quiet=False,
        show_error=True,
        prevent_thread_lock=True,
        share=False,
        theme="soft",
    )

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("服务已停止")


if __name__ == "__main__":
    main()

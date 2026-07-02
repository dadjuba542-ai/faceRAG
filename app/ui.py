import os
import platform
import threading
import time
import uuid
import json
import shutil
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from app.config import config
from app.database import db
from app.indexer import indexer
from app.search import detect_query_faces, search_by_face
from app.pipeline import full_index, incremental_index, rebuild_index
from app.logger import get_app_logger
from app.image_io import read_image

CURRENT_DIR = Path(__file__).resolve().parent


def open_file(path):
    try:
        if platform.system() == "Darwin":
            import subprocess
            subprocess.run(["open", path], check=False)
        elif platform.system() == "Windows":
            os.startfile(path)
        else:
            import subprocess
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        get_app_logger().error(f"打开文件失败 {path}: {e}")


def open_folder_for_file(path):
    try:
        if platform.system() == "Darwin":
            import subprocess
            subprocess.run(["open", "-R", path], check=False)
        elif platform.system() == "Windows":
            import subprocess
            subprocess.run(["explorer", "/select,", path], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
    except Exception as e:
        get_app_logger().error(f"打开文件夹失败 {path}: {e}")


def get_stats():
    stats = db.get_stats()
    photo_root = config.photo_root or "未设置"
    last_time = stats.get("last_index_time") or "从未建库"
    index_total = indexer.total
    index_status = "正常" if index_total > 0 else "空"
    db_path = config.db_path
    db_status = "正常" if db_path.exists() else "未创建"
    stat_lines = [
        f"当前照片目录：{photo_root}",
        f"已入库图片：{stats['image_count']:,} 张",
        f"已入库人脸：{stats['face_count']:,} 张",
        f"上次建库时间：{last_time}",
        f"索引状态：{index_status} ({index_total} 条向量)",
        f"数据库状态：{db_status}",
    ]
    return stat_lines, "正常" if index_total > 0 else "空"


def select_directory(path):
    if path and os.path.isdir(path):
        config.photo_root = path
    lines, _ = get_stats()
    return "\n".join(lines)


def choose_directory_dialog(current_path):
    selected_path = current_path.strip() if current_path else config.photo_root
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askdirectory(
            initialdir=selected_path if selected_path and os.path.isdir(selected_path) else None,
            title="选择照片目录",
            mustexist=True,
        )
        root.destroy()
    except Exception as e:
        get_app_logger().error(f"打开目录选择框失败: {e}")
        lines, _ = get_stats()
        return current_path or config.photo_root, "\n".join(lines), "打开目录选择框失败，请直接粘贴目录路径。"

    if picked and os.path.isdir(picked):
        config.photo_root = picked
        lines, _ = get_stats()
        return picked, "\n".join(lines), f"已选择目录：{picked}"

    lines, _ = get_stats()
    return current_path or config.photo_root, "\n".join(lines), "未选择新目录。"


def resolve_photo_root(path_input):
    if path_input:
        path_input = path_input.strip()
    if path_input and os.path.isdir(path_input):
        config.photo_root = path_input
        return config.photo_root, None
    if config.photo_root and os.path.isdir(config.photo_root):
        return config.photo_root, None
    return "", "请先设置有效的照片目录。"


def run_full_index(path_input, progress=gr.Progress()):
    photo_root, error = resolve_photo_root(path_input)
    if error:
        lines, _ = get_stats()
        yield "\n".join(lines), error
        return

    progress(0, desc="正在扫描图片...")

    status_text = "正在扫描图片..."

    def update_progress(img_path, face_count, processed=0, total=0, total_faces=0, failed=0):
        nonlocal status_text
        if total > 0:
            pct = processed / total
            progress(pct, desc=f"正在建库: {processed}/{total}")
            status_text = (
                f"正在建库：{processed} / {total}\n"
                f"已检测人脸：{total_faces}\n"
                f"失败图片：{failed}\n"
                f"当前文件：{img_path or '...'}"
            )
        else:
            status_text = f"处理中... 已检测人脸: {total_faces}"

    full_index(photo_root, update_progress)
    progress(1.0, desc="建库完成")
    lines, _ = get_stats()
    yield "\n".join(lines), "建库完成！"


def run_incremental_index(path_input, progress=gr.Progress()):
    photo_root, error = resolve_photo_root(path_input)
    if error:
        lines, _ = get_stats()
        yield "\n".join(lines), error
        return

    progress(0, desc="正在扫描变更...")

    def update_progress(img_path, face_count, processed=0, total=0, total_faces=0, failed=0):
        if total > 0:
            pct = processed / total
            progress(pct, desc=f"增量更新: {processed}/{total}")

    incremental_index(photo_root, update_progress)
    progress(1.0, desc="增量更新完成")
    lines, _ = get_stats()
    yield "\n".join(lines), "增量更新完成！"


def run_rebuild_index(path_input, progress=gr.Progress()):
    photo_root, error = resolve_photo_root(path_input)
    if error:
        lines, _ = get_stats()
        yield "\n".join(lines), error
        return

    progress(0, desc="正在重建...")

    def update_progress(img_path, face_count, processed=0, total=0, total_faces=0, failed=0):
        if total > 0:
            pct = processed / total
            progress(pct, desc=f"重建中: {processed}/{total}")

    rebuild_index(photo_root, update_progress)
    progress(1.0, desc="重建完成")
    lines, _ = get_stats()
    yield "\n".join(lines), "重建完成！"


def on_upload_query(image):
    if image is None:
        return [], None, "", ""

    temp_dir = config.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = str(temp_dir / f"query_{uuid.uuid4().hex}.jpg")

    if isinstance(image, str):
        temp_path = image
    else:
        Image.fromarray(image).save(temp_path, "JPEG", quality=90)

    faces = detect_query_faces(temp_path)
    face_images = []
    face_data = []

    for i, face in enumerate(faces):
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2] if isinstance(image, np.ndarray) else (0, 0)
        if h == 0:
            img_cv = read_image(temp_path)
            if img_cv is not None:
                h, w = img_cv.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if isinstance(image, np.ndarray):
            crop = image[y1:y2, x1:x2]
        else:
            img_cv = read_image(temp_path)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            crop_pil = Image.fromarray(crop)
            face_images.append(crop_pil)
            face_data.append({"bbox": bbox, "det_score": face["det_score"], "index": i})

    return face_images, (face_data if face_data else None), temp_path, ""


def on_face_select(evt: gr.SelectData, face_data_state):
    if face_data_state and evt.index < len(face_data_state):
        return face_data_state[evt.index]
    return None


def run_search(query_path, selected_face, top_k, threshold):
    if not query_path or not os.path.exists(query_path):
        return "请先上传查询图片。", [], "请先完成搜索后再打包。"
    if not selected_face:
        return "请选择要搜索的人脸。", [], "请先完成搜索后再打包。"

    face_idx = selected_face.get("index", 0)
    faces = detect_query_faces(query_path)
    if not faces or face_idx >= len(faces):
        return "未检测到清晰人脸，请上传更清晰的人脸照片。", [], "请先完成搜索后再打包。"

    result = search_by_face(faces[face_idx], top_k=int(top_k), threshold=float(threshold))
    if "error" in result:
        return result["error"], [], "请先完成搜索后再打包。"

    results = result.get("results", [])
    if not results:
        return "未找到匹配结果。", [], "没有可打包的搜索结果。"

    html_parts = [
        f'<div style="margin-bottom:10px;font-size:13px;color:#666;">'
        f'搜索耗时：{result.get("elapsed", 0):.2f}s | 共 {result.get("total_found", 0)} 条结果'
        f'</div>'
    ]

    for r in results:
        crop_html = ""
        if r["crop_path"] and os.path.exists(r["crop_path"]):
            crop_html = (f'<img src="/file={r["crop_path"]}" '
                         f'style="width:80px;height:80px;object-fit:cover;border-radius:6px;">')
        else:
            crop_html = ('<div style="width:80px;height:80px;background:#e8e8e8;'
                         'border-radius:6px;display:flex;align-items:center;'
                         'justify-content:center;font-size:11px;color:#999;">无图</div>')

        exists_note = ""
        if not r["image_exists"]:
            exists_note = '<span style="color:#d32f2f;margin-left:8px;font-size:12px;">原图不存在</span>'

        escaped_path = json.dumps(r["image_path"], ensure_ascii=False)
        bbox_str = ",".join(str(b) for b in r["bbox"])
        view_url = f"/view_image?path={quote(r['image_path'])}&bbox={bbox_str}"

        html_parts.append(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:12px;
                    border:1px solid #e0e0e0;border-radius:8px;margin-bottom:8px;
                    background:#fafafa;">
            <div style="font-weight:600;color:#555;min-width:32px;font-size:14px;">#{r['rank']}</div>
            {crop_html}
            <div style="flex:1;min-width:0;">
                <div style="font-weight:600;font-size:14px;">相似度：{r['similarity']:.4f}</div>
                <div style="font-size:12px;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r['file_name']}</div>
                <div style="font-size:11px;color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r['image_path']}</div>
                <div style="font-size:12px;color:#666;margin-top:2px;">{os.path.basename(r['folder_path'])}</div>
                {exists_note}
            </div>
            <div style="display:flex;gap:4px;flex-wrap:wrap;min-width:180px;">
                <button onclick='fetch("/action/open_image?p="+encodeURIComponent({escaped_path}))
                        .then(r => r.json())
                        .then(data => {{ if (!data.ok) alert(data.error || "打开原图失败"); }})
                        .catch(() => alert("打开原图失败"))'
                        style="padding:5px 10px;font-size:12px;cursor:pointer;
                               border:1px solid #ccc;border-radius:4px;background:#fff;
                               white-space:nowrap;">打开原图</button>
                <button onclick='fetch("/action/open_folder?p="+encodeURIComponent({escaped_path}))
                        .then(r => r.json())
                        .then(data => {{ if (!data.ok) alert(data.error || "打开文件夹失败"); }})
                        .catch(() => alert("打开文件夹失败"))'
                        style="padding:5px 10px;font-size:12px;cursor:pointer;
                               border:1px solid #ccc;border-radius:4px;background:#fff;
                               white-space:nowrap;">打开文件夹</button>
                <button onclick="window.open('{view_url}','_blank','width=1200,height=900')"
                        style="padding:5px 10px;font-size:12px;cursor:pointer;
                               border:1px solid #ccc;border-radius:4px;background:#fff;
                               white-space:nowrap;">查看大图</button>
            </div>
        </div>
        """)

    return "".join(html_parts), results, f"可打包 {len(results)} 条搜索结果。"


def package_search_results(search_results):
    if not search_results:
        return "请先完成一次搜索，再执行打包。"

    export_root = config.data_dir / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    export_dir = export_root / f"search_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    seen_paths = set()
    manifest_lines = []

    for item in search_results:
        image_path = item.get("image_path", "")
        if not image_path or image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        if not os.path.exists(image_path):
            skipped += 1
            continue

        src_path = Path(image_path)
        safe_name = f"{item['rank']:03d}_{src_path.name}"
        target_path = export_dir / safe_name
        dedupe_index = 1
        while target_path.exists():
            target_path = export_dir / f"{item['rank']:03d}_{dedupe_index}_{src_path.name}"
            dedupe_index += 1

        shutil.copy2(src_path, target_path)
        copied += 1
        manifest_lines.append(
            f"{item['rank']}\t{item['similarity']}\t{src_path}\t{target_path.name}"
        )

    manifest_path = export_dir / "manifest.tsv"
    manifest_path.write_text(
        "rank\tsimilarity\tsource_path\texport_name\n" + "\n".join(manifest_lines),
        encoding="utf-8",
    )

    open_file(str(export_dir))
    return (
        f"打包完成：复制 {copied} 张图片到 {export_dir}"
        + (f"；跳过 {skipped} 张不存在的原图。" if skipped else "。")
    )


def create_app():
    with gr.Blocks(title="会议照片以脸搜图工具") as app:
        app.queue(default_concurrency_limit=5)

        stats_md = gr.Markdown("\n".join(get_stats()[0]))
        status_output = gr.Textbox(label="状态", lines=6, value="正在等待操作...")

        with gr.Tabs() as tabs:
            with gr.TabItem("首页", id=0):
                gr.Markdown("# 会议照片以脸搜图工具\n\n### 图库状态")

                dir_input = gr.Textbox(
                    label="当前照片目录",
                    value=config.photo_root,
                    placeholder="D:/公司会议照片 或 /Users/xxx/照片",
                )

                with gr.Row():
                    select_dir_btn = gr.Button("更换目录", variant="secondary")
                    refresh_stats_btn = gr.Button("刷新状态", variant="secondary")

                with gr.Row():
                    index_btn = gr.Button("开始建库", variant="primary", size="lg")
                    incremental_btn = gr.Button("增量更新", variant="secondary")
                    rebuild_btn = gr.Button("重新建库", variant="stop")

                index_btn.click(
                    fn=run_full_index,
                    inputs=[dir_input],
                    outputs=[stats_md, status_output],
                    concurrency_limit=1,
                    api_name="run_full_index",
                )
                incremental_btn.click(
                    fn=run_incremental_index,
                    inputs=[dir_input],
                    outputs=[stats_md, status_output],
                    concurrency_limit=1,
                    api_name="run_incremental_index",
                )
                rebuild_btn.click(
                    fn=run_rebuild_index,
                    inputs=[dir_input],
                    outputs=[stats_md, status_output],
                    concurrency_limit=1,
                    api_name="run_rebuild_index",
                )
                select_dir_btn.click(
                    fn=choose_directory_dialog,
                    inputs=dir_input,
                    outputs=[dir_input, stats_md, status_output],
                    api_name="choose_directory_dialog",
                )
                refresh_stats_btn.click(
                    fn=lambda: "\n".join(get_stats()[0]),
                    outputs=[stats_md],
                    api_name="refresh_stats",
                )

            with gr.TabItem("搜索", id=1):
                gr.Markdown("# 以脸搜图")

                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.Image(
                            label="上传查询图片",
                            type="numpy",
                            height=300,
                        )
                        query_path_state = gr.State("")
                        face_gallery = gr.Gallery(
                            label="检测到的人脸（点击选择后搜索）",
                            object_fit="contain",
                            height=160,
                            columns=4,
                            preview=False,
                        )
                        selected_face_state = gr.State(None)

                    with gr.Column(scale=1):
                        with gr.Row():
                            top_k_input = gr.Number(
                                label="Top K",
                                value=config["index"]["top_k_default"],
                                minimum=1,
                                maximum=500,
                                precision=0,
                            )
                            threshold_input = gr.Slider(
                                label="相似度阈值",
                                minimum=0.0,
                                maximum=1.0,
                                value=config["index"]["similarity_threshold"],
                                step=0.05,
                            )
                        with gr.Row():
                            search_btn = gr.Button("开始搜索", variant="primary", size="lg")
                            package_btn = gr.Button("一键打包结果", variant="secondary")

                        results_html = gr.HTML(label="搜索结果")
                        search_results_state = gr.State([])
                        package_status = gr.Textbox(label="打包状态", lines=2, value="请先完成一次搜索。")

                query_image.change(
                    fn=lambda image: (*on_upload_query(image), [], "请先完成一次搜索。"),
                    inputs=query_image,
                    outputs=[face_gallery, selected_face_state, query_path_state, results_html, search_results_state, package_status],
                    api_name="on_upload_query",
                )

                face_gallery.select(
                    fn=on_face_select,
                    inputs=selected_face_state,
                    outputs=selected_face_state,
                    api_name="on_face_select",
                )

                search_btn.click(
                    fn=run_search,
                    inputs=[query_path_state, selected_face_state, top_k_input, threshold_input],
                    outputs=[results_html, search_results_state, package_status],
                    concurrency_limit=1,
                    api_name="run_search",
                )

                package_btn.click(
                    fn=package_search_results,
                    inputs=[search_results_state],
                    outputs=[package_status],
                    concurrency_limit=1,
                    api_name="package_search_results",
                )

            with gr.TabItem("日志", id=2):
                gr.Markdown("# 日志")
                log_refresh_btn = gr.Button("刷新日志")
                log_output = gr.Textbox(
                    label="最近日志",
                    lines=30,
                    max_lines=50,
                )

                def read_logs():
                    lines = []
                    for name in ["app", "indexing"]:
                        lp = config.logs_dir / f"{name}.log"
                        if lp.exists():
                            with open(lp, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                            if content:
                                last = "\n".join(content.split("\n")[-50:])
                                lines.append(f"=== {name}.log ===\n{last}")
                    return "\n\n".join(lines) if lines else "暂无日志。"

                log_refresh_btn.click(fn=read_logs, outputs=log_output)

        app.load(fn=lambda: "\n".join(get_stats()[0]), outputs=[stats_md])

    return app

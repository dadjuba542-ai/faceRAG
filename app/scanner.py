import os
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from app.config import Config
from app.logger import get_app_logger


@dataclass
class ImageInfo:
    path: str
    file_name: str
    folder_path: str
    file_size: int
    modified_time: float
    file_hash: str = ""


def scan_directory(photo_root: str, supported_extensions: List[str]) -> List[ImageInfo]:
    logger = get_app_logger()
    results = []
    root = Path(photo_root)
    if not root.exists():
        logger.error(f"照片目录不存在: {photo_root}")
        return results

    extensions_lower = [ext.lower() for ext in supported_extensions]

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions_lower:
            continue
        stat = file_path.stat()
        results.append(ImageInfo(
            path=str(file_path.resolve()),
            file_name=file_path.name,
            folder_path=str(file_path.parent.resolve()),
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
        ))
    logger.info(f"扫描完成: 共找到 {len(results)} 张图片")
    return results


def compute_file_hash(file_path: str, chunk_size=8192) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_changed_images(
    current_images: List[ImageInfo],
    db_images: dict,
) -> tuple:
    new_images = []
    changed_images = []
    deleted_paths = []

    db_by_path = {row["image_path"]: row for row in db_images}

    current_paths = set()
    for img in current_images:
        current_paths.add(img.path)
        if img.path not in db_by_path:
            new_images.append(img)
        else:
            db_row = db_by_path[img.path]
            if (img.file_size != db_row["file_size"]
                    or img.modified_time != db_row["modified_time"]):
                changed_images.append(img)

    for db_path in db_by_path:
        if db_path not in current_paths:
            deleted_paths.append(db_path)

    return new_images, changed_images, deleted_paths

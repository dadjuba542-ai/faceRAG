import os
import cv2
import time
from pathlib import Path
from typing import Callable, Optional

from app.config import config
from app.logger import get_app_logger, get_indexing_logger
from app.scanner import scan_directory, get_changed_images, ImageInfo
from app.detector import face_detector
from app.embedding import get_embedding
from app.database import db
from app.indexer import indexer


def process_single_image(image_path: str, progress_callback: Optional[Callable] = None) -> dict:
    logger = get_app_logger()
    idx_logger = get_indexing_logger()
    result = {
        "path": image_path,
        "success": False,
        "faces_count": 0,
        "error": None,
    }

    img = cv2.imread(image_path)
    if img is None:
        result["error"] = "无法读取图片"
        db.insert_failed_image(image_path, "无法读取图片")
        idx_logger.warning(f"无法读取: {image_path}")
        return result

    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_detector.detect(img_rgb)
    if not faces:
        result["success"] = True
        result["faces_count"] = 0
        return result

    stat = os.stat(image_path)
    file_name = os.path.basename(image_path)
    folder_path = os.path.dirname(image_path)

    image_id = db.insert_image(
        image_path=image_path,
        file_name=file_name,
        folder_path=folder_path,
        file_size=stat.st_size,
        modified_time=stat.st_mtime,
        width=width,
        height=height,
    )
    if image_id < 0:
        result["error"] = "数据库插入失败"
        return result

    save_crop = config["thumbnail"]["save_crop"]
    crop_size = config["thumbnail"]["crop_size"]
    jpeg_quality = config["thumbnail"]["jpeg_quality"]

    face_count = 0
    embeddings_list = []
    faiss_ids_list = []

    next_faiss_id = db.get_next_faiss_id()

    for i, face in enumerate(faces):
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        crop_path = ""

        if save_crop:
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size > 0:
                crop_h, crop_w = face_crop.shape[:2]
                if crop_h > 0 and crop_w > 0:
                    scale = crop_size / max(crop_h, crop_w)
                    new_w = max(1, int(crop_w * scale))
                    new_h = max(1, int(crop_h * scale))
                    face_crop_resized = cv2.resize(face_crop, (new_w, new_h),
                                                    interpolation=cv2.INTER_AREA)
                    crop_filename = f"face_{next_faiss_id + face_count:08d}.jpg"
                    crop_path = str(config.face_crops_dir / crop_filename)
                    cv2.imwrite(crop_path, cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        embedding = get_embedding(face)
        embeddings_list.append(embedding)
        faiss_ids_list.append(next_faiss_id + face_count)

        db.insert_face(
            image_id=image_id,
            faiss_id=next_faiss_id + face_count,
            crop_path=crop_path,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            detection_score=face["det_score"],
        )
        face_count += 1

    if face_count > 0:
        import numpy as np
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        ids_array = np.array(faiss_ids_list, dtype=np.int64)
        indexer.add(embeddings_array, ids_array)

    result["success"] = True
    result["faces_count"] = face_count

    if progress_callback:
        progress_callback(image_path, face_count)

    return result


def full_index(photo_root: str, progress_callback: Optional[Callable] = None):
    logger = get_app_logger()
    idx_logger = get_indexing_logger()

    extensions = config["supported_extensions"]
    images = scan_directory(photo_root, extensions)
    total = len(images)
    logger.info(f"开始建库: {photo_root}, 共 {total} 张图片")

    processed = 0
    total_faces = 0
    failed = 0

    for img in images:
        result = process_single_image(img.path, progress_callback)
        processed += 1
        if not result["success"]:
            failed += 1
        total_faces += result["faces_count"]

        if progress_callback:
            progress_callback(None, None, processed, total, total_faces, failed)

    indexer.save()
    db.set_setting("photo_root", photo_root)
    db.set_setting("last_index_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    idx_logger.info(
        f"建库完成: 共处理 {processed} 张, "
        f"检测到 {total_faces} 张人脸, "
        f"失败 {failed} 张"
    )
    logger.info(f"建库完成: {processed}/{total}, 人脸 {total_faces}, 失败 {failed}")


def incremental_index(photo_root: str, progress_callback: Optional[Callable] = None):
    logger = get_app_logger()
    idx_logger = get_indexing_logger()

    extensions = config["supported_extensions"]
    current_images = scan_directory(photo_root, extensions)
    db_images = db.get_all_images()

    new_images, changed_images, deleted_paths = get_changed_images(current_images, db_images)

    logger.info(
        f"增量更新: 新增 {len(new_images)}, "
        f"修改 {len(changed_images)}, "
        f"删除 {len(deleted_paths)}"
    )

    if deleted_paths:
        db.mark_deleted_images(deleted_paths)
        logger.info(f"标记 {len(deleted_paths)} 张缺失图片")

    total_to_process = len(new_images) + len(changed_images)
    processed = 0
    total_faces = 0
    failed = 0

    for img in new_images + changed_images:
        result = process_single_image(img.path, progress_callback)
        processed += 1
        if not result["success"]:
            failed += 1
        total_faces += result["faces_count"]
        if progress_callback:
            progress_callback(None, None, processed, total_to_process, total_faces, failed)

    indexer.save()
    db.set_setting("last_index_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    idx_logger.info(
        f"增量更新完成: 处理 {processed} 张, "
        f"新增人脸 {total_faces}, "
        f"失败 {failed} 张"
    )
    logger.info(f"增量更新完成: {processed} 张, 人脸 {total_faces}, 失败 {failed}")


def rebuild_index(photo_root: str, progress_callback: Optional[Callable] = None):
    db.clear_all()
    indexer.reset()

    crop_dir = config.face_crops_dir
    if crop_dir.exists():
        import shutil
        shutil.rmtree(str(crop_dir))
        crop_dir.mkdir(parents=True, exist_ok=True)

    full_index(photo_root, progress_callback)

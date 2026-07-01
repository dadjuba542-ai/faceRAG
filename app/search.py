import os
import cv2
import time
import numpy as np
from typing import List, Dict, Optional

from app.config import config
from app.detector import face_detector
from app.embedding import get_embedding
from app.database import db
from app.indexer import indexer
from app.logger import get_app_logger


def search_by_face(face_result: dict, top_k: int = 50, threshold: float = 0.35) -> dict:
    logger = get_app_logger()
    start = time.time()

    embedding = get_embedding(face_result)
    distances, indices = indexer.search(embedding, top_k)

    results = []
    for i in range(len(distances)):
        sim = float(distances[i])
        if sim < threshold:
            continue
        faiss_id = int(indices[i])
        if faiss_id < 0:
            continue
        face_data = db.get_face_by_faiss_id(faiss_id)
        if not face_data:
            continue
        image_exists = os.path.exists(face_data["image_path"])
        results.append({
            "rank": len(results) + 1,
            "similarity": round(sim, 4),
            "face_id": face_data["id"],
            "faiss_id": faiss_id,
            "image_path": face_data["image_path"],
            "file_name": face_data["file_name"],
            "folder_path": face_data["folder_path"],
            "crop_path": face_data["crop_path"] or "",
            "bbox": [
                face_data["bbox_x1"],
                face_data["bbox_y1"],
                face_data["bbox_x2"],
                face_data["bbox_y2"],
            ],
            "detection_score": face_data["detection_score"],
            "image_exists": image_exists,
        })

    elapsed = time.time() - start
    logger.info(f"搜索完成: 返回 {len(results)} 条, 耗时 {elapsed:.3f}s")

    return {
        "query_face": {
            "bbox": face_result.get("bbox", []),
            "det_score": face_result.get("det_score", 0),
        },
        "results": results,
        "elapsed": round(elapsed, 3),
        "total_found": len(results),
    }


def detect_query_faces(image_path: str) -> List[dict]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect(img_rgb)
    return faces


def search_by_image(image_path: str, face_index: int = 0,
                    top_k: int = 50, threshold: float = 0.35) -> dict:
    faces = detect_query_faces(image_path)
    if not faces:
        return {"error": "未检测到清晰人脸，请上传更清晰的人脸照片。", "results": []}
    if face_index < 0 or face_index >= len(faces):
        face_index = 0
    return search_by_face(faces[face_index], top_k, threshold)

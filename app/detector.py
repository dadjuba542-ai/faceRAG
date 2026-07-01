import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import insightface
from insightface.app import FaceAnalysis

from app.config import config


class FaceDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        import onnxruntime as ort
        available = [p for p in ort.get_available_providers()]
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in providers if p in available]

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(Path(__file__).resolve().parent.parent / "models"),
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.ctx_id = 0
        self.min_face_size = config["face_detection"]["min_face_size"]
        self.detection_threshold = config["face_detection"]["detection_threshold"]

    def detect(self, image: np.ndarray) -> List[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox[:4]
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue
            if face.det_score < self.detection_threshold:
                continue
            results.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": float(face.det_score),
                "landmark": face.landmark.tolist() if face.landmark is not None else None,
                "embedding": face.normed_embedding,
            })
        return results

    def detect_from_path(self, image_path: str) -> List[dict]:
        img = cv2.imread(image_path)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect(img)


face_detector = FaceDetector()

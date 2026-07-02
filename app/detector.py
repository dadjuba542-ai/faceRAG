import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import insightface
from insightface.app import FaceAnalysis

from app.config import config
from app.image_io import read_image
from app.logger import get_app_logger


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
        self.logger = get_app_logger()
        self.providers = [p for p in ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'] if p in available]
        self.model_root = str(Path(__file__).resolve().parent.parent / "models")
        self.app = self._create_app(self.providers)
        self.cpu_app = None
        self.cpu_det_size = None
        self.ctx_id = 0
        self.current_det_size = None
        self.min_face_size = config["face_detection"]["min_face_size"]
        self.detection_threshold = config["face_detection"]["detection_threshold"]
        retry_cfg = config["face_detection"].get("aggressive_retry", {})
        self.aggressive_retry_enabled = retry_cfg.get("enabled", True)
        self.aggressive_min_face_size = retry_cfg.get("min_face_size", 32)
        self.aggressive_detection_threshold = retry_cfg.get("detection_threshold", 0.3)
        self.aggressive_det_size = retry_cfg.get("det_size", 1024)
        self.aggressive_run_on_no_face_only = retry_cfg.get("run_on_no_face_only", True)
        self._set_det_size(640)

    def _create_app(self, providers: List[str]) -> FaceAnalysis:
        return FaceAnalysis(
            name="buffalo_l",
            root=self.model_root,
            providers=providers,
        )

    def _get_cpu_app(self) -> FaceAnalysis:
        if self.cpu_app is None:
            self.cpu_app = self._create_app(['CPUExecutionProvider'])
            self.cpu_det_size = None
        return self.cpu_app

    def _set_det_size(self, det_size: int):
        if self.current_det_size == det_size:
            return
        self.app.prepare(ctx_id=self.ctx_id, det_size=(det_size, det_size))
        self.current_det_size = det_size

    def _set_cpu_det_size(self, det_size: int):
        cpu_app = self._get_cpu_app()
        if self.cpu_det_size == det_size:
            return
        cpu_app.prepare(ctx_id=self.ctx_id, det_size=(det_size, det_size))
        self.cpu_det_size = det_size

    def _format_faces(self, faces, min_face_size: int, detection_threshold: float) -> List[dict]:
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox[:4]
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < min_face_size or face_h < min_face_size:
                continue
            if face.det_score < detection_threshold:
                continue
            results.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": float(face.det_score),
                "landmark": face.landmark.tolist() if face.landmark is not None else None,
                "embedding": face.normed_embedding,
            })
        return results

    def _bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _merge_faces(self, base_faces: List[dict], extra_faces: List[dict]) -> List[dict]:
        merged = list(base_faces)
        for extra_face in extra_faces:
            duplicated = False
            for base_face in merged:
                if self._bbox_iou(base_face["bbox"], extra_face["bbox"]) >= 0.5:
                    duplicated = True
                    if extra_face["det_score"] > base_face["det_score"]:
                        base_face.update(extra_face)
                    break
            if not duplicated:
                merged.append(extra_face)
        merged.sort(key=lambda item: item["det_score"], reverse=True)
        return merged

    def _detect_once(
        self,
        image: np.ndarray,
        *,
        det_size: int,
        min_face_size: int,
        detection_threshold: float,
        use_cpu_only: bool = False,
    ) -> List[dict]:
        detector_app = self.app
        provider_desc = self.providers
        if use_cpu_only:
            self._set_cpu_det_size(det_size)
            detector_app = self._get_cpu_app()
            provider_desc = ['CPUExecutionProvider']
        else:
            self._set_det_size(det_size)
        try:
            faces = detector_app.get(image)
        except Exception as e:
            self.logger.warning(f"人脸检测失败，det_size={det_size}，providers={provider_desc}: {e}")
            if use_cpu_only or self.providers == ['CPUExecutionProvider']:
                raise
            self.logger.warning("首检推理失败，当前图片自动改用 CPUExecutionProvider 重试")
            self._set_cpu_det_size(det_size)
            faces = self._get_cpu_app().get(image)
        return self._format_faces(faces, min_face_size, detection_threshold)

    def detect(self, image: np.ndarray) -> List[dict]:
        conservative_faces = self._detect_once(
            image,
            det_size=640,
            min_face_size=self.min_face_size,
            detection_threshold=self.detection_threshold,
        )
        if not self.aggressive_retry_enabled:
            return conservative_faces

        should_retry = not conservative_faces or not self.aggressive_run_on_no_face_only
        if not should_retry:
            return conservative_faces

        aggressive_faces = self._detect_once(
            image,
            det_size=self.aggressive_det_size,
            min_face_size=self.aggressive_min_face_size,
            detection_threshold=self.aggressive_detection_threshold,
            use_cpu_only=True,
        )
        return self._merge_faces(conservative_faces, aggressive_faces)

    def detect_from_path(self, image_path: str) -> List[dict]:
        img = read_image(image_path)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect(img)


face_detector = FaceDetector()

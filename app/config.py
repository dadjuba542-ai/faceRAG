import os
import yaml
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.yaml"


DEFAULT_CONFIG = {
    "photo_root": "",
    "index": {
        "top_k_default": 50,
        "similarity_threshold": 0.35,
        "embedding_dim": 512,
        "faiss_index_path": str(APP_DIR / "data" / "faiss.index"),
    },
    "face_detection": {
        "min_face_size": 60,
        "detection_threshold": 0.5,
        "enable_quality_filter": True,
    },
    "thumbnail": {
        "save_crop": True,
        "crop_size": 160,
        "jpeg_quality": 80,
    },
    "runtime": {
        "host": "127.0.0.1",
        "port": 7860,
        "auto_open_browser": True,
    },
    "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
}


class Config:
    def __init__(self):
        self.path = CONFIG_PATH
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return self._merge_defaults(yaml.safe_load(f) or {})
        return dict(DEFAULT_CONFIG)

    def _merge_defaults(self, cfg):
        merged = dict(DEFAULT_CONFIG)
        for k, v in cfg.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, allow_unicode=True, indent=2)

    @property
    def photo_root(self):
        return self.data.get("photo_root", "")

    @photo_root.setter
    def photo_root(self, value):
        self.data["photo_root"] = value
        self.save()

    @property
    def data_dir(self):
        return APP_DIR / "data"

    @property
    def face_crops_dir(self):
        return self.data_dir / "face_crops"

    @property
    def temp_dir(self):
        return self.data_dir / "temp"

    @property
    def logs_dir(self):
        return self.data_dir / "logs"

    @property
    def db_path(self):
        return self.data_dir / "faces.db"

    @property
    def faiss_index_path(self):
        return Path(self.data["index"]["faiss_index_path"])

    def ensure_dirs(self):
        for d in [self.data_dir, self.face_crops_dir, self.temp_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        return self.data.get(key, DEFAULT_CONFIG.get(key))

    def get(self, key, default=None):
        return self.data.get(key, default)


config = Config()

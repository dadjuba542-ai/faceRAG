import sqlite3
import time
import os
from pathlib import Path
from typing import List, Optional, Dict

from app.config import Config, config
from app.logger import get_app_logger


class Database:
    def __init__(self, config: Config):
        self.db_path = config.db_path
        self.logger = get_app_logger()
        self._init_tables()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self):
        conn = self._conn()
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL UNIQUE,
                file_name TEXT,
                folder_path TEXT,
                file_size INTEGER,
                modified_time REAL,
                file_hash TEXT,
                width INTEGER,
                height INTEGER,
                indexed_at TEXT,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                faiss_id INTEGER NOT NULL UNIQUE,
                crop_path TEXT,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                detection_score REAL,
                quality_score REAL,
                created_at TEXT,
                FOREIGN KEY(image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS failed_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                error_message TEXT,
                failed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_faces_faiss_id ON faces(faiss_id);
            CREATE INDEX IF NOT EXISTS idx_images_path ON images(image_path);
        """)
        conn.commit()
        conn.close()

    def insert_image(self, image_path: str, file_name: str, folder_path: str,
                     file_size: int, modified_time: float, file_hash: str = "",
                     width: int = 0, height: int = 0) -> int:
        conn = self._conn()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images
                (image_path, file_name, folder_path, file_size, modified_time,
                 file_hash, width, height, indexed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, (image_path, file_name, folder_path, file_size, modified_time,
                  file_hash, width, height, now))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"插入图片记录失败 {image_path}: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()

    def update_image_status(self, image_path: str, status: str):
        conn = self._conn()
        conn.execute("UPDATE images SET status=? WHERE image_path=?", (status, image_path))
        conn.commit()
        conn.close()

    def get_all_images(self) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM images WHERE status='active'").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_images_under_root(self, root_path: str) -> List[Dict]:
        root_abs = os.path.abspath(root_path)
        scoped_images = []
        for row in self.get_all_images():
            image_path = os.path.abspath(row["image_path"])
            try:
                if os.path.commonpath([root_abs, image_path]) == root_abs:
                    scoped_images.append(row)
            except ValueError:
                continue
        return scoped_images

    def get_image_by_path(self, image_path: str) -> Optional[Dict]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM images WHERE image_path=?", (image_path,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_next_faiss_id(self) -> int:
        conn = self._conn()
        row = conn.execute("SELECT COALESCE(MAX(faiss_id), -1) + 1 as nid FROM faces").fetchone()
        conn.close()
        return row["nid"]

    def insert_face(self, image_id: int, faiss_id: int, crop_path: str,
                    bbox_x1: int, bbox_y1: int, bbox_x2: int, bbox_y2: int,
                    detection_score: float) -> int:
        conn = self._conn()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO faces
                (image_id, faiss_id, crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 detection_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, faiss_id, crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                  detection_score, now))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"插入人脸记录失败 faiss_id={faiss_id}: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()

    def get_face_by_faiss_id(self, faiss_id: int) -> Optional[Dict]:
        conn = self._conn()
        row = conn.execute("""
            SELECT f.*, i.image_path, i.file_name, i.folder_path, i.modified_time, i.status
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE f.faiss_id = ?
        """, (faiss_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_faces_by_image_path(self, image_path: str) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT f.* FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE i.image_path = ?
        """, (image_path,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_face_ids_by_image_path(self, image_path: str) -> List[int]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT f.faiss_id
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE i.image_path = ?
        """, (image_path,)).fetchall()
        conn.close()
        return [int(r["faiss_id"]) for r in rows]

    def get_all_face_ids(self) -> List[int]:
        conn = self._conn()
        rows = conn.execute("SELECT faiss_id FROM faces ORDER BY faiss_id").fetchall()
        conn.close()
        return [int(r["faiss_id"]) for r in rows]

    def get_orphan_face_ids(self) -> List[int]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT f.faiss_id
            FROM faces f
            LEFT JOIN images i ON f.image_id = i.id
            WHERE i.id IS NULL
            ORDER BY f.faiss_id
        """).fetchall()
        conn.close()
        return [int(r["faiss_id"]) for r in rows]

    def get_stats(self) -> Dict:
        conn = self._conn()
        image_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM images WHERE status='active'"
        ).fetchone()["cnt"]
        face_count = conn.execute("SELECT COUNT(*) as cnt FROM faces").fetchone()["cnt"]
        failed_count = conn.execute("SELECT COUNT(*) as cnt FROM failed_images").fetchone()["cnt"]
        last_time = conn.execute(
            "SELECT MAX(indexed_at) as t FROM images"
        ).fetchone()["t"] or ""
        conn.close()
        return {
            "image_count": image_count,
            "face_count": face_count,
            "failed_count": failed_count,
            "last_index_time": last_time,
        }

    def insert_failed_image(self, image_path: str, error_message: str):
        conn = self._conn()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO failed_images (image_path, error_message, failed_at) VALUES (?, ?, ?)",
            (image_path, error_message, now),
        )
        conn.commit()
        conn.close()

    def get_all_failed_images(self) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM failed_images ORDER BY failed_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_setting(self, key: str, default: str = "") -> str:
        conn = self._conn()
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        conn = self._conn()
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()

    def clear_all(self):
        conn = self._conn()
        conn.execute("DELETE FROM faces")
        conn.execute("DELETE FROM images")
        conn.execute("DELETE FROM failed_images")
        conn.commit()
        conn.close()

    def delete_image_and_faces(self, image_path: str):
        conn = self._conn()
        try:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT id FROM images WHERE image_path=?",
                (image_path,),
            ).fetchone()
            if not row:
                return
            image_id = row["id"]
            cursor.execute("DELETE FROM faces WHERE image_id=?", (image_id,))
            cursor.execute("DELETE FROM images WHERE id=?", (image_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_faces_by_faiss_ids(self, faiss_ids: List[int]):
        if not faiss_ids:
            return
        conn = self._conn()
        try:
            conn.executemany(
                "DELETE FROM faces WHERE faiss_id=?",
                [(int(faiss_id),) for faiss_id in faiss_ids],
            )
            conn.commit()
        finally:
            conn.close()

    def mark_deleted_images(self, deleted_paths: List[str]):
        conn = self._conn()
        for path in deleted_paths:
            conn.execute("UPDATE images SET status='missing' WHERE image_path=?", (path,))
        conn.commit()
        conn.close()


db = Database(config)

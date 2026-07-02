import os

import cv2
import numpy as np


def read_image(image_path: str, flags: int = cv2.IMREAD_COLOR):
    if os.name == "nt":
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    return cv2.imread(image_path, flags)

"""
图像预处理模块：对拍摄的作业图像进行自动校正和增强。
- 透视校正：检测纸张边缘 → 四点透视变换拉正
- 增强：自适应对比度 + 锐化
"""
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from config import PREPROCESS_CONTRAST_ALPHA, PREPROCESS_SHARPEN_STRENGTH


class ImagePreprocessor:
    """作业图像预处理流水线。"""

    def __init__(self, contrast_alpha: float = PREPROCESS_CONTRAST_ALPHA,
                 sharpen_strength: float = PREPROCESS_SHARPEN_STRENGTH):
        self.contrast_alpha = contrast_alpha
        self.sharpen_strength = sharpen_strength

    @staticmethod
    def bytes_to_ndarray(data: bytes) -> "np.ndarray":
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像数据")
        return img

    def process(self, image: np.ndarray, deskew: bool = True,
                enhance: bool = True) -> np.ndarray:
        if deskew:
            image = self.deskew(image)
        if enhance:
            image = self.enhance(image)
        return image

    # ── 校正 ──

    def deskew(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("未检测到边缘，返回原图")
            return image

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) != 4:
            rect = cv2.minAreaRect(largest)
            approx = cv2.boxPoints(rect)
            approx = np.intp(approx)

        return self._four_point_transform(image, approx.reshape(4, 2))

    # ── 增强 ──

    def enhance(self, image: np.ndarray) -> np.ndarray:
        enhanced = cv2.convertScaleAbs(image, alpha=self.contrast_alpha, beta=0)
        if self.sharpen_strength > 0:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]) * self.sharpen_strength
            kernel[1, 1] = 5 * self.sharpen_strength + (1 - self.sharpen_strength)
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced

    # ── 透视变换 ──

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))


# ── 便捷函数 ──

def imread_safe(filepath: str | Path) -> np.ndarray:
    """安全读取图像（兼容中文路径）"""
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {filepath}")
    return img

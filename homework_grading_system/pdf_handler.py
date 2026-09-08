"""
PDF 处理模块 — 将 PDF 文件转换为图像列表
使用 PyMuPDF (fitz) 实现，Windows 无需额外依赖
"""
import cv2
import fitz
import numpy as np
from loguru import logger


class PDFHandler:
    """PDF 转图像转换器"""

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def convert(self, pdf_bytes: bytes, max_pages: int = 0) -> list[np.ndarray]:
        """将 PDF 字节流转换为 numpy 图像数组列表

        Args:
            pdf_bytes: PDF 文件字节流
            max_pages: 最大页数限制，0=全部

        Returns:
            list[np.ndarray]: BGR 格式的图像数组（每页一张）
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)
        pages = min(total, max_pages) if max_pages > 0 else total
        logger.info(f"PDF 共 {total} 页，处理前 {pages} 页 (DPI={self.dpi})")

        images = []
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            # RGB → BGR (OpenCV 格式)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA → BGR
                img = img[:, :, :3]
            elif pix.n == 3:  # RGB → BGR
                img = img[:, :, ::-1]
            elif pix.n == 1:  # Gray → BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img)

        doc.close()
        logger.debug(f"PDF 转换完成: {len(images)} 张图像")
        return images

    @staticmethod
    def is_pdf(filename: str) -> bool:
        return filename.lower().endswith('.pdf')

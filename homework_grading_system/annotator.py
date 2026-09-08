"""
批注标注模块 (v4.5)。
全图总览：仅顶部总分栏（无题框）
单题批注：绿色/红色边框 + ✓/✗ 标记 + 得分 + 分析
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from grader import FullGradingResult, GradingResult
from ocr_api import QuestionRegion


# ── 颜色常量 (BGR) ──
GREEN = (0, 180, 80)
RED = (0, 40, 220)
BLUE = (220, 120, 40)
ORANGE = (0, 140, 240)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
BG_OVERLAY = (255, 255, 240)


class ResultAnnotator:
    """在作业图像上标注批改结果。"""

    def __init__(self, font_size: int = 40, line_thickness: int = 4):
        self.font_size = font_size
        self.line_thickness = line_thickness
        self._font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]

    # ── 主入口 ──

    def annotate(self, image: np.ndarray,
                 grading: FullGradingResult,
                 regions: list[QuestionRegion] = None) -> np.ndarray:
        """在图像上绘制完整的批改标注（v4.5：全图总览不画题框，仅保留顶部注释）。"""
        result_img = image.copy()

        # 顶部总分栏
        self._draw_score_header(result_img, grading)

        return result_img

    def annotate_single_question(self, question_img: np.ndarray,
                                   result: GradingResult,
                                   question_num: int = 1) -> np.ndarray:
        """为单道题的裁剪图像添加批注，返回独立批注图。

        样式：顶部状态栏(题号+对错+得分) + 底部分析栏 + 彩色边框。
        """
        h, w = question_img.shape[:2]
        header_h = 60
        footer_h = 70
        border = 8

        # 创建带边距的画布
        new_h = header_h + h + footer_h + border * 2
        new_w = w + border * 2
        canvas = np.full((new_h, new_w, 3), 250, dtype=np.uint8)  # 浅灰背景

        # 贴上原图
        canvas[header_h + border:header_h + border + h,
               border:border + w] = question_img

        # 颜色
        is_correct = result.is_correct
        main_color = GREEN if is_correct else RED

        # ── 彩色边框 ──
        cv2.rectangle(canvas, (2, 2), (new_w - 2, new_h - 2),
                      main_color, border)

        # ── 顶部状态栏 ──
        cv2.rectangle(canvas, (border, border),
                      (new_w - border, header_h + border),
                      (40, 40, 40), -1)

        # 对错标记 + 题号
        mark = "✓" if is_correct else "✗"
        self._draw_text_absolute(canvas, (20, 14),
                                 f"{question_num}  {mark}",
                                 GREEN if is_correct else (255, 80, 80),
                                 self.font_size + 6)

        # 得分
        score_text = f"{result.score:.1f} / {result.max_score:.1f} 分"
        self._draw_text_absolute(canvas, (new_w - 320, 14),
                                 score_text, WHITE, self.font_size)

        # ── 底部分析栏 ──
        cv2.rectangle(canvas, (border, new_h - footer_h - border),
                      (new_w - border, new_h - border),
                      (245, 245, 238), -1)
        cv2.line(canvas,
                 (border, new_h - footer_h - border),
                 (new_w - border, new_h - footer_h - border),
                 (200, 200, 200), 1)

        y_text = new_h - footer_h - border + 8
        if is_correct:
            msg = result.suggestion or "回答正确！"
            self._draw_text_absolute(canvas, (20, y_text),
                                     f"✓  {msg}", GREEN, self.font_size - 2)
        else:
            # 标准答案
            if result.reference_answer:
                ref = result.reference_answer[:60]
                self._draw_text_absolute(canvas, (20, y_text),
                                         f"答案: {ref}",
                                         BLUE, self.font_size - 4)
                y_text += 26
            # 错因
            if result.analysis:
                analysis = result.analysis[:80]
                self._draw_text_absolute(canvas, (20, y_text),
                                         f"分析: {analysis}",
                                         ORANGE, self.font_size - 4)

        return canvas

    def _draw_text_absolute(self, img: np.ndarray, pos: tuple[int, int],
                             text: str, color: tuple[int, int, int],
                             size: int):
        """在 canvas 绝对坐标上绘制文字。"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_pil_font(size)
        draw.text(pos, text, fill=color, font=font)
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── 绘制方法 ──

    def _draw_question_box(self, img: np.ndarray,
                           region: QuestionRegion, result: GradingResult):
        """绘制题目边框、题号和对错标记。"""
        x1, y1, x2, y2 = [int(v) for v in region.bbox]
        color = GREEN if result.is_correct else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.line_thickness + 1)

        # 题号圆形标记
        q_num = result.question_index + 1
        cx, cy = x1 + 22, y1 + 22
        cv2.circle(img, (cx, cy), 16, color, -1)
        num_text = str(q_num)
        (tw, th), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, num_text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

        # 对错标记
        mark = "✓" if result.is_correct else "✗"
        mark_color = GREEN if result.is_correct else RED
        self._draw_text_line(img, (x1 + 48, y1 + 30), mark, mark_color,
                             self.font_size + 10, bold=True)

    def _draw_question_score(self, img: np.ndarray,
                             region: QuestionRegion, result: GradingResult):
        """在题目旁边标注得分。"""
        x1, y1, x2, y2 = [int(v) for v in region.bbox]
        score_text = f"{result.score:.1f}/{result.max_score:.1f}"
        color = GREEN if result.is_correct else RED
        self._draw_text_line(img, (x2 - 180, y1 + 10), score_text, color,
                             self.font_size)

        # 不绘制分析文字，仅用题号标识

    def _draw_score_header(self, img: np.ndarray, grading: FullGradingResult):
        """绘制顶部总分栏。"""
        h, w = img.shape[:2]
        header_h = 110

        # 背景条
        cv2.rectangle(img, (0, 0), (w, header_h), BLACK, -1)
        cv2.rectangle(img, (0, header_h - 2), (w, header_h), WHITE, 1)

        # 总分
        score_text = f"Total: {grading.total_score:.1f} / {grading.max_total_score:.0f}"
        self._draw_text_line(img, (20, 20), score_text, WHITE, self.font_size + 6, bold=True)

        # 正确率
        acc = grading.accuracy_rate()
        acc_color = GREEN if acc >= 0.6 else RED
        acc_text = f"Accuracy: {acc*100:.1f}%"
        self._draw_text_line(img, (20, 56), acc_text, acc_color, self.font_size)

        # 日期
        from datetime import datetime
        date_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._draw_text_line(img, (w - 260, 20), date_text, WHITE, self.font_size - 4)

    def _draw_text_line(self, img: np.ndarray, pos: tuple[int, int],
                        text: str, color: tuple[int, int, int],
                        size: int = None, bold: bool = False):
        """使用 PIL 绘制中文文本到 OpenCV 图像。"""
        size = size or self.font_size
        # 转换为 PIL 图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_pil_font(size)

        draw.text(pos, text, fill=color, font=font)
        # 写回 OpenCV 格式
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_pil_font(self, size: int) -> ImageFont.FreeTypeFont:
        """获取可用的中文字体。"""
        for fp in self._font_paths:
            if Path(fp).exists():
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        # fallback: 默认字体（可能不支持中文）
        return ImageFont.load_default()

    @staticmethod
    def _find_region(idx: int, regions: list[QuestionRegion]) -> QuestionRegion | None:
        """根据 index 找到对应的区域。"""
        if not regions:
            return None
        for r in regions:
            if r.index == idx:
                return r
        return regions[idx] if idx < len(regions) else None

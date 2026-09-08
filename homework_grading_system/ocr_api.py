"""
OCR 识别模块 — 两种真实 OCR 后端
  - AliyunOCRClient : 阿里云 OCR SDK（教育切题 + 教育识别）
  - MiMoVisionClient: MiMo 多模态大模型视觉识别
"""
import asyncio
import base64
import json
import re

import cv2
import numpy as np
from openai import AsyncOpenAI
from loguru import logger

from config import (
    LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
    ALIYUN_ACCESS_KEY, ALIYUN_ACCESS_SECRET, ALIYUN_OCR_ENDPOINT,
    MAX_CONCURRENT_OCR, REQUEST_TIMEOUT,
    RETRY_MAX, RETRY_BACKOFF,
    MIMO_DIRECT_PROMPT, DEFAULT_TOTAL_SCORE,
)

# ── 数据结构 ──

class QuestionRegion:
    """题目区域"""
    def __init__(self, index, bbox=(0, 0, 0, 0)):
        self.index = index
        self.bbox = list(bbox)

    @property
    def has_valid_bbox(self):
        return self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]

    def __repr__(self):
        return f"QRegion({self.index}, bbox={self.bbox})"


class OCRResult:
    """OCR 识别结果"""
    def __init__(self, text="", confidence=0.0, question_index=0,
                 stem="", options="", answer=""):
        self.text = text
        self.confidence = confidence
        self.question_index = question_index
        self.stem = stem
        self.options = options
        self.answer = answer

    @property
    def structured_text(self):
        if self.stem or self.options or self.answer:
            parts = []
            if self.stem: parts.append(f"【题目】{self.stem}")
            if self.options: parts.append(f"【选项】{self.options}")
            if self.answer: parts.append(f"【学生作答】{self.answer}")
            return "\n".join(parts)
        return self.text

    def __repr__(self):
        return f"OCR(q{self.question_index}): {(self.stem or self.text)[:40]}..."


# ── 提示词 ──

FULL_PIPELINE_PROMPT = """请仔细分析这张试卷图片，完成以下任务：
1. 识别试卷中每道题的位置和内容
2. 区分题干、选项（如有）和学生手写作答
3. 按题目顺序返回结构化结果

【坐标说明】用百分比表示题目在图片中的位置（0=最左/最上, 100=最右/最下）：
- x: 题目区域左上角距图片左边界的百分比（0-100）
- y: 题目区域左上角距图片上边界的百分比（0-100）
- width: 题目区域占图片宽度的百分比（0-100）
- height: 题目区域占图片高度的百分比（0-100）
例如：一道题在图片上半部分中间，x=10, y=20, width=80, height=15

严格按以下JSON格式返回（只返回JSON）：
{
  "questions": [
    {"index":0,"bbox":{"x":10,"y":20,"width":80,"height":15},"stem":"题干","options":"选项(无则空)","answer":"学生作答","full_text":"完整文本"}
  ]
}"""


# ── 工具函数 ──

def _resize_b64(image_base64, max_dim=2048):
    """等比缩小长边，返回 base64 字符串"""
    raw = base64.b64decode(image_base64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_base64
    h, w = img.shape[:2]
    max_side = max(w, h)
    if max_side <= max_dim:
        return image_base64
    scale = max_dim / max_side
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.debug(f"图片压缩: {w}x{h} → {int(w*scale)}x{int(h*scale)}")
    return base64.b64encode(buf).decode()


def parse_json(raw):
    """安全解析 JSON，失败时返回空 dict"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    logger.warning(f"JSON解析失败: {raw[:100]}")
    return {}


# ═══════════════════════════════════════════════════════════════
#  阿里云 OCR 客户端（真实 SDK：教育切题 + 教育识别）
# ═══════════════════════════════════════════════════════════════

class AliyunOCRClient:
    """阿里云 OCR — 使用官方 SDK 调用教育切题 + 教育识别 API"""

    MAX_IMAGE_DIM = 2048

    def __init__(self, access_key=None, access_secret=None, endpoint=None):
        self.access_key = access_key or ALIYUN_ACCESS_KEY
        self.access_secret = access_secret or ALIYUN_ACCESS_SECRET
        self.endpoint = endpoint or ALIYUN_OCR_ENDPOINT
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)
        self._available = bool(self.access_key and self.access_secret)
        if not self._available:
            logger.warning("阿里云 OCR 凭证未配置，将降级为 MiMo 视觉识别")

    async def full_pipeline(self, image_base64=""):
        """阿里云完整流水线：切题 → 逐题 OCR → 返回结果"""
        if not self._available:
            # 降级：用 MiMo 视觉做 OCR（对调用方透明）
            logger.info("阿里云凭证缺失，降级使用 MiMo 视觉 OCR")
            client = MiMoVisionClient()
            return await client.full_pipeline(image_base64)

        image_base64 = _resize_b64(image_base64, self.MAX_IMAGE_DIM)

        try:
            # 1. 教育切题
            regions = await self._paper_cut(image_base64)
            if not regions or len(regions) <= 1:
                logger.warning(f"阿里云切题仅返回 {len(regions) if regions else 0} 个区域，使用整图识别")
                return await self._fallback_ocr(image_base64)

            # 2. 逐题 OCR
            tasks = [self._question_ocr(image_base64, r) for r in regions]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for i, r in enumerate(raw):
                if isinstance(r, Exception):
                    logger.warning(f"阿里云OCR Q{i} 失败: {r}")
                    results.append(OCRResult("", 0.0, i))
                else:
                    results.append(r)

            logger.info(f"阿里云OCR完成: {len(results)}题")
            return regions, results

        except Exception as e:
            logger.error(f"阿里云OCR流水线失败: {e}，降级使用 MiMo 视觉")
            client = MiMoVisionClient()
            return await client.full_pipeline(image_base64)

    async def _paper_cut(self, image_base64):
        """调用阿里云教育切题 API"""
        from alibabacloud_ocr_api20210707.client import Client as OcrClient
        from alibabacloud_ocr_api20210707 import models as ocr_models
        from alibabacloud_tea_openapi.models import Config
        from alibabacloud_tea_util.models import RuntimeOptions

        config = Config(
            access_key_id=self.access_key,
            access_key_secret=self.access_secret,
            endpoint=self.endpoint,
        )
        client = OcrClient(config)
        body_stream = base64.b64decode(image_base64)

        async with self._semaphore:
            req = ocr_models.RecognizeEduPaperCutRequest(
                image_type="photo",
                body=body_stream,
            )
            runtime = RuntimeOptions()
            resp = await asyncio.to_thread(
                client.recognize_edu_paper_cut_with_options, req, runtime
            )
            data = parse_json(resp.body.data)
            # 阿里云 OCR SDK 返回的 Data 是 JSON 字符串
            if isinstance(data, str):
                data = parse_json(data)

        page_list = data.get("page_list", [data]) if isinstance(data, dict) else [{}]
        regions = []
        for page in page_list:
            subjects = page.get("subject_list", [])
            for subj in subjects:
                bbox = subj.get("brace", {}).get("brace", {})
                x1 = bbox.get("x", 0)
                y1 = bbox.get("y", 0)
                x2 = x1 + bbox.get("width", 100)
                y2 = y1 + bbox.get("height", 100)
                regions.append(QuestionRegion(len(regions), bbox=[x1, y1, x2, y2]))

        logger.info(f"阿里云切题: {len(regions)} 个区域")
        return regions

    async def _question_ocr(self, image_base64, region: QuestionRegion):
        """调用阿里云教育识别 API"""
        from alibabacloud_ocr_api20210707.client import Client as OcrClient
        from alibabacloud_ocr_api20210707 import models as ocr_models
        from alibabacloud_tea_openapi.models import Config
        from alibabacloud_tea_util.models import RuntimeOptions

        config = Config(
            access_key_id=self.access_key,
            access_key_secret=self.access_secret,
            endpoint=self.endpoint,
        )
        client = OcrClient(config)

        # 裁剪题目区域
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        if region.has_valid_bbox:
            x1, y1, x2, y2 = [max(0, min(int(v), d)) for v, d in
                              zip(region.bbox, [w, h, w, h])]
            crop = img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else img
        else:
            crop = img

        _, buf = cv2.imencode(".png", crop)
        body_stream = buf.tobytes()

        async with self._semaphore:
            req = ocr_models.RecognizeEduQuestionOcrRequest(body=body_stream)
            runtime = RuntimeOptions()
            resp = await asyncio.to_thread(
                client.recognize_edu_question_ocr_with_options, req, runtime
            )
            data = parse_json(resp.body.data)
            if isinstance(data, str):
                data = parse_json(data)

        # 解析教育 OCR 结果
        content = data.get("content", "") if isinstance(data, dict) else ""
        figure_list = data.get("figure", []) if isinstance(data, dict) else []

        stem = content if isinstance(content, str) else ""
        answer = ""
        if isinstance(figure_list, list):
            for fig in figure_list:
                if isinstance(fig, dict) and fig.get("type") == "handwriting":
                    answer = fig.get("content", "")

        return OCRResult(
            text=f"{stem}\n{answer}".strip(),
            confidence=0.85,
            question_index=region.index,
            stem=stem,
            options="",
            answer=answer,
        )

    async def _fallback_ocr(self, image_base64):
        """回退：整图通用文字识别"""
        from alibabacloud_ocr_api20210707.client import Client as OcrClient
        from alibabacloud_ocr_api20210707 import models as ocr_models
        from alibabacloud_tea_openapi.models import Config
        from alibabacloud_tea_util.models import RuntimeOptions

        config = Config(
            access_key_id=self.access_key,
            access_key_secret=self.access_secret,
            endpoint=self.endpoint,
        )
        client = OcrClient(config)
        body_stream = base64.b64decode(image_base64)

        async with self._semaphore:
            req = ocr_models.RecognizeAllTextRequest(
                type="General",
                body=body_stream,
            )
            runtime = RuntimeOptions()
            resp = await asyncio.to_thread(
                client.recognize_all_text_with_options, req, runtime
            )
            data = parse_json(resp.body.data)
            if isinstance(data, str):
                data = parse_json(data)

        content = data.get("content", "") if isinstance(data, dict) else ""
        # 粗略分题：按换行分组
        lines = [l.strip() for l in str(content).split("\n") if l.strip()]
        results = []
        regions = []
        for i, line in enumerate(lines):
            regions.append(QuestionRegion(i))
            results.append(OCRResult(text=line, confidence=0.8, question_index=i))
        logger.info(f"阿里云通用OCR: {len(results)} 行文字")
        return regions, results


# ═══════════════════════════════════════════════════════════════
#  MiMo 多模态大模型客户端（视觉识别 + 端到端批改）
# ═══════════════════════════════════════════════════════════════

class MiMoVisionClient:
    """MiMo 多模态大模型 — 利用视觉能力"看"图识字 + 端到端批改"""

    MAX_IMAGE_DIM = 2048

    def __init__(self, api_key=None, base_url=None, model=None):
        self.client = AsyncOpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=base_url or LLM_API_BASE,
            timeout=REQUEST_TIMEOUT,
        )
        self.model = model or LLM_MODEL
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)

    async def full_pipeline(self, image_base64=""):
        """MiMo 视觉 OCR：整图发给大模型，分题 + 识别一次完成"""
        image_base64 = _resize_b64(image_base64, self.MAX_IMAGE_DIM)

        # 解码获取 MiMo 实际看到的图像尺寸（用于百分比→像素换算）
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ref_h, ref_w = ref_img.shape[:2]

        for attempt in range(RETRY_MAX):
            try:
                async with self._semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": FULL_PIPELINE_PROMPT},
                        ]}],
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                data = parse_json(resp.choices[0].message.content)
                questions = data.get("questions", [])

                regions, ocr_results = [], []
                for q in (questions if questions else [data]):
                    idx = q.get("index", len(ocr_results))
                    b = q.get("bbox", {})
                    # 百分比 → 像素（基于 MiMo 实际看到的图像尺寸）
                    bx = int(b.get("x", 0) / 100 * ref_w)
                    by = int(b.get("y", 0) / 100 * ref_h)
                    bw = int(b.get("width", 0) / 100 * ref_w)
                    bh = int(b.get("height", 0) / 100 * ref_h)
                    regions.append(QuestionRegion(idx, bbox=[bx, by, bx + bw, by + bh]))
                    ocr_results.append(OCRResult(
                        text=q.get("full_text", ""),
                        confidence=0.9,
                        question_index=idx,
                        stem=q.get("stem", ""),
                        options=q.get("options", ""),
                        answer=q.get("answer", ""),
                    ))

                if questions:
                    logger.info(f"MiMo 视觉分题+OCR: {len(questions)}题")
                    return regions, ocr_results

            except Exception as e:
                logger.warning(f"MiMo 视觉OCR失败(attempt {attempt+1}): {e}")
                if attempt < RETRY_MAX - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))

        raise RuntimeError(f"MiMo 视觉OCR失败，已重试{RETRY_MAX}次")

    async def grade_directly(self, image_base64=""):
        """MiMo 端到端批改：一张图一次调用完成 OCR + 分题 + 批改 + 分析"""
        image_base64 = _resize_b64(image_base64, self.MAX_IMAGE_DIM)

        # 解码获取 MiMo 实际看到的图像尺寸（用于百分比→像素换算）
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ref_h, ref_w = ref_img.shape[:2]

        for attempt in range(RETRY_MAX):
            try:
                async with self._semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": MIMO_DIRECT_PROMPT},
                        ]}],
                        max_tokens=4096,
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                data = parse_json(resp.choices[0].message.content)
                questions = data.get("questions", [])
                summary = data.get("summary", "")

                if not questions:
                    raise ValueError("MiMo 直连未返回题目数据")

                ocr_results = []
                regions = []
                for q in questions:
                    idx = q.get("index", len(ocr_results))
                    b = q.get("bbox", {})
                    # 百分比 → 像素（基于 MiMo 实际看到的图像尺寸）
                    bx = int(b.get("x", 0) / 100 * ref_w)
                    by = int(b.get("y", 0) / 100 * ref_h)
                    bw = int(b.get("width", 0) / 100 * ref_w)
                    bh = int(b.get("height", 0) / 100 * ref_h)
                    regions.append(QuestionRegion(idx, bbox=[bx, by, bx + bw, by + bh]))
                    ocr_results.append(OCRResult(
                        text=q.get("stem", ""),
                        confidence=0.9,
                        question_index=idx,
                        stem=q.get("stem", ""),
                        options=q.get("options", ""),
                        answer=q.get("student_answer", ""),
                    ))

                grading_data = {
                    "total_score": sum(q.get("score", 0) for q in questions),
                    "max_total_score": sum(q.get("max_score", 0) for q in questions),
                    "summary": summary,
                    "questions": questions,
                }
                logger.info(f"MiMo直连完成: {len(questions)}题, "
                           f"总分{grading_data['total_score']:.0f}/{grading_data['max_total_score']:.0f}")
                return regions, ocr_results, grading_data

            except Exception as e:
                logger.warning(f"MiMo直连失败(attempt {attempt+1}): {e}")
                if attempt < RETRY_MAX - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))

        raise RuntimeError(f"MiMo直连失败，已重试{RETRY_MAX}次")

    async def grade_directly_stream(self, image_base64=""):
        """流式批改：stream=True 接收 token，收完后整包解析 JSON，逐题 yield。

        Yields: (QuestionRegion, OCRResult, dict) 每道题分别推送
        """
        image_base64 = _resize_b64(image_base64, self.MAX_IMAGE_DIM)

        # 解码获取 MiMo 实际看到的图像尺寸（用于百分比→像素换算）
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ref_h, ref_w = ref_img.shape[:2]

        for attempt in range(RETRY_MAX):
            success = False
            try:
                async with self._semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": MIMO_DIRECT_PROMPT},
                        ]}],
                        max_tokens=4096,
                        stream=True,
                        temperature=0.2,
                    )

                # 收完整流内容
                full_text = ""
                async for chunk in resp:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_text += delta.content

                # 整包解析 JSON
                data = parse_json(full_text)
                questions = data.get("questions", [])

                if not questions:
                    raise ValueError("MiMo 流式未返回题目数据")

                for q in questions:
                    idx = q.get("index", len(questions))
                    b = q.get("bbox", {})
                    # 百分比 → 像素（基于 MiMo 实际看到的图像尺寸）
                    bx = int(b.get("x", 0) / 100 * ref_w)
                    by = int(b.get("y", 0) / 100 * ref_h)
                    bw = int(b.get("width", 0) / 100 * ref_w)
                    bh = int(b.get("height", 0) / 100 * ref_h)
                    region = QuestionRegion(idx, bbox=[bx, by, bx + bw, by + bh])
                    ocr = OCRResult(
                        text=q.get("stem", ""), confidence=0.9,
                        question_index=idx,
                        stem=q.get("stem", ""),
                        options=q.get("options", ""),
                        answer=q.get("student_answer", ""),
                    )
                    yield region, ocr, q

                logger.info(f"MiMo流式完成: {len(questions)}题")
                success = True

            except Exception as e:
                logger.warning(f"MiMo流式失败(attempt {attempt+1}): {e}")
                if attempt < RETRY_MAX - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))

            if success:
                return

        raise RuntimeError(f"MiMo流式失败，已重试{RETRY_MAX}次")

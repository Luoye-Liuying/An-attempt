"""
大模型批改模块 — 将OCR文本发送给大模型批改并评分
支持 OpenAI 兼容协议
"""
import asyncio
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from loguru import logger

from config import (
    LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
    MAX_CONCURRENT_GRADING, REQUEST_TIMEOUT,
    RETRY_MAX, RETRY_BACKOFF, DEFAULT_TOTAL_SCORE,
)
from ocr_api import OCRResult, parse_json


# ── 数据结构 ──

@dataclass
class GradingResult:
    question_index: int
    question_text: str = ""
    is_correct: bool = False
    score: float = 0.0
    max_score: float = 0.0
    analysis: str = ""
    suggestion: str = ""
    reference_answer: str = ""
    bbox: list = None


@dataclass
class FullGradingResult:
    total_score: float = 0.0
    max_total_score: float = 100.0
    question_results: list[GradingResult] = field(default_factory=list)
    summary: str = ""

    def accuracy_rate(self):
        return self.total_score / self.max_total_score if self.max_total_score > 0 else 0.0

    @classmethod
    def from_direct_result(cls, data: dict) -> "FullGradingResult":
        """从 MiMo 直连的原始字典构建 FullGradingResult"""
        questions = data.get("questions", [])
        qrs = []
        for q in questions:
            b = q.get("bbox", {})
            qrs.append(GradingResult(
                question_index=q.get("index", len(qrs)),
                question_text=q.get("stem", ""),
                is_correct=bool(q.get("is_correct")),
                score=float(q.get("score", 0)),
                max_score=float(q.get("max_score", 0)),
                analysis=q.get("analysis", ""),
                suggestion=q.get("suggestion", ""),
                reference_answer=q.get("reference_answer", ""),
                bbox=[b.get("x", 0), b.get("y", 0),
                      b.get("x", 0) + b.get("width", 0),
                      b.get("y", 0) + b.get("height", 0)],
            ))
        total = sum(q.score for q in qrs)
        mx = sum(q.max_score for q in qrs)
        return cls(
            total_score=total, max_total_score=mx,
            question_results=qrs,
            summary=data.get("summary", f"{len(qrs)}题 总分{total:.0f}/{mx:.0f}"),
        )


# ── 批改提示词 ──

GRADING_PROMPT = """你是中小学作业批改老师。请分析以下题目并严格按JSON格式返回：
{
  "reference_answer": "标准答案（含详细解题过程）",
  "is_correct": true或false,
  "analysis": "本题考点解析：无论对错都要分析，正确时说清考点和解题思路，错误时说明错因和正确做法",
  "suggestion": "改进建议：正确时鼓励并提示可优化点，错误时给出具体改进方向",
  "score_ratio": 0.0到1.0的得分比例
}"""


# ── 批改引擎 ──

class LLMGrader:
    """大模型批改引擎"""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.client = AsyncOpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=base_url or LLM_API_BASE,
            timeout=REQUEST_TIMEOUT,
        )
        self.model = model or LLM_MODEL
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GRADING)

    async def grade_single(self, ocr: OCRResult, max_score=10.0) -> GradingResult:
        """批改单题"""
        async with self._semaphore:
            for attempt in range(RETRY_MAX):
                try:
                    return await self._do_grade(ocr, max_score)
                except Exception as e:
                    logger.error(f"批改Q{ocr.question_index}失败(attempt {attempt+1}): {e}")
                    if attempt < RETRY_MAX - 1:
                        await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))
            return GradingResult(ocr.question_index, ocr.text, score=0, max_score=max_score,
                                 analysis="批改失败")

    async def grade_all(self, ocr_results: list[OCRResult],
                        scores: list[float] = None) -> FullGradingResult:
        """并发批改全部题目"""
        n = len(ocr_results)
        if scores is None:
            per = round(DEFAULT_TOTAL_SCORE / max(n, 1), 1)
            scores = [per] * n

        tasks = [self.grade_single(ocr, scores[i] if i < len(scores) else 10)
                 for i, ocr in enumerate(ocr_results)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        qrs = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                qrs.append(GradingResult(i, ocr_results[i].text if i < n else "",
                                         analysis=f"异常: {r}"))
            else:
                qrs.append(r)

        total = sum(q.score for q in qrs)
        mx = sum(q.max_score for q in qrs)
        correct = sum(1 for q in qrs if q.is_correct)

        return FullGradingResult(
            total_score=total, max_total_score=mx,
            question_results=qrs,
            summary=f"{len(qrs)}题 对{correct} 总分{total:.0f}/{mx:.0f} ({total/mx*100:.0f}%)" if mx > 0 else "无题目",
        )

    async def _do_grade(self, ocr: OCRResult, max_score: float) -> GradingResult:
        """执行批改请求"""
        content = ocr.structured_text
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GRADING_PROMPT},
                {"role": "user", "content": f"请批改：\n{content}\n满分：{max_score}分"},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        parsed = parse_json(resp.choices[0].message.content)
        score = round(min(max_score * parsed.get("score_ratio", 0), max_score))

        return GradingResult(
            question_index=ocr.question_index,
            question_text=ocr.text,
            is_correct=bool(parsed.get("is_correct")),
            score=score, max_score=max_score,
            analysis=parsed.get("analysis", ""),
            suggestion=parsed.get("suggestion", ""),
            reference_answer=parsed.get("reference_answer", ""),
        )

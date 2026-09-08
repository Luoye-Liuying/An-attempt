"""
系统配置 — 大模型API参数
通过 .env 文件或环境变量加载
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── 大模型 API (MiMo / OpenAI 兼容协议) ──
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ── 阿里云 OCR (真实 SDK) ──
ALIYUN_ACCESS_KEY = os.getenv("ALIYUN_ACCESS_KEY", "")
ALIYUN_ACCESS_SECRET = os.getenv("ALIYUN_ACCESS_SECRET", "")
ALIYUN_OCR_ENDPOINT = os.getenv("ALIYUN_OCR_ENDPOINT", "ocr-api.cn-hangzhou.aliyuncs.com")

# ── 图像预处理 ──
PREPROCESS_CONTRAST_ALPHA = 1.2
PREPROCESS_SHARPEN_STRENGTH = 1.0

# ── 批改 ──
DEFAULT_TOTAL_SCORE = 100

# ── 并发控制 ──
MAX_CONCURRENT_OCR = 5
MAX_CONCURRENT_GRADING = 10
REQUEST_TIMEOUT = 60
RETRY_MAX = 3
RETRY_BACKOFF = 1.0

# ── PDF ──
PDF_DPI = 150
PDF_MAX_PAGES = 10    # 单次最大处理页数

# ── MiMo 直连模式提示词 ──
MIMO_DIRECT_PROMPT = """你是中小学作业批改专家。请仔细分析这张试卷图片。

【任务】
1. 识别图片中每一道题目的内容（题干、选项）和在图片中的位置坐标
2. 读取学生的手写作答内容
3. 根据你的知识判断答案是否正确
4. 为每道题打分（满分由题目数量决定：100分 ÷ 题目数）
5. 给出总体评语

【坐标说明】用百分比表示题目在图片中的位置（0=最左/最上, 100=最右/最下）：
- x: 题目区域左上角距图片左边界的百分比（0-100）
- y: 题目区域左上角距图片上边界的百分比（0-100）
- width: 题目区域占图片宽度的百分比（0-100）
- height: 题目区域占图片高度的百分比（0-100）
例如：一道题在图片上半部分中间，x=10, y=20, width=80, height=15

【输出格式】只返回JSON，不包含其他内容：
{
  "questions": [
    {
      "index": 0,
      "bbox": {"x": 50, "y": 100, "width": 800, "height": 200},
      "stem": "题干文本",
      "options": "A. xx B. xx C. xx（若无选项则留空）",
      "student_answer": "学生作答内容",
      "reference_answer": "正确答案",
      "is_correct": true,
      "score": 10.0,
      "max_score": 10.0,
      "analysis": "本题考点解析：无论对错都要分析，正确时说清考点和解题思路",
      "suggestion": "改进建议"
    }
  ],
  "summary": "总体评语"
}

【评分准则】
- 完全正确：score = max_score
- 部分正确：根据正确程度给分
- 完全错误：score = 0
- 不要过于严格，对手写体识别要合理宽容"""

# ── 输出路径 ──
OUTPUT_DIR = BASE_DIR / "output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
QUESTIONS_DIR = OUTPUT_DIR / "questions"

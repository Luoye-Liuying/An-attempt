"""
作业批改系统 v4.5 — FastAPI 服务
端点: GET / | POST /api/grade | POST /api/grade/stream (真流式) | POST /api/chat | GET/DELETE /api/history
"""
import asyncio
import base64
import json
import pathlib
import time
import uuid

import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import OUTPUT_DIR, ANNOTATED_DIR, QUESTIONS_DIR, LLM_API_KEY, LLM_API_BASE, LLM_MODEL
from preprocess import ImagePreprocessor
from ocr_api import MiMoVisionClient
from grader import LLMGrader, FullGradingResult
from annotator import ResultAnnotator
from database import init_db, save_task, save_question, get_task, list_tasks, delete_task

app = FastAPI(title="作业批改系统", version="4.4")
_STATIC = pathlib.Path(__file__).resolve().parent / "static"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

mimo_ocr = MiMoVisionClient()
grader = LLMGrader()
annotator = ResultAnnotator(font_size=40, line_thickness=4)
preprocessor = ImagePreprocessor()

logger.info("作业批改系统 v4.4 就绪")

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
async def serve_frontend():
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.4"}


# ── 工具 ──

SCORE_PER_QUESTION = 10  # 每题满分 10 分


def _apply_scoring(grading: FullGradingResult):
    """统一计分：每题 10 分，对=10，错/半对=0，总分 = 10 × 题数"""
    n = len(grading.question_results)
    correct = 0
    for qr in grading.question_results:
        qr.max_score = SCORE_PER_QUESTION
        qr.score = SCORE_PER_QUESTION if qr.is_correct else 0.0
        if qr.is_correct:
            correct += 1
    grading.max_total_score = n * SCORE_PER_QUESTION
    grading.total_score = correct * SCORE_PER_QUESTION
    grading.summary = f"{n}题 对{correct} 总分{grading.total_score}/{grading.max_total_score} ({correct}/{n})"


async def _read_and_preprocess(upload: UploadFile):
    """读取上传图片并编码为 base64"""
    file_bytes = await upload.read()
    img = preprocessor.bytes_to_ndarray(file_bytes)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buf).decode()
    return img, img_b64, w, h


# ── 批改 ──

@app.post("/api/grade")
async def grade(
    image: UploadFile = File(...),
    mode: str = Query("mimo_direct"),
    deskew: bool = Query(True),
    enhance: bool = Query(True),
):
    task_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        img, img_b64, w, h = await _read_and_preprocess(image)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    logger.info(f"[{task_id}] {w}x{h} {image.filename}")

    # MiMo OCR + 批改
    regions, ocr_results = await mimo_ocr.full_pipeline(img_b64)
    grading = await grader.grade_all(ocr_results)
    _apply_scoring(grading)
    elapsed = time.time() - t0

    # ── bbox 坐标缩放 (MiMo内部缩至2048px, 需映射回原图) ──
    h_img, w_img = img.shape[:2]
    max_side = max(w_img, h_img)
    scale_2048 = max_side / mimo_ocr.MAX_IMAGE_DIM if max_side > mimo_ocr.MAX_IMAGE_DIM else 1.0
    if scale_2048 != 1.0:
        for r in regions:
            r.bbox = [int(v * scale_2048) for v in r.bbox]

    # ── 增强 + 先缩放到900px（标注后再缩放会导致字体太小） ──
    img = preprocessor.process(img, deskew=False, enhance=enhance)
    if w_img > 900:
        scale_900 = 900 / w_img
        img = cv2.resize(img, (900, int(h_img * scale_900)))
        # bbox 同步缩放到 900px 空间
        for r in regions:
            r.bbox = [int(v * scale_900) for v in r.bbox]

    # ── bbox 合并到 GradingResult ──
    for qr in grading.question_results:
        region = ResultAnnotator._find_region(qr.question_index, regions)
        if region and region.has_valid_bbox:
            qr.bbox = region.bbox

    # ── 标注 (900px 空间，字体清晰) ──
    loop = asyncio.get_event_loop()
    annotated = await loop.run_in_executor(
        None, annotator.annotate, img.copy(), grading, regions)

    # ── 标注图 JPEG 编码 ──
    _, ov_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    overview_b64 = base64.b64encode(ov_buf).decode()

    await save_task(task_id, image.filename or "unknown", mode,
                    grading.total_score, grading.max_total_score,
                    round(grading.accuracy_rate(), 3), grading.summary,
                    len(grading.question_results), round(elapsed, 1))
    for qr in grading.question_results:
        await save_question(task_id, {
            "index": qr.question_index,
            "question_text": qr.question_text,
            "student_answer": qr.question_text,
            "reference_answer": qr.reference_answer,
            "is_correct": qr.is_correct,
            "score": qr.score, "max_score": qr.max_score,
            "analysis": qr.analysis, "suggestion": qr.suggestion,
            "question_image": "",
        })

    return {
        "task_id": task_id, "filename": image.filename or "unknown",
        "question_count": len(grading.question_results),
        "total_score": grading.total_score,
        "max_total_score": grading.max_total_score,
        "accuracy": round(grading.accuracy_rate(), 3),
        "summary": grading.summary,
        "overview_image": overview_b64,
        "questions": [{
            "index": qr.question_index,
            "question_text": qr.question_text,
            "is_correct": qr.is_correct,
            "score": qr.score, "max_score": qr.max_score,
            "reference_answer": qr.reference_answer,
            "analysis": qr.analysis, "suggestion": qr.suggestion,
            "bbox": qr.bbox,
        } for qr in grading.question_results],
        "elapsed_seconds": round(elapsed, 1),
    }


# ── SSE 真流式批改 ──

@app.post("/api/grade/stream")
async def grade_stream(
    image: UploadFile = File(...),
    enhance: bool = Query(True),
):
    """真流式：阶段一 OCR 分题 → 阶段二 逐题并行批改，as_completed 谁先完先推"""
    task_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    async def generate():
        # 1. 读取图片
        try:
            img, img_b64, w, h = await _read_and_preprocess(image)
        except ValueError as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return

        logger.info(f"[{task_id}] SSE {w}x{h} {image.filename}")

        # 2. 阶段一：MiMo OCR 识别题目（一次调用，只分题+识字）
        try:
            yield f"event: progress\ndata: {json.dumps({'msg': '🔍 正在识别题目…'}, ensure_ascii=False)}\n\n"
            regions, ocr_results = await mimo_ocr.full_pipeline(img_b64)
        except Exception as e:
            logger.error(f"[{task_id}] OCR失败: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return

        n = len(ocr_results)
        if n == 0:
            yield f"event: error\ndata: {json.dumps({'error': '未识别到题目'}, ensure_ascii=False)}\n\n"
            return

        yield f"event: progress\ndata: {json.dumps({'msg': f'📋 {n} 题已识别，并行批改中…'}, ensure_ascii=False)}\n\n"

        # 3. bbox 从 2048 缩放回原图
        max_side = max(w, h)
        scale_2048 = max_side / mimo_ocr.MAX_IMAGE_DIM if max_side > mimo_ocr.MAX_IMAGE_DIM else 1.0
        if scale_2048 != 1.0:
            for r in regions:
                r.bbox = [int(v * scale_2048) for v in r.bbox]

        # 4. 阶段二：逐题并行批改，as_completed 真流式 — 谁先批完谁先推
        gr_map = {}

        async def grade_one(idx, ocr_result):
            gr = await grader.grade_single(ocr_result, SCORE_PER_QUESTION)
            return idx, gr

        tasks = [grade_one(i, ocr) for i, ocr in enumerate(ocr_results)]

        for coro in asyncio.as_completed(tasks):
            idx, gr = await coro
            gr_map[idx] = gr

            region = ResultAnnotator._find_region(idx, regions)
            if region and region.has_valid_bbox:
                gr.bbox = region.bbox

            ev = json.dumps({
                'index': idx,
                'question_text': gr.question_text,
                'student_answer': ocr_results[idx].answer,
                'reference_answer': gr.reference_answer,
                'is_correct': gr.is_correct,
                'score': gr.score,
                'max_score': gr.max_score,
                'analysis': gr.analysis,
                'suggestion': gr.suggestion,
                'bbox': gr.bbox,
            }, ensure_ascii=False)
            yield f"event: question\ndata: {ev}\n\n"

        # 5. 汇总 + 总览图
        qrs = [gr_map[i] for i in sorted(gr_map)]
        total = sum(q.score for q in qrs)
        max_total = sum(q.max_score for q in qrs)
        correct = sum(1 for q in qrs if q.is_correct)
        summary = f"{n}题 对{correct} 总分{total:.0f}/{max_total:.0f}"

        try:
            img_proc = preprocessor.process(img, deskew=False, enhance=enhance)
            if w > 900:
                scale_900 = 900 / w
                img_proc = cv2.resize(img_proc, (900, int(h * scale_900)))
                for r in regions:
                    r.bbox = [int(v * scale_900) for v in r.bbox]

            grading = FullGradingResult(
                total_score=total, max_total_score=max_total,
                question_results=qrs, summary=summary)

            loop = asyncio.get_event_loop()
            annotated = await loop.run_in_executor(
                None, annotator.annotate, img_proc.copy(), grading, regions)

            _, ov_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            overview_b64 = base64.b64encode(ov_buf).decode()
            elapsed = time.time() - t0

            ov = json.dumps({
                'task_id': task_id,
                'overview_image': overview_b64,
                'total_score': total,
                'max_total_score': max_total,
                'accuracy': round(total / max_total, 3) if max_total > 0 else 0,
                'summary': summary,
                'question_count': n,
                'elapsed_seconds': round(elapsed, 1),
            }, ensure_ascii=False)
            yield f"event: overview\ndata: {ov}\n\n"

            await save_task(task_id, image.filename or "unknown", "stream",
                            total, max_total,
                            round(total / max_total, 3) if max_total > 0 else 0,
                            summary, n, round(elapsed, 1))
            for gr in qrs:
                await save_question(task_id, {
                    "index": gr.question_index,
                    "question_text": gr.question_text,
                    "student_answer": gr.question_text,
                    "reference_answer": gr.reference_answer,
                    "is_correct": gr.is_correct,
                    "score": gr.score, "max_score": gr.max_score,
                    "analysis": gr.analysis, "suggestion": gr.suggestion,
                    "question_image": "",
                })

            yield f"event: done\ndata: {json.dumps({'task_id': task_id, 'elapsed': round(elapsed, 1)}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"[{task_id}] 标注失败: {e}")
            yield f"event: error\ndata: {json.dumps({'error': f'标注失败: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── Chat ──

CHAT_SYSTEM = """你是作业辅导老师。根据提供的批改记录回答学生问题。用友好中文，简洁明了。"""


@app.post("/api/chat")
async def chat(request: dict):
    task_id = request.get("task_id", "")
    message = request.get("message", "").strip()
    history = request.get("history", [])

    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    context_parts = [CHAT_SYSTEM]
    if task_id:
        task = await get_task(task_id)
        if task:
            context_parts.append(f"\n【批改记录】{task.get('filename','')}")
            context_parts.append(f"总分:{task.get('total_score',0)}/{task.get('max_total_score',100)}")
            for q in task.get("questions", []):
                s = "✓" if q.get("is_correct") else "✗"
                context_parts.append(
                    f"{q.get('question_index',0)+1}:{s} {q.get('score',0)}/{q.get('max_score',0)} "
                    f"题目:{q.get('question_text','')} "
                    f"答案:{q.get('reference_answer','')} "
                    f"解析:{q.get('analysis','')}")

    msgs = [{"role": "system", "content": "\n".join(context_parts)}]
    for h in history[-10:]:
        if h.get("role") in ("user", "assistant"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": message})

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE, timeout=60)
        resp = await client.chat.completions.create(
            model=LLM_MODEL, messages=msgs, max_tokens=1024, temperature=0.7)
        return {"reply": resp.choices[0].message.content, "task_id": task_id}
    except Exception as e:
        logger.error(f"Chat 失败: {e}")
        return JSONResponse({"error": f"对话失败: {e}"}, status_code=500)


# ── 历史 ──

@app.get("/api/history")
async def history_list(page: int = Query(1), size: int = Query(20)):
    tasks = await list_tasks(page, min(size, 50))
    return {"page": page, "size": size, "items": tasks}


@app.get("/api/history/{task_id}")
async def history_detail(task_id: str):
    task = await get_task(task_id)
    if not task:
        return JSONResponse({"error": "记录不存在"}, status_code=404)
    return task


@app.delete("/api/history/{task_id}")
async def history_delete(task_id: str):
    ok = await delete_task(task_id)
    if not ok:
        return JSONResponse({"error": "记录不存在"}, status_code=404)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="::", port=8080)

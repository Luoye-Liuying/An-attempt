"""
测试运行器 — 批量批改测试样本，输出报告
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import cv2
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, ANNOTATED_DIR, QUESTIONS_DIR
from preprocess import ImagePreprocessor, imread_safe
from ocr_api import AliyunOCRClient
from grader import LLMGrader
from annotator import ResultAnnotator

TEST_DIR = Path(__file__).resolve().parent / "samples"


async def process_one(fp, preprocessor, ocr_client, grader, annotator):
    """处理单张样本"""
    t0 = time.time()
    try:
        img = imread_safe(fp)
        h, w = img.shape[:2]
        processed = preprocessor.process(img)
        _, buf = cv2.imencode(".png", img)
        img_b64 = base64.b64encode(buf).decode()

        # OCR + 批改
        regions, ocr_results = await ocr_client.full_pipeline(img_b64)
        # 坐标缩放
        scale = max(w, h) / 2048.0 if max(w, h) > 2048 else 1.0
        for r in regions:
            r.bbox = [int(v * scale) for v in r.bbox]

        grading = await grader.grade_all(ocr_results)

        # 标注
        annotated = annotator.annotate(processed, grading, regions)
        out_path = ANNOTATED_DIR / f"{fp.stem}_annotated.png"
        cv2.imwrite(str(out_path), annotated)

        # 逐题批注
        q_dir = QUESTIONS_DIR / fp.stem
        q_dir.mkdir(parents=True, exist_ok=True)
        for q in grading.question_results:
            region = annotator._find_region(q.question_index, regions)
            crop = img.copy()
            if region and region.has_valid_bbox:
                x1, y1, x2, y2 = [max(0, min(int(v), d)) for v, d in
                                  zip(region.bbox, [w, h, w, h])]
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2].copy()
            q_img = annotator.annotate_single_question(crop, q, q.question_index + 1)
            cv2.imwrite(str(q_dir / f"Q{q.question_index+1}.png"), q_img)

        elapsed = time.time() - t0
        return {"file": fp.name, "score": grading.total_score, "max": grading.max_total_score,
                "accuracy": round(grading.accuracy_rate(), 3), "questions": len(ocr_results),
                "elapsed": round(elapsed, 1),
                "details": [{"idx": q.question_index, "correct": q.is_correct,
                             "score": f"{q.score}/{q.max_score}", "analysis": q.analysis}
                            for q in grading.question_results]}
    except Exception as e:
        return {"file": fp.name, "error": str(e), "elapsed": round(time.time()-t0, 1)}


async def run_tests():
    if not TEST_DIR.exists():
        return {"error": "测试样本目录不存在"}
    samples = sorted(TEST_DIR.glob("*.jpg"))
    if not samples:
        return {"error": "未找到样本(jpg)"}

    logger.info(f"阿里云OCR模式 — {len(samples)}张样本")

    ocr = AliyunOCRClient()
    grader = LLMGrader()
    preprocessor = ImagePreprocessor()
    annotator = ResultAnnotator()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tasks = [process_one(fp, preprocessor, ocr, grader, annotator) for fp in samples]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]

    report = {
        "total": len(samples),
        "ok": len(ok), "errors": len(err),
        "elapsed": round(elapsed, 1),
        "avg": round(sum(r["elapsed"] for r in ok) / max(len(ok), 1), 1),
        "results": results,
    }

    with open(OUTPUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"完成: {len(ok)}/{len(samples)} 成功, {elapsed:.1f}s, 均{report['avg']}s/张")
    return report


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
               level="INFO")

    report = asyncio.run(run_tests())

    print(f"\n{'='*50}\n测试报告\n{'='*50}")
    if "error" in report:
        print(f"错误: {report['error']}")
        return

    print(f"样本: {report['total']} | 成功: {report['ok']} | 失败: {report['errors']}")
    print(f"耗时: {report['elapsed']}s | 均: {report['avg']}s/张\n")

    for r in report["results"]:
        if "error" in r:
            print(f"  [FAIL] {r['file']}: {r['error']}")
        else:
            print(f"  [OK] {r['file']}: {r['score']}/{r['max']} ({r['accuracy']*100:.0f}%) "
                  f"{r['questions']}题 {r['elapsed']}s")


if __name__ == "__main__":
    main()

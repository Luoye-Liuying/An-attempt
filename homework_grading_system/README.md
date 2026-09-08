# 作业批改系统 v4.3

基于大模型的中小学作业自动批改系统，MiMo 端到端 + SQLite 持久化。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API 密钥
cp .env.example .env
# 编辑 .env 填入 MiMo API 密钥

# 3. 启动
python run.py
# → http://localhost:8080
```

## 批改流程

```
上传图片/PDF → MiMo 多模态识别+切题 → 大模型逐题批改 → SQLite保存 → 前端展示
```

- MiMo 端到端：一次调用完成 OCR 识别 + 题目切分 + 答案判断 + 评分 + 评语
- 批改结果自动存入 SQLite，支持历史回溯
- `/api/chat` 端点提供 AI 辅导对话，以批改记录为上下文

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/grade` | 上传图片/PDF 批改 |
| POST | `/api/chat` | AI辅导对话（基于批改记录） |
| GET | `/api/history` | 历史记录列表（分页） |
| GET | `/api/history/{id}` | 历史记录详情 |
| DELETE | `/api/history/{id}` | 删除历史记录 |

## 文件结构

```
├── run.py              # 启动入口（v4.3横幅 + IP地址检测）
├── server.py           # FastAPI 服务（grade/chat/history端点）
├── config.py           # 配置（从 .env 读取，含MiMo提示词）
├── ocr_api.py          # OCR 模块（MiMo视觉 + 阿里云SDK备用）
├── grader.py           # 大模型批改引擎
├── annotator.py        # 批注标注绘制
├── preprocess.py       # 图像预处理（校正+增强）
├── pdf_handler.py      # PDF 转图像
├── database.py         # SQLite持久化（tasks + questions）
├── test_runner.py      # 批量测试脚本
├── static/index.html   # Web 前端（拖拽/拍照/Chat/历史）
├── samples/            # 测试样本（6张jpg）
└── output/             # 输出（自动创建）
```

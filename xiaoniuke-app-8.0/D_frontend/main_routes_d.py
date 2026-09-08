"""
小牛客 - 在线编程评测系统
======================================
整合了前端界面、题目管理、AI 服务、判题服务四大模块。

启动方式：
    cd xiaoniuke.app
    pip install -r requirements.txt
    python main.py

    浏览器打开 http://127.0.0.1:8000

默认测试账号：
    教师：teacher1 / 123456
    学生：student1 / 123456
    教师验证码：Zmjjkk
"""

from fastapi import FastAPI, Request, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import bcrypt as bcrypt_lib
from contextlib import asynccontextmanager
from jinja2 import Environment, FileSystemLoader
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from urllib.parse import urlencode, quote
import json
import secrets

from C_base_service import (
    init_db, get_db, seed_data, renumber_problems,
    User, Problem, Submission,
    UserRole, Difficulty, ReviewStatus, CompileStatus
)
from B_judge_module import CodeJudge
from A_problem_module import ai_service

# ---------- 教师注册校验码 ----------
TEACHER_VERIFICATION_CODE = "Zmjjkk"


# ---------- 应用初始化 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库和种子数据"""
    init_db()
    seed_data()
    yield


app = FastAPI(
    title="小牛客",
    description="在线编程评测系统 — 支持教师出题、AI 辅助、自动判题",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件 & 模板
app.mount("/static", StaticFiles(directory="static"), name="static")
jinja_env = Environment(loader=FileSystemLoader("templates"), autoescape=True)


def render(name: str, **kwargs) -> HTMLResponse:
    """使用 Jinja2 渲染模板"""
    template = jinja_env.get_template(name)
    return HTMLResponse(template.render(**kwargs))


# ---------- 会话中间件 ----------
app.add_middleware(
    SessionMiddleware,
    secret_key="xiaoniuke-secret-key-2026",
    session_cookie="xiaoniuke_session",
    max_age=86400,  # 24小时
)


# ---------- 认证工具 ----------
def get_current_user(request: Request, db: Session) -> User | None:
    """获取当前登录用户"""
    user_id = request.session.get("user_id")
    if user_id:
        return db.query(User).filter(User.id == user_id).first()
    return None


def require_login(request: Request, db: Session) -> User:
    """要求登录，同时检查是否被其他设备挤下线"""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    # 单设备登录：检查 session token 是否匹配
    session_token = request.session.get("token", "")
    if user.session_token and session_token != user.session_token:
        request.session.clear()
        raise HTTPException(status_code=303, headers={"Location": "/login?kicked=1"})
    return user


def require_role(request: Request, db: Session, role: UserRole) -> User:
    """要求特定角色，否则重定向到对应仪表盘"""
    user = require_login(request, db)
    if user.role != role:
        if user.role == UserRole.teacher:
            raise HTTPException(status_code=303, headers={"Location": "/teacher/dashboard"})
        else:
            raise HTTPException(status_code=303, headers={"Location": "/student/dashboard"})
    return user


# ====================== 公开页面 ======================


# ===== D-前端 路由 =====
@app.get("/student/dashboard", response_class=HTMLResponse)
async def student_dashboard(
    request: Request,
    difficulty: str = Query(None),
    tag: str = Query(None),
    search: str = Query(None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    """学生仪表盘：浏览题库（分页 + 搜索）"""
    user = require_role(request, db, UserRole.student)
    page_size = 15

    base_query = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved)
    if difficulty:
        base_query = base_query.filter(Problem.difficulty == Difficulty(difficulty))
    if tag:
        base_query = base_query.filter(Problem.tags.contains(tag))
    if search:
        base_query = base_query.filter(Problem.title.contains(search))

    total_count = base_query.count()
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    problems = base_query.order_by(Problem.display_order.asc()).offset((page - 1) * page_size).limit(page_size).all()

    # 收集所有标签（从全部题目中）
    all_problems = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved).all()
    all_tags = set()
    for p in all_problems:
        if p.tags:
            for t in p.tags.split(","):
                t = t.strip()
                if t:
                    all_tags.add(t)

    # 计算每道题的得分率 + 出题老师
    problem_score_rates = {}
    problem_creators = {}
    for p in problems:
        subs = db.query(Submission).filter(Submission.problem_id == p.id).all()
        if subs:
            total_possible = len(subs) * 100
            total_actual = sum(s.score for s in subs)
            problem_score_rates[p.id] = round(total_actual / total_possible * 100, 1)
        else:
            problem_score_rates[p.id] = None
        if p.creator:
            problem_creators[p.id] = p.creator.username

    return render(
        "student/dashboard.html",
        request=request,
        user=user,
        problems=problems,
        all_tags=sorted(all_tags),
        current_difficulty=difficulty,
        current_tag=tag,
        current_search=search or "",
        problem_score_rates=problem_score_rates,
        problem_creators=problem_creators,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


@app.get("/student/problem/{problem_id}", response_class=HTMLResponse)
async def student_answer_page(
    request: Request,
    problem_id: int,
    db: Session = Depends(get_db),
):
    """学生答题页面"""
    user = require_role(request, db, UserRole.student)
    problem = db.query(Problem).filter(
        Problem.id == problem_id,
        Problem.review_status == ReviewStatus.approved,
    ).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    submissions = (
        db.query(Submission)
        .filter(
            Submission.problem_id == problem_id,
            Submission.user_id == user.id,
        )
        .order_by(Submission.created_at.desc())
        .all()
    )

    # 检查是否有刚提交的结果
    last_submission = None
    last_test_results = []
    sid = request.query_params.get("submission_id")
    if sid:
        try:
            last_sub = db.query(Submission).filter(Submission.id == int(sid)).first()
            if last_sub:
                last_submission = last_sub
                last_test_results = last_sub.get_test_results()
        except (ValueError, TypeError):
            pass

    # 获取测试用例数量
    test_cases = problem.get_test_cases()
    test_case_count = len(test_cases)
    passed_cases = last_submission.score * test_case_count // 100 if last_submission and test_case_count > 0 else 0
    # 只要有过一次 >=60 分的提交就显示参考解答
    show_ref_answer = any(s.score >= 60 for s in submissions) if submissions else False

    # 获取上一题和下一题的 ID（用于导航，按 display_order）
    all_approved = (
        db.query(Problem)
        .filter(Problem.review_status == ReviewStatus.approved)
        .order_by(Problem.display_order.asc())
        .all()
    )
    prev_id = None
    next_id = None
    for i, p in enumerate(all_approved):
        if p.id == problem_id:
            if i > 0:
                prev_id = all_approved[i - 1].id
            if i < len(all_approved) - 1:
                next_id = all_approved[i + 1].id
            break

    # 上次提交的代码（用于回填编辑器）
    last_code = request.query_params.get("last_code", "")

    return render(
        "student/answer.html",
        request=request,
        user=user,
        problem=problem,
        submissions=submissions,
        last_submission=last_submission,
        last_test_results=last_test_results,
        test_case_count=test_case_count,
        passed_cases=passed_cases,
        prev_id=prev_id,
        next_id=next_id,
        last_code=last_code,
        show_ref_answer=show_ref_answer,
    )



# 启动代码需与主项目保持一致

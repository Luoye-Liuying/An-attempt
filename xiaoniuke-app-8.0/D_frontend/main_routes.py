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

@app.get("/", response_class=HTMLResponse)

# ===== D-前端 路由 =====

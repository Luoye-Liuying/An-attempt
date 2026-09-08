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

from fastapi import FastAPI, Request, Form, Depends, HTTPException, Query, UploadFile, File
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
import os
import uuid

from C_base_service import (
    init_db, get_db, seed_data, renumber_problems, migrate_constraints_to_statement,
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
    migrate_constraints_to_statement()
    yield


app = FastAPI(
    title="小牛客",
    description="在线编程评测系统 — 支持教师出题、AI 辅助、自动判题",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件 & 模板
app.mount("/static", StaticFiles(directory="D_frontend/static"), name="static")
jinja_env = Environment(loader=FileSystemLoader("D_frontend/templates"), autoescape=True)


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
async def index(request: Request):
    """首页"""
    return render("index.html", request=request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """登录页面"""
    return render("login.html", request=request)


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """处理登录"""
    user = db.query(User).filter(User.username == username).first()
    if not user or not bcrypt_lib.checkpw(
        password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        return render("login.html", request=request, error="用户名或密码错误")

    # 生成新 session_token，使旧登录失效（单设备登录）
    new_token = secrets.token_hex(32)
    user.session_token = new_token
    db.commit()

    request.session["user_id"] = user.id
    request.session["username"] = user.username
    request.session["role"] = user.role.value
    request.session["token"] = new_token
    request.session["display_name"] = user.display_name or user.username
    request.session["avatar"] = user.avatar or ""
    request.session["bio"] = user.bio or ""
    request.session["gender"] = user.gender or "无"

    return RedirectResponse("/", status_code=303)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """注册页面"""
    return render("register.html", request=request)


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    role: str = Form("student"),
    teacher_code: str = Form(""),
    db: Session = Depends(get_db),
):
    """处理注册"""
    if not username or len(username) < 3:
        return render("register.html", request=request, error="用户名至少3个字符")
    if len(password) < 6:
        return render("register.html", request=request, error="密码至少6个字符")
    if password != confirm_password:
        return render("register.html", request=request, error="两次密码不一致")

    # 教师注册需要校验码
    if role == "teacher":
        if not teacher_code or teacher_code.strip() != TEACHER_VERIFICATION_CODE:
            return render("register.html", request=request,
                          error="教师校验码错误，请联系校方获取正确的校验码")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        return render("register.html", request=request, error="用户名已存在")

    user = User(
        username=username,
        password_hash=bcrypt_lib.hashpw(
            password.encode("utf-8"), bcrypt_lib.gensalt()
        ).decode("utf-8"),
        role=UserRole(role),
    )
    db.add(user)
    db.commit()

    return RedirectResponse("/login?registered=1", status_code=303)


@app.get("/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    """退出登录"""
    user = get_current_user(request, db)
    if user:
        user.session_token = None
        db.commit()
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# ====================== 个人设置（通用，教师/学生均可用） ======================

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """个人设置页面"""
    user = require_login(request, db)
    saved = request.query_params.get("saved")
    return render("profile.html", request=request, user=user, saved=saved)


@app.post("/profile")
async def profile_save(
    request: Request,
    display_name: str = Form(""),
    gender: str = Form("无"),
    bio: str = Form(""),
    avatar_value: str = Form(""),
    avatar_file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """保存个人设置"""
    user = require_login(request, db)

    # 更新显示名称（允许空值=不设置，显示为账号名）
    user.display_name = display_name.strip() if display_name.strip() else None

    # 更新性别
    user.gender = gender if gender in ("男", "女", "无") else "无"

    # 更新个性签名
    user.bio = bio.strip() if bio.strip() else None

    # 更新头像
    if avatar_file and avatar_file.filename:
        # 用户上传了自定义头像
        try:
            # 验证文件类型
            ext = os.path.splitext(avatar_file.filename)[1].lower()
            allowed_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
            if ext not in allowed_exts:
                return render("profile.html", request=request, user=user,
                              error="不支持的文件格式，请上传 PNG/JPG/GIF/WebP/SVG 格式的图片")

            # 验证文件大小（读取前检查不了大小，但可以通过实际读取后判断）
            contents = await avatar_file.read()
            if len(contents) > 2 * 1024 * 1024:
                return render("profile.html", request=request, user=user,
                              error="图片大小不能超过 2MB")

            # 生成唯一文件名并保存
            unique_name = f"{uuid.uuid4().hex}{ext}"
            save_dir = os.path.join("D_frontend", "static", "avatars", "uploaded")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, unique_name)
            with open(save_path, "wb") as f:
                f.write(contents)

            # 删除旧的上传头像（如果是上传的）
            if user.avatar and user.avatar.startswith("uploaded/"):
                old_path = os.path.join("D_frontend", "static", "avatars", user.avatar)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

            user.avatar = f"uploaded/{unique_name}"

        except Exception as e:
            return render("profile.html", request=request, user=user,
                          error=f"头像上传失败: {str(e)}")
    elif avatar_value:
        # 选择了默认头像
        # 删除旧的上传头像
        if user.avatar and user.avatar.startswith("uploaded/"):
            old_path = os.path.join("D_frontend", "static", "avatars", user.avatar)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        user.avatar = avatar_value
    else:
        # 没有选择任何头像，保留原有头像或使用默认
        if not user.avatar:
            user.avatar = "default/1.svg"

    db.commit()

    # 更新 session
    request.session["display_name"] = user.display_name or user.username
    request.session["avatar"] = user.avatar or ""
    request.session["bio"] = user.bio or ""
    request.session["gender"] = user.gender or "无"

    return RedirectResponse("/profile?saved=1", status_code=303)


# ====================== 教师端 ======================

@app.get("/teacher/dashboard", response_class=HTMLResponse)
async def teacher_dashboard(
    request: Request,
    my_only: str = Query(None),
    search: str = Query(None),
    tag: str = Query(None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    """教师仪表盘：查看、管理题目（分页 + 搜索 + 标签筛选）"""
    user = require_role(request, db, UserRole.teacher)
    page_size = 15

    base_query = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved)
    if my_only == "1":
        base_query = base_query.filter(Problem.created_by == user.id)
    if search:
        base_query = base_query.filter(Problem.title.contains(search))
    if tag:
        base_query = base_query.filter(Problem.tags.contains(tag))

    total_all = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved).count()
    total_count = base_query.count()
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    problems = base_query.order_by(Problem.display_order.asc()).offset((page - 1) * page_size).limit(page_size).all()

    # 自己出的题目数
    my_count = db.query(Problem).filter(
        Problem.review_status == ReviewStatus.approved,
        Problem.created_by == user.id,
    ).count()

    # 统计各难度数量
    easy_count = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved, Problem.difficulty == Difficulty.easy).count()
    medium_count = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved, Problem.difficulty == Difficulty.medium).count()
    hard_count = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved, Problem.difficulty == Difficulty.hard).count()

    # 收集所有标签
    all_problems = db.query(Problem).filter(Problem.review_status == ReviewStatus.approved).all()
    all_tags = set()
    for p in all_problems:
        if p.tags:
            for t in p.tags.split(","):
                t = t.strip()
                if t:
                    all_tags.add(t)

    # 计算每道题的得分率和提交人数
    problem_score_rates = {}
    problem_submission_counts = {}
    for p in problems:
        subs = db.query(Submission).filter(Submission.problem_id == p.id).all()
        if subs:
            total_possible = len(subs) * 100
            total_actual = sum(s.score for s in subs)
            problem_score_rates[p.id] = round(total_actual / total_possible * 100, 1)
        else:
            problem_score_rates[p.id] = None
        problem_submission_counts[p.id] = len(subs)

    return render(
        "teacher/dashboard.html",
        request=request,
        user=user,
        problems=problems,
        problem_score_rates=problem_score_rates,
        problem_submission_counts=problem_submission_counts,
        my_only=my_only,
        my_count=my_count,
        total_count=total_all,
        easy_count=easy_count,
        medium_count=medium_count,
        hard_count=hard_count,
        all_tags=sorted(all_tags),
        current_tag=tag or "",
        current_search=search or "",
        page=page,
        total_pages=total_pages,
    )


@app.get("/teacher/problems/add", response_class=HTMLResponse)
async def teacher_add_problem_page(request: Request, db: Session = Depends(get_db)):
    """新增题目页面"""
    user = require_role(request, db, UserRole.teacher)
    return render("teacher/add_problem.html", request=request, user=user)


@app.post("/teacher/problems/add")
async def teacher_add_problem(
    request: Request,
    title: str = Form(...),
    statement: str = Form(...),
    constraints: str = Form(""),
    tags: str = Form(""),
    difficulty: str = Form("easy"),
    standard_answer: str = Form(""),
    test_cases_json: str = Form(""),
    db: Session = Depends(get_db),
):
    """教师手动新增题目（AI 生成通过 AJAX 接口单独调用）"""
    user = require_role(request, db, UserRole.teacher)

    # 清理标签
    clean_tags = ",".join([t.strip() for t in tags.replace("，", ",").split(",") if t.strip()])

    problem = Problem(
        title=title,
        statement=statement,
        tags=clean_tags,
        difficulty=Difficulty(difficulty),
        standard_answer=standard_answer,
        source="manual",
        review_status=ReviewStatus.approved,
        created_by=user.id,
    )
    # 保存测试用例
    if test_cases_json.strip():
        try:
            cases = json.loads(test_cases_json)
            problem.set_test_cases(cases)
        except json.JSONDecodeError:
            pass  # JSON 解析失败则忽略

    db.add(problem)
    db.commit()
    # 新题目分配正确的显示序号
    renumber_problems(db)
    return RedirectResponse("/teacher/dashboard?added=1", status_code=303)


@app.post("/teacher/problems/ai-generate")
async def teacher_ai_generate(
    request: Request,
    title: str = Form(...),
    statement: str = Form(...),
    tags: str = Form(""),
    difficulty: str = Form("easy"),
    db: Session = Depends(get_db),
):
    """AI 生成标准答案和测试用例（AJAX 接口，返回 JSON）"""
    user = require_role(request, db, UserRole.teacher)

    clean_tags = ",".join([t.strip() for t in tags.replace("，", ",").split(",") if t.strip()])

    try:
        result = ai_service.generate_answer(
            title=title,
            statement=statement,
            tags=clean_tags,
            difficulty=difficulty,
        )
        from fastapi.responses import JSONResponse
        return JSONResponse({
            "success": True,
            "standard_answer": result.get("standard_answer", ""),
            "test_cases": result.get("test_cases", []),
        })
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse({
            "success": False,
            "error": str(e),
        }, status_code=500)


@app.get("/teacher/problems/{problem_id}/view", response_class=HTMLResponse)
async def teacher_view_problem(
    request: Request,
    problem_id: int,
    db: Session = Depends(get_db),
):
    """教师查看任意题目详情（只读，不可编辑）"""
    user = require_role(request, db, UserRole.teacher)
    problem = db.query(Problem).filter(Problem.id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    test_cases_json = json.dumps(problem.get_test_cases(), ensure_ascii=False) if problem.test_cases else ""
    is_owner = problem.created_by == user.id

    return render(
        "teacher/edit_problem.html",
        request=request,
        user=user,
        problem=problem,
        test_cases_json=test_cases_json,
        read_only=not is_owner,
    )


@app.get("/teacher/problems/{problem_id}/edit", response_class=HTMLResponse)
async def teacher_edit_problem_page(
    request: Request,
    problem_id: int,
    db: Session = Depends(get_db),
):
    """编辑题目页面"""
    user = require_role(request, db, UserRole.teacher)
    problem = db.query(Problem).filter(
        Problem.id == problem_id, Problem.created_by == user.id
    ).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    test_cases_json = json.dumps(problem.get_test_cases(), ensure_ascii=False) if problem.test_cases else ""

    return render(
        "teacher/edit_problem.html",
        request=request,
        user=user,
        problem=problem,
        test_cases_json=test_cases_json,
    )


@app.post("/teacher/problems/{problem_id}/edit")
async def teacher_edit_problem(
    request: Request,
    problem_id: int,
    title: str = Form(...),
    statement: str = Form(...),
    tags: str = Form(""),
    difficulty: str = Form("easy"),
    standard_answer: str = Form(""),
    test_cases_json: str = Form(""),
    db: Session = Depends(get_db),
):
    """处理编辑题目"""
    user = require_role(request, db, UserRole.teacher)
    problem = db.query(Problem).filter(
        Problem.id == problem_id, Problem.created_by == user.id
    ).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    problem.title = title
    problem.statement = statement
    problem.tags = ",".join([t.strip() for t in tags.replace("，", ",").split(",") if t.strip()])
    problem.difficulty = Difficulty(difficulty)
    problem.standard_answer = standard_answer

    if test_cases_json.strip():
        try:
            cases = json.loads(test_cases_json)
            problem.set_test_cases(cases)
        except json.JSONDecodeError:
            pass

    db.commit()
    return RedirectResponse("/teacher/dashboard?updated=1", status_code=303)


@app.get("/teacher/problems/{problem_id}/delete")
async def teacher_delete_problem(
    request: Request,
    problem_id: int,
    db: Session = Depends(get_db),
):
    """删除题目"""
    user = require_role(request, db, UserRole.teacher)
    problem = db.query(Problem).filter(
        Problem.id == problem_id, Problem.created_by == user.id
    ).first()
    if problem:
        db.query(Submission).filter(Submission.problem_id == problem_id).delete()
        db.delete(problem)
        db.commit()
        renumber_problems(db)  # 删除后重排序号
    return RedirectResponse("/teacher/dashboard?deleted=1", status_code=303)


@app.get("/teacher/problems/{problem_id}/submissions", response_class=HTMLResponse)
async def teacher_view_submissions(
    request: Request,
    problem_id: int,
    db: Session = Depends(get_db),
):
    """教师查看某道题的所有学生提交详情"""
    user = require_role(request, db, UserRole.teacher)
    problem = db.query(Problem).filter(Problem.id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    # 获取所有学生提交（按时间倒序）
    submissions = (
        db.query(Submission)
        .filter(Submission.problem_id == problem_id)
        .order_by(Submission.created_at.desc())
        .all()
    )

    # 关联学生用户名
    sub_data = []
    for s in submissions:
        student = db.query(User).filter(User.id == s.user_id).first()
        sub_data.append({
            "id": s.id,
            "student_name": (student.display_name or student.username) if student else "未知",
            "code": s.code,
            "language": s.language,
            "compile_status": s.compile_status.value if s.compile_status else "?",
            "score": s.score,
            "compile_error": s.compile_error,
            "test_results": s.get_test_results(),
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "",
        })

    return render(
        "teacher/view_submissions.html",
        request=request,
        user=user,
        problem=problem,
        submissions=sub_data,
    )


# ====================== 学生端 ======================

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
            problem_creators[p.id] = p.creator.display_name or p.creator.username

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
        ai_feedback=last_submission.compile_error if last_submission and last_submission.compile_error and (last_submission.compile_status.value == "compile_success" or last_submission.compile_status.value == "compile_error") else None,
    )


@app.post("/student/problem/{problem_id}/submit")
async def student_submit_code(
    request: Request,
    problem_id: int,
    code: str = Form(""),
    language: str = Form("python"),
    db: Session = Depends(get_db),
):
    """
    学生提交代码 → 判题
    判题流程：
      1. 语法检查 → 有错误：0 分，显示错误信息
      2. 语法通过 → 用测试用例实际运行 → 每组通过得 (100/总数) 分
    """
    user = require_role(request, db, UserRole.student)

    # 空代码校验
    if not code or not code.strip():
        return render("student/answer.html", request=request, user=user,
                       problem=db.query(Problem).filter(Problem.id == problem_id).first(),
                       submissions=[], last_submission=None, last_test_results=[],
                       test_case_count=0, passed_cases=0, prev_id=None, next_id=None,
                       last_code=code, show_ref_answer=False, error="代码不能为空")

    problem = db.query(Problem).filter(Problem.id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="题目不存在")

    test_cases = problem.get_test_cases()

    # 执行判题
    result = CodeJudge.judge(code, language, test_cases)

    # 创建提交记录
    submission = Submission(
        problem_id=problem_id,
        user_id=user.id,
        code=code,
        language=language,
        compile_status=CompileStatus(result["compile_status"]),
        score=result["score"],
        compile_error=result.get("compile_error"),
    )
    if result.get("test_results"):
        submission.set_test_results(result["test_results"])

    # AI 代码评测（异步，失败不影响提交）
    ai_feedback = ""
    if ai_service.is_ai_available:
        try:
            ai_feedback = ai_service.review_code(
                title=problem.title,
                statement=problem.statement,
                student_code=code,
                test_results=result.get("test_results", []),
                score=result["score"],
            )
        except Exception:
            print(f"[AI] review_code 失败（不影响提交判分）: {e}")

    # 把 AI 反馈追加到 compile_error 字段（复用字段存 AI 反馈）
    if ai_feedback and result["compile_status"] == "compile_success":
        submission.compile_error = ai_feedback
    elif ai_feedback:
        submission.compile_error = ai_feedback
    elif result.get("compile_error"):
        submission.compile_error = result["compile_error"]

    db.add(submission)
    db.commit()
    db.refresh(submission)

    params = {
        "submitted": "1",
        "submission_id": str(submission.id),
        "last_code": code,
    }
    query_string = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    return RedirectResponse(
        f"/student/problem/{problem_id}?{query_string}",
        status_code=303,
    )


# ---------- 启动 ----------
if __name__ == "__main__":
    import sys
    import socket

    # 修复 Windows 控制台 GBK 编码无法输出 emoji 的问题
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # 获取本机局域网 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("114.114.114.114", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "无法获取"

    print("=" * 60)
    try:
        print("\U0001f42e  小牛客 - 在线编程评测系统 v8.0")
    except UnicodeEncodeError:
        print("[小牛客] 在线编程评测系统 v7.0")
    print("=" * 60)
    print(f"  本机访问: http://127.0.0.1:8000")
    if local_ip != "无法获取":
        print(f"  局域网访问: http://{local_ip}:8000")
    print("  测试教师账号: teacher1 / 123456")
    print("  测试学生账号: student1 / 123456")
    print("  教师验证码: Zmjjkk")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

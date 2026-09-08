"""
小牛客 - 数据库模型定义（整合版）
包含：用户表、题目表（含测试用例）、提交记录表
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Enum, Float, ForeignKey,
    create_engine, text as sa_text
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import enum
import json

DATABASE_URL = "sqlite:///./xiaoniuke.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------- 枚举类型 ----------
class UserRole(str, enum.Enum):
    teacher = "teacher"
    student = "student"


class Difficulty(str, enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ReviewStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class CompileStatus(str, enum.Enum):
    compile_success = "compile_success"
    compile_error = "compile_error"
    system_error = "system_error"


# ---------- 数据表 ----------
class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.student)
    session_token = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 个性化设置字段（全部可选，登录后自主设置）
    display_name = Column(String(100), nullable=True)   # 显示名称（可含中文，默认为 username）
    gender = Column(String(20), nullable=True)           # 性别：男/女/无（未设置）
    bio = Column(Text, nullable=True)                    # 个性签名
    avatar = Column(String(255), nullable=True)          # 头像路径：default/N.svg 或 uploaded/xxx.svg

    problems = relationship("Problem", back_populates="creator")
    submissions = relationship("Submission", back_populates="user")


class Problem(Base):
    """题目表 - 含测试用例"""
    __tablename__ = "problem"

    id = Column(Integer, primary_key=True, autoincrement=True)
    display_order = Column(Integer, default=0, index=True)  # 显示序号，删除时自动重排
    title = Column(String(200), nullable=False)
    statement = Column(Text, nullable=False)          # 题目描述
    constraints = Column(Text, nullable=True)          # 输入/输出约束
    tags = Column(String(500), nullable=True)          # 知识点标签，逗号分隔
    difficulty = Column(Enum(Difficulty), default=Difficulty.easy)
    standard_answer = Column(Text, nullable=True)      # AI / 教师提供的参考解答
    test_cases = Column(Text, nullable=True)           # JSON: [{"input":"...","output":"..."},...]
    source = Column(String(20), default="manual")      # manual / ai
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.pending)
    created_by = Column(Integer, ForeignKey("user.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    creator = relationship("User", back_populates="problems")
    submissions = relationship("Submission", back_populates="problem")

    def get_test_cases(self) -> list:
        """解析 JSON 格式的测试用例"""
        if not self.test_cases:
            return []
        try:
            return json.loads(self.test_cases)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_test_cases(self, cases: list):
        """保存测试用例为 JSON"""
        self.test_cases = json.dumps(cases, ensure_ascii=False)


class Submission(Base):
    __tablename__ = "submission"

    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(Integer, ForeignKey("problem.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    code = Column(Text, nullable=False)
    language = Column(String(20), nullable=False, default="python")
    compile_status = Column(Enum(CompileStatus), nullable=True)
    score = Column(Integer, default=0)
    compile_error = Column(Text, nullable=True)
    test_results = Column(Text, nullable=True)         # JSON: [{"case":1, "input":"...", "expected":"...", "actual":"...", "passed":bool}, ...]
    created_at = Column(DateTime, default=datetime.utcnow)

    problem = relationship("Problem", back_populates="submissions")
    user = relationship("User", back_populates="submissions")

    def get_test_results(self) -> list:
        """解析测试结果 JSON"""
        if not self.test_results:
            return []
        try:
            return json.loads(self.test_results)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_test_results(self, results: list):
        """保存测试结果为 JSON"""
        self.test_results = json.dumps(results, ensure_ascii=False)


# ---------- 工具函数 ----------
def renumber_problems(db):
    """重排所有题目的 display_order，确保连续无间隙
    按 id 升序排列，最早创建的题目获得序号 1
    """
    problems = db.query(Problem).filter(
        Problem.review_status == ReviewStatus.approved
    ).order_by(Problem.id.asc()).all()
    for i, p in enumerate(problems, 1):
        p.display_order = i
    db.commit()


def init_db():
    """创建所有表"""
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import inspect
    inspector = inspect(engine)
    # 如果 display_order 列尚不存在（旧数据库升级），尝试添加
    try:
        cols = [c['name'] for c in inspector.get_columns('problem')]
        if 'display_order' not in cols:
            with engine.connect() as conn:
                conn.execute(sa_text('ALTER TABLE problem ADD COLUMN display_order INTEGER DEFAULT 0'))
                conn.commit()
    except Exception:
        pass
    # 用户表个性化字段迁移（旧数据库升级）
    try:
        user_cols = [c['name'] for c in inspector.get_columns('user')]
        with engine.connect() as conn:
            if 'display_name' not in user_cols:
                conn.execute(sa_text("ALTER TABLE user ADD COLUMN display_name VARCHAR(100)"))
            if 'gender' not in user_cols:
                conn.execute(sa_text("ALTER TABLE user ADD COLUMN gender VARCHAR(20)"))
            if 'bio' not in user_cols:
                conn.execute(sa_text("ALTER TABLE user ADD COLUMN bio TEXT"))
            if 'avatar' not in user_cols:
                conn.execute(sa_text("ALTER TABLE user ADD COLUMN avatar VARCHAR(255)"))
            conn.commit()
    except Exception:
        pass


def migrate_constraints_to_statement():
    """将已有题目的约束条件合并到题目描述中（约束条件字段已在前端移除）"""
    db = SessionLocal()
    try:
        problems = db.query(Problem).filter(
            Problem.constraints.isnot(None),
            Problem.constraints != ""
        ).all()
        for p in problems:
            if p.constraints and p.constraints.strip():
                constraints_text = p.constraints.strip()
                # 检查是否已经在 statement 末尾包含约束内容
                if constraints_text not in p.statement:
                    p.statement += f"\n\n【约束条件】\n{constraints_text}"
                p.constraints = None  # 清空约束字段
        if problems:
            db.commit()
            print(f"[Migration] 已将 {len(problems)} 道题目的约束条件合并到题目描述")
    except Exception as e:
        db.rollback()
        print(f"[Migration] 约束迁移失败: {e}")
    finally:
        db.close()


def get_db():
    """获取数据库会话（FastAPI 依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_data():
    """插入演示数据"""
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            return

        import bcrypt

        # 创建测试用户
        teacher = User(
            username="teacher1",
            password_hash=bcrypt.hashpw("123456".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            role=UserRole.teacher,
        )
        student = User(
            username="student1",
            password_hash=bcrypt.hashpw("123456".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            role=UserRole.student,
        )
        db.add_all([teacher, student])
        db.flush()

        # 示例题目 1：两数之和
        p1 = Problem(
            title="两数之和",
            statement="给定两个整数 a 和 b，计算它们的和并输出结果。\n\n输入格式：一行两个整数，空格分隔。\n输出格式：一个整数，表示两数之和。",
            constraints="1 <= a, b <= 1000",
            tags="数学,基础",
            difficulty=Difficulty.easy,
            standard_answer="# 读取两个整数并输出它们的和\na, b = map(int, input().split())\nprint(a + b)",
            source="manual",
            review_status=ReviewStatus.approved,
            created_by=teacher.id,
        )
        p1.set_test_cases([
            {"input": "1 3", "output": "4"},
            {"input": "10 20", "output": "30"},
            {"input": "100 200", "output": "300"},
            {"input": "0 0", "output": "0"},
            {"input": "999 1", "output": "1000"},
            {"input": "500 500", "output": "1000"},
            {"input": "7 8", "output": "15"},
            {"input": "123 456", "output": "579"},
            {"input": "99 1", "output": "100"},
            {"input": "250 250", "output": "500"},
        ])

        # 示例题目 2：判断奇偶
        p2 = Problem(
            title="判断奇偶数",
            statement="给定一个整数 n，判断它是奇数还是偶数。\n\n输入格式：一行一个整数 n。\n输出格式：如果是偶数输出 \"Even\"，如果是奇数输出 \"Odd\"。",
            constraints="1 <= n <= 10^9",
            tags="数学,条件判断",
            difficulty=Difficulty.easy,
            standard_answer="# 判断奇偶数\nn = int(input())\nif n % 2 == 0:\n    print('Even')\nelse:\n    print('Odd')",
            source="manual",
            review_status=ReviewStatus.approved,
            created_by=teacher.id,
        )
        p2.set_test_cases([
            {"input": "2", "output": "Even"},
            {"input": "1", "output": "Odd"},
            {"input": "100", "output": "Even"},
            {"input": "99", "output": "Odd"},
            {"input": "0", "output": "Even"},
            {"input": "77777", "output": "Odd"},
            {"input": "88888", "output": "Even"},
            {"input": "13", "output": "Odd"},
            {"input": "24680", "output": "Even"},
            {"input": "13579", "output": "Odd"},
        ])

        # 示例题目 3：最大值
        p3 = Problem(
            title="三个数的最大值",
            statement="给定三个整数，找出其中最大的一个并输出。\n\n输入格式：一行三个整数，空格分隔。\n输出格式：一个整数，表示最大值。",
            constraints="1 <= a, b, c <= 10000",
            tags="数学,条件判断",
            difficulty=Difficulty.easy,
            standard_answer="# 找出三个数的最大值\na, b, c = map(int, input().split())\nprint(max(a, b, c))",
            source="manual",
            review_status=ReviewStatus.approved,
            created_by=teacher.id,
        )
        p3.set_test_cases([
            {"input": "1 2 3", "output": "3"},
            {"input": "10 5 8", "output": "10"},
            {"input": "100 200 150", "output": "200"},
            {"input": "7 7 7", "output": "7"},
            {"input": "999 1 500", "output": "999"},
            {"input": "3 9 6", "output": "9"},
            {"input": "1000 999 998", "output": "1000"},
            {"input": "42 100 57", "output": "100"},
            {"input": "88 66 99", "output": "99"},
            {"input": "1 10000 5000", "output": "10000"},
        ])

        db.add_all([p1, p2, p3])
        db.commit()

        # 设置显示序号
        renumber_problems(db)

        print("[OK] 演示数据已就绪 — 教师: teacher1/123456  学生: student1/123456")
    except Exception as e:
        db.rollback()
        print(f"[WARN] 种子数据失败: {e}")
    finally:
        db.close()

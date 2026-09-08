"""
小牛客 - 代码评判模块
判题流程：
  1. 语法检查 → 语法错误则 0 分
  2. 语法通过 → 用测试用例实际运行 → 每组通过得 (100/总组数) 分
"""
import ast
import py_compile
import subprocess
import tempfile
import os
import shutil
import time


class CodeJudge:
    """代码判题器"""

    @staticmethod
    def check_syntax(code: str, language: str = "python") -> dict:
        """
        语法/编译检查（含代码质量预检）
        返回: {"ok": bool, "error": str | None}
        """
        # 第一步：代码质量预检查（防止无意义的短文本）
        if language == "python":
            quality = CodeJudge._check_code_quality(code)
            if not quality["ok"]:
                return quality

        # 第二步：语法编译检查
        if language == "python":
            return CodeJudge._check_python_syntax(code)
        elif language == "cpp":
            return CodeJudge._check_cpp_syntax(code)
        else:
            return {"ok": False, "error": f"暂不支持的编程语言: {language}"}

    @staticmethod
    def _check_code_quality(code: str) -> dict:
        """
        代码质量预检查 —— 只拦截空代码和无意义内容
        """
        stripped = code.strip()

        # 1. 代码不能为空
        if not stripped:
            return {"ok": False, "error": "代码不能为空，请编写完整的 Python 程序"}

        # 2. 拒绝纯数字/纯字母等明显无意义的输入
        if stripped.isdigit() or stripped.isalpha():
            return {
                "ok": False,
                "error": f"代码无效：请输入 Python 代码，而非纯{'数字' if stripped.isdigit() else '字母'}"
            }

        return {"ok": True, "error": None}

    @staticmethod
    def _check_python_syntax(code: str) -> dict:
        """Python 语法检查"""
        try:
            # 方法1：ast.parse 检查语法树
            ast.parse(code)
            # 方法2：py_compile 编译检查（更严格）
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            )
            try:
                tmp.write(code)
                tmp.close()
                py_compile.compile(tmp.name, doraise=True)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            return {"ok": True, "error": None}
        except SyntaxError as e:
            return {
                "ok": False,
                "error": f"语法错误: 第 {e.lineno} 行, {e.msg}"
            }
        except py_compile.PyCompileError as e:
            return {
                "ok": False,
                "error": f"编译错误: {str(e)[:300]}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"语法检查异常: {str(e)[:300]}"
            }

    @staticmethod
    def _check_cpp_syntax(code: str) -> dict:
        """C++ 编译检查（需要 g++ 环境）"""
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="judge_")
            src_path = os.path.join(tmp_dir, "solution.cpp")
            exe_path = os.path.join(tmp_dir, "solution.exe")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)
            proc = subprocess.run(
                ["g++", src_path, "-o", exe_path, "-O2", "-std=c++17"],
                capture_output=True, text=True, timeout=15
            )
            if proc.returncode == 0:
                return {"ok": True, "error": None}
            else:
                return {
                    "ok": False,
                    "error": (proc.stderr or proc.stdout)[:500]
                }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "编译超时（超过15秒）"}
        except FileNotFoundError:
            return {"ok": False, "error": "g++ 编译器未安装"}
        except Exception as e:
            return {"ok": False, "error": f"编译异常: {str(e)[:300]}"}
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def run_test_cases(code: str, language: str, test_cases: list) -> dict:
        """
        运行测试用例
        test_cases: [{"input": "1 3", "output": "4"}, ...]
        返回: {
            "passed": int,
            "total": int,
            "score": int,
            "results": [{"case": 1, "input": "...", "expected": "...", "actual": "...", "passed": bool}, ...]
        }
        """
        if language == "python":
            return CodeJudge._run_python_test_cases(code, test_cases)
        else:
            return {
                "passed": 0,
                "total": len(test_cases),
                "score": 0,
                "results": [{
                    "case": i + 1,
                    "input": tc["input"],
                    "expected": tc["output"],
                    "actual": f"不支持的语言: {language}",
                    "passed": False,
                } for i, tc in enumerate(test_cases)],
                "error": f"暂不支持的编程语言: {language}"
            }

    @staticmethod
    def _run_python_test_cases(code: str, test_cases: list) -> dict:
        """用 subprocess 隔离运行学生代码，逐个测试"""
        results = []
        passed = 0
        total = len(test_cases)
        timeout_seconds = 5  # 每个测试用例最多运行 5 秒
        tmp_dir = None

        if total == 0:
            return {"passed": 0, "total": 0, "score": 100, "results": []}

        try:
            # 创建临时目录存放代码
            tmp_dir = tempfile.mkdtemp(prefix="judge_run_")
            code_path = os.path.join(tmp_dir, "solution.py")

            # 写入学生代码
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            for i, tc in enumerate(test_cases):
                case_num = i + 1
                input_data = tc.get("input", "")
                expected_output = tc.get("output", "").strip()

                try:
                    # 用 subprocess 运行，传递 stdin
                    proc = subprocess.run(
                        ["python", "-I", code_path],
                        input=input_data,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        cwd=tmp_dir,
                    )

                    # 检查运行时错误
                    if proc.returncode != 0:
                        stderr = (proc.stderr or "").strip()
                        results.append({
                            "case": case_num,
                            "input": input_data,
                            "expected": expected_output,
                            "actual": f"运行时错误: {stderr[:200]}",
                            "passed": False,
                        })
                        continue

                    actual_output = proc.stdout.strip()

                    # 比较输出（忽略末尾空白差异）
                    is_pass = (actual_output == expected_output)

                    if is_pass:
                        passed += 1

                    results.append({
                        "case": case_num,
                        "input": input_data,
                        "expected": expected_output,
                        "actual": actual_output,
                        "passed": is_pass,
                    })

                except subprocess.TimeoutExpired:
                    results.append({
                        "case": case_num,
                        "input": input_data,
                        "expected": expected_output,
                        "actual": f"运行超时（超过 {timeout_seconds} 秒）",
                        "passed": False,
                    })
                except Exception as e:
                    results.append({
                        "case": case_num,
                        "input": input_data,
                        "expected": expected_output,
                        "actual": f"运行异常: {str(e)[:200]}",
                        "passed": False,
                    })

        except Exception as e:
            # 整个判题过程出错
            for i, tc in enumerate(test_cases):
                results.append({
                    "case": i + 1,
                    "input": tc.get("input", ""),
                    "expected": tc.get("output", ""),
                    "actual": f"判题系统错误: {str(e)[:200]}",
                    "passed": False,
                })
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

        score = round(passed / total * 100) if total > 0 else 0
        return {
            "passed": passed,
            "total": total,
            "score": score,
            "results": results,
        }

    @staticmethod
    def judge(code: str, language: str, test_cases: list) -> dict:
        """
        完整判题流程：
        1. 语法检查
        2. 语法通过 → 运行测试用例
        返回完整的评判结果
        """
        # 第一步：语法检查
        syntax_result = CodeJudge.check_syntax(code, language)
        if not syntax_result["ok"]:
            return {
                "compile_status": "compile_error",
                "compile_error": syntax_result["error"],
                "score": 0,
                "test_results": [],
                "total_cases": len(test_cases) if test_cases else 0,
                "passed_cases": 0,
            }

        # 第二步：运行测试用例
        if not test_cases:
            return {
                "compile_status": "no_test_cases",
                "compile_error": "该题暂无测试用例，无法评测",
                "score": 0,
                "test_results": [],
                "total_cases": 0,
                "passed_cases": 0,
            }

        test_result = CodeJudge.run_test_cases(code, language, test_cases)
        return {
            "compile_status": "compile_success",
            "compile_error": None,
            "score": test_result["score"],
            "test_results": test_result["results"],
            "total_cases": test_result["total"],
            "passed_cases": test_result["passed"],
        }

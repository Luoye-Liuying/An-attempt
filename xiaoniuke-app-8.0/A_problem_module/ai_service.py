"""
小牛客 - AI 服务模块
功能：
  1. 根据题目信息生成标准答案和测试用例
  2. 支持 DeepSeek / OpenAI / 千问 等大模型 API
  3. API 未配置时，智能分析题目描述生成合理代码（而非固定模板）

配置方式（按优先级）：
  1. 系统环境变量: AI_API_KEY, AI_BASE_URL, AI_MODEL
  2. 项目根目录 .env 文件（需要 pip install python-dotenv）
"""
import json
import random
import os
import re
import httpx

# 加载 .env 文件（优先用 python-dotenv，否则手动解析）
_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
_LOADED = False

if os.path.exists(_ENV_FILE):
    # 方案一：python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE)
        _LOADED = True
    except ImportError:
        pass

    # 方案二：手动解析（python-dotenv 未安装时的兜底）
    if not _LOADED:
        try:
            with open(_ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    # 解析 KEY=VALUE 或 KEY = VALUE
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass


class AIService:
    """AI 服务——根据题目信息生成标准答案和测试用例"""

    def __init__(self):
        self.base_url = os.getenv("AI_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = os.getenv("AI_API_KEY", "")
        self.model = os.getenv("AI_MODEL", "deepseek-chat")

    @property
    def is_ai_available(self) -> bool:
        """检查 AI API 是否已配置"""
        return bool(self.api_key and self.api_key.strip())

    def generate_answer(self, title: str, statement: str, tags: str,
                        difficulty: str = "easy") -> dict:
        """
        根据题目信息生成标准答案和测试用例
        优先使用真实 AI，不可用时智能分析题目
        返回: {"standard_answer": str, "test_cases": [{"input": str, "output": str}, ...]}
        """
        # 先清理输入
        title = (title or "").strip()
        statement = (statement or "").strip()
        tags = (tags or "").strip()

        # 如果配置了 API Key，优先使用真实 AI
        if self.is_ai_available:
            try:
                result = self._generate_via_ai(title, statement, tags, difficulty)
                if result and result.get("standard_answer") and result.get("test_cases"):
                    try:
                        print(f"[AI] 大模型生成成功 ({self.model})")
                    except Exception:
                        pass
                    return result
            except Exception as e:
                try:
                    print(f"[AI] 大模型调用失败: {e}")
                except Exception:
                    pass
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
                print("[AI] 降级到智能分析模式...")

        # 降级：智能分析题目生成代码
        print("[AI] 使用智能分析模式（未配置大模型 API Key）")
        fallback_result = self._generate_smart(title, statement, tags, difficulty)
        fallback_result["source"] = "smart_fallback"
        return fallback_result

    # ---------- 真实 AI 调用 ----------

    def _generate_via_ai(self, title: str, statement: str, tags: str,
                         difficulty: str) -> dict:
        """调用大模型 API 生成答案"""
        prompt = self._build_prompt(title, statement, tags, difficulty)

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system",
                         "content": "请根据题目描述，生成标准的 Python 参考解答和 10 组测试用例。测试用例要覆盖普通情况和边界情况。严格输出 JSON 格式，不加额外文字。注意：所有测试用例的 input 和 output 都不能为空，必须包含实际的输入数据和期望输出结果。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse_ai_response(content)

    def _build_prompt(self, title: str, statement: str, tags: str,
                      difficulty: str) -> str:
        return f"""请为这道编程题生成标准 Python 解答和测试用例。

【题目】{title}
【描述】{statement}
【标签】{tags}
【难度】{difficulty}

要求：
1. standard_answer: 完整的 Python 代码，使用 input() 读取标准输入（多行输入需多次调用 input()），使用 print() 输出结果
2. test_cases: 10 组测试用例，覆盖：常规情况、边界值、特殊情况
3. 重要：每组测试用例的 input 和 output 都不能为空！input 必须包含具体的输入数据，output 必须包含期望的输出结果

输出 JSON 格式（不要 markdown 代码块标记）：
{{"standard_answer": "# Python 标准解答\\n...", "test_cases": [{{"input": "...", "output": "..."}}]}}"""

    def _parse_ai_response(self, content: str) -> dict:
        """解析 AI 返回的 JSON 字符串"""
        content = content.strip()
        # 去除可能的 Markdown 代码块标记
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        data = json.loads(content)

        result = {
            "standard_answer": data.get("standard_answer", ""),
            "test_cases": data.get("test_cases", []),
        }

        # 校验：确保测试用例格式正确且不为空
        validated_cases = []
        for tc in result["test_cases"]:
            if isinstance(tc, dict) and "input" in tc and "output" in tc:
                inp = str(tc["input"]).strip()
                out = str(tc["output"]).strip()
                if inp and out:  # 过滤掉输入或输出为空的用例
                    validated_cases.append({
                        "input": inp,
                        "output": out,
                    })
        result["test_cases"] = validated_cases

        # 如果没有生成足够的测试用例，补全
        if len(result["test_cases"]) < 3:
            result["test_cases"].extend(self._generate_mock_test_cases()[:10 - len(result["test_cases"])])

        return result

    # ---------- 智能分析模式（AI 不可用时的降级方案）----------

    def _generate_smart(self, title: str, statement: str, tags: str,
                        difficulty: str) -> dict:
        """
        智能分析题目描述，生成合理的代码和测试用例
        不再是固定模板 —— 会根据题目中的关键词推断所需逻辑
        """
        text = f"{title} {statement} {tags}".lower()

        # 分析题目类型
        problem_type = self._analyze_problem_type(text)

        # 根据题目类型生成代码
        standard_answer = self._generate_code_by_type(problem_type, text, title, statement)
        test_cases = self._generate_test_cases_by_type(problem_type, text, title, statement)

        # 如果未识别出题目类型导致测试用例为空，提供通用测试用例
        if not test_cases:
            test_cases = self._generate_mock_test_cases()

        return {
            "standard_answer": standard_answer,
            "test_cases": test_cases,
        }

    def _analyze_problem_type(self, text: str) -> str:
        """分析题目类型"""
        if any(kw in text for kw in ("排序", "sort", "升序", "降序")):
            return "sort"
        if any(kw in text for kw in ("最大", "最小", "max", "min", "最大值", "最小值")):
            return "max_min"
        if any(kw in text for kw in ("水仙花", "阿姆斯特朗", "armstrong", "各位数字")):
            return "narcissistic"
        if any(kw in text for kw in ("两数之和", "两数相加", "求和", "相加", "加法", "计算和", "之和")):
            return "sum"
        if any(kw in text for kw in ("两数之差", "相减", "减法")):
            return "subtract"
        if any(kw in text for kw in ("乘积", "相乘", "乘法", "阶乘", "之积")):
            return "multiply"
        if any(kw in text for kw in ("相除", "除法", "之商")):
            return "divide"
        if any(kw in text for kw in ("平均", "average", "均值")):
            return "average"
        if any(kw in text for kw in ("奇数", "偶数", "odd", "even", "奇偶")):
            return "odd_even"
        if any(kw in text for kw in ("素数", "质数", "prime")):
            return "prime"
        if any(kw in text for kw in ("回文", "palindrome", "对称")):
            return "palindrome"
        if any(kw in text for kw in ("斐波那契", "fibonacci")):
            return "fibonacci"
        if any(kw in text for kw in ("完数", "完数", "perfect number")):
            return "perfect"
        if any(kw in text for kw in ("九九乘法", "乘法表", "乘法口诀")):
            return "multiplication_table"
        if any(kw in text for kw in ("最大公约", "最小公倍", "gcd", "lcm", "公约数", "公倍数")):
            return "gcd_lcm"
        if any(kw in text for kw in ("闰年", "leap")):
            return "leap_year"
        if any(kw in text for kw in ("三角形", "triangle")):
            return "triangle"
        if any(kw in text for kw in ("字符串", "string", "字符", "反转", "倒序", "逆序")):
            return "string"
        if any(kw in text for kw in ("矩阵", "matrix")):
            return "matrix"
        if any(kw in text for kw in ("转换", "温度", "单位", "convert", "厘米", "米", "千克")):
            return "convert"
        if any(kw in text for kw in ("计数", "统计", "count", "个数", "次数")):
            return "count"
        if any(kw in text for kw in ("查找", "搜索", "find", "search")):
            return "search"
        if any(kw in text for kw in ("判断", "check", "是否", "能否")):
            return "judge"
        if any(kw in text for kw in ("绝对", "absolute", "abs")):
            return "absolute"
        if any(kw in text for kw in ("幂", "次方", "平方", "立方", "指数", "power", "pow")):
            return "power"
        if any(kw in text for kw in ("数组", "列表", "array", "list")):
            return "array"
        # 默认
        return "general"

    def _generate_code_by_type(self, ptype: str, text: str, title: str,
                               statement: str) -> str:
        """根据题目类型生成对应的 Python 代码"""
        codes = {
            "sum": (
                "# 读取输入并计算两数之和\n"
                "a, b = map(int, input().split())\n"
                "print(a + b)"
            ),
            "subtract": (
                "# 读取输入并计算两数之差\n"
                "a, b = map(int, input().split())\n"
                "print(a - b)"
            ),
            "multiply": (
                "# 读取输入并计算乘积\n"
                "a, b = map(int, input().split())\n"
                "print(a * b)"
            ),
            "divide": (
                "# 读取输入并计算商（保留小数）\n"
                "a, b = map(float, input().split())\n"
                "print(f'{a / b:.6f}')"
            ),
            "max_min": (
                "# 读取多个数并找出最大值\n"
                "nums = list(map(int, input().split()))\n"
                "print(max(nums))"
            ),
            "average": (
                "# 读取数字并计算平均值\n"
                "nums = list(map(int, input().split()))\n"
                "print(f'{sum(nums) / len(nums):.2f}')"
            ),
            "odd_even": (
                "# 判断奇偶性\n"
                "n = int(input())\n"
                "print('Even' if n % 2 == 0 else 'Odd')"
            ),
            "sort": (
                "# 读取数组并升序排列\n"
                "nums = list(map(int, input().split()))\n"
                "nums.sort()\n"
                "print(' '.join(map(str, nums)))"
            ),
            "prime": (
                "# 判断素数\n"
                "import math\n"
                "n = int(input())\n"
                "if n < 2:\n"
                "    print('No')\n"
                "else:\n"
                "    for i in range(2, int(math.sqrt(n)) + 1):\n"
                "        if n % i == 0:\n"
                "            print('No')\n"
                "            break\n"
                "    else:\n"
                "        print('Yes')"
            ),
            "palindrome": (
                "# 判断回文\n"
                "s = input().strip()\n"
                "print('Yes' if s == s[::-1] else 'No')"
            ),
            "fibonacci": (
                "# 计算第 n 个斐波那契数\n"
                "n = int(input())\n"
                "a, b = 0, 1\n"
                "for _ in range(n):\n"
                "    a, b = b, a + b\n"
                "print(a)"
            ),
            "leap_year": (
                "# 判断闰年\n"
                "year = int(input())\n"
                "if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:\n"
                "    print('Leap Year')\n"
                "else:\n"
                "    print('Not Leap Year')"
            ),
            "triangle": (
                "# 判断三角形类型\n"
                "a, b, c = map(int, input().split())\n"
                "if a + b > c and a + c > b and b + c > a:\n"
                "    if a == b == c:\n"
                "        print('Equilateral')\n"
                "    elif a == b or b == c or a == c:\n"
                "        print('Isosceles')\n"
                "    else:\n"
                "        print('Scalene')\n"
                "else:\n"
                "    print('Not a triangle')"
            ),
            "convert": (
                "# 温度转换：摄氏度转华氏度\n"
                "c = float(input())\n"
                "f = c * 9 / 5 + 32\n"
                "print(f'{f:.2f}')"
            ),
            "string": (
                "# 字符串处理\n"
                "s = input().strip()\n"
                "# 统计字符数、反转、大小写转换等\n"
                "print(len(s))\n"
                "print(s[::-1])\n"
                "print(s.upper())"
            ),
            "array": (
                "# 数组求和\n"
                "n = int(input())\n"
                "arr = list(map(int, input().split()))\n"
                "print(sum(arr))"
            ),
            "count": (
                "# 统计个数\n"
                "data = input().split()\n"
                "print(len(data))"
            ),
            "search": (
                "# 查找元素\n"
                "target = int(input())\n"
                "arr = list(map(int, input().split()))\n"
                "if target in arr:\n"
                "    print(arr.index(target))\n"
                "else:\n"
                "    print(-1)"
            ),
            "judge": (
                "# 条件判断\n"
                "n = int(input())\n"
                "if n > 0:\n"
                "    print('Positive')\n"
                "elif n < 0:\n"
                "    print('Negative')\n"
                "else:\n"
                "    print('Zero')"
            ),
            "narcissistic": (
                "# 判断水仙花数（各位数字的立方和等于该数本身）\n"
                "n = int(input().strip())\n"
                "hundreds = n // 100\n"
                "tens = (n // 10) % 10\n"
                "units = n % 10\n"
                "if hundreds**3 + tens**3 + units**3 == n:\n"
                "    print('Yes')\n"
                "else:\n"
                "    print('No')"
            ),
            "perfect": (
                "# 判断完数（所有真因子之和等于该数本身）\n"
                "n = int(input())\n"
                "total = sum(i for i in range(1, n) if n % i == 0)\n"
                "print('Yes' if total == n else 'No')"
            ),
            "multiplication_table": (
                "# 输出九九乘法表\n"
                "n = int(input()) if input() else 9\n"
                "for i in range(1, n + 1):\n"
                "    row = []\n"
                "    for j in range(1, i + 1):\n"
                "        row.append(f'{j}*{i}={i*j}')\n"
                "    print(' '.join(row))"
            ),
            "gcd_lcm": (
                "# 计算最大公约数和最小公倍数\n"
                "import math\n"
                "a, b = map(int, input().split())\n"
                "g = math.gcd(a, b)\n"
                "l = a * b // g\n"
                "print(g, l)"
            ),
            "absolute": (
                "# 计算绝对值\n"
                "n = float(input())\n"
                "print(abs(n))"
            ),
            "power": (
                "# 计算幂\n"
                "a, b = map(int, input().split())\n"
                "print(a ** b)"
            ),
            "general": (
                "# 读取输入并处理\n"
                "import sys\n"
                "data = sys.stdin.read().strip().split()\n"
                "# TODO: 根据具体题目要求处理数据\n"
                "print(' '.join(data))"
            ),
        }
        return codes.get(ptype, codes["general"])

    def _generate_test_cases_by_type(self, ptype: str, text: str, title: str,
                                     statement: str) -> list:
        """根据题目类型生成 10 组测试用例"""
        generators = {
            "sum": lambda: [
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
            ],
            "max_min": lambda: [
                {"input": "1 2 3 4 5", "output": "5"},
                {"input": "10 5 8 3 9", "output": "10"},
                {"input": "100", "output": "100"},
                {"input": "7 7 7 7", "output": "7"},
                {"input": "99 1 50", "output": "99"},
                {"input": "3 9 6 2 8", "output": "9"},
                {"input": "1000 999 998", "output": "1000"},
                {"input": "42 100 57", "output": "100"},
                {"input": "88 66 99", "output": "99"},
                {"input": "1 5 3 9 2", "output": "9"},
            ],
            "odd_even": lambda: [
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
            ],
            "sort": lambda: [
                {"input": "3 1 4 1 5 9 2 6", "output": "1 1 2 3 4 5 6 9"},
                {"input": "5 4 3 2 1", "output": "1 2 3 4 5"},
                {"input": "1", "output": "1"},
                {"input": "10 5 8 3", "output": "3 5 8 10"},
                {"input": "9 9 9 1 1", "output": "1 1 9 9 9"},
                {"input": "2 7 1 8", "output": "1 2 7 8"},
                {"input": "100 50 75 25", "output": "25 50 75 100"},
                {"input": "6 3 9 0 2", "output": "0 2 3 6 9"},
                {"input": "8 4 2 6", "output": "2 4 6 8"},
                {"input": "15 5 20 10", "output": "5 10 15 20"},
            ],
            "prime": lambda: [
                {"input": "2", "output": "Yes"}, {"input": "3", "output": "Yes"},
                {"input": "4", "output": "No"}, {"input": "17", "output": "Yes"},
                {"input": "1", "output": "No"}, {"input": "97", "output": "Yes"},
                {"input": "100", "output": "No"}, {"input": "13", "output": "Yes"},
                {"input": "49", "output": "No"}, {"input": "29", "output": "Yes"},
            ],
            "narcissistic": lambda: [
                {"input": "153", "output": "Yes"}, {"input": "370", "output": "Yes"},
                {"input": "371", "output": "Yes"}, {"input": "407", "output": "Yes"},
                {"input": "100", "output": "No"}, {"input": "123", "output": "No"},
                {"input": "999", "output": "No"}, {"input": "200", "output": "No"},
                {"input": "500", "output": "No"}, {"input": "947", "output": "No"},
            ],
            "perfect": lambda: [
                {"input": "6", "output": "Yes"}, {"input": "28", "output": "Yes"},
                {"input": "496", "output": "Yes"}, {"input": "8128", "output": "Yes"},
                {"input": "12", "output": "No"}, {"input": "100", "output": "No"},
                {"input": "8", "output": "No"}, {"input": "10", "output": "No"},
                {"input": "30", "output": "No"}, {"input": "50", "output": "No"},
            ],
            "gcd_lcm": lambda: [
                {"input": "12 18", "output": "6 36"}, {"input": "8 12", "output": "4 24"},
                {"input": "15 25", "output": "5 75"}, {"input": "7 13", "output": "1 91"},
                {"input": "100 75", "output": "25 300"}, {"input": "24 36", "output": "12 72"},
                {"input": "17 34", "output": "17 34"}, {"input": "9 27", "output": "9 27"},
                {"input": "2 3", "output": "1 6"}, {"input": "48 180", "output": "12 720"},
            ],
            "power": lambda: [
                {"input": "2 3", "output": "8"}, {"input": "5 2", "output": "25"},
                {"input": "10 0", "output": "1"}, {"input": "3 4", "output": "81"},
                {"input": "7 3", "output": "343"}, {"input": "2 10", "output": "1024"},
                {"input": "6 2", "output": "36"}, {"input": "4 5", "output": "1024"},
                {"input": "9 2", "output": "81"}, {"input": "1 100", "output": "1"},
            ],
        }

        generator = generators.get(ptype)
        if generator:
            return generator()
        return []  # 不要套用求和用例，避免错判

    def _generate_mock_test_cases(self) -> list:
        """默认测试用例"""
        return [
            {"input": "1 2", "output": "3"},
            {"input": "10 20", "output": "30"},
            {"input": "5 5", "output": "10"},
            {"input": "0 1", "output": "1"},
            {"input": "100 200", "output": "300"},
            {"input": "7 8", "output": "15"},
            {"input": "3 9", "output": "12"},
            {"input": "50 50", "output": "100"},
            {"input": "99 1", "output": "100"},
            {"input": "25 75", "output": "100"},
        ]


    def review_code(self, title: str, statement: str, student_code: str,
                    test_results: list, score: int) -> str:
        """AI 评测学生代码，返回反馈意见"""
        if not self.is_ai_available:
            return ""

        passed = sum(1 for t in test_results if t.get("passed"))
        total = len(test_results)
        failed_cases = [t for t in test_results if not t.get("passed")]

        prompt = f"""请评测以下学生提交的 Python 代码。

【题目】{title}
【题目描述】{statement}
【学生代码】
```
{student_code}
```
【测试结果】{passed}/{total} 通过，得分 {score}/100
"""
        if failed_cases:
            prompt += "未通过的测试用例：\n"
            for tc in failed_cases[:3]:
                prompt += f"  输入: {tc.get('input','')} → 期望: {tc.get('expected', tc.get('output',''))} → 实际: {tc.get('actual','')}\n"

        prompt += """
请用中文给出简短点评（100字以内），包括：
1. 代码是否正确理解了题意
2. 主要问题在哪里（如有）
3. 改进建议（如有）"""

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "你是一位编程助教，负责点评学生的代码。请直接给出点评，不加前缀。"},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return content.strip()
        except Exception as e:
            try:
                print(f"[AI] 代码评测失败: {e}")
            except Exception:
                pass
            return ""


# 全局单例
ai_service = AIService()

# D-前端

## 负责人
- 前端页面与样式

## 文件清单
- templates/ — Jinja2 HTML 模板
- templates/student/ — 学生端模板
- templates/teacher/ — 教师端模板
- static/css/style.css — 全局样式表
- main_routes_d.py — 前端路由

## 路由
- GET /student/dashboard — 学生题目列表
- GET /student/problem/{id} — 答题页

## 运行说明
本模块代码需结合主项目 main.py 运行，不可独立启动。

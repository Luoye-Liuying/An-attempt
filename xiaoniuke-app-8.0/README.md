# E-测试运维

## 负责人
- 测试、部署、环境配置

## 文件清单
- requirements.txt — 项目依赖清单
- .env — 环境变量配置
- xiaoniuke.db — SQLite 项目数据库

## 职责
- 编写测试用例与边界场景
- 搭建 Docker 部署环境
- 准备演示数据
- 回归测试与日志收集

## 启动项目
cd xiaoniuke.app
uvicorn main:app --reload

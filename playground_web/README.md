# Nimble 多模态 SOTA 交互体验台 (`playground_web`)

本目录提取自 `100.80.18.39:/root/projects/jev-real-ad-1000/web`（原版备份于 `playground_web/legacy/`），并采用 **React 18 + Material UI (`@mui/material` v5)** 重构为支持**多方法共享底座热切换与同题横评**的交互体验台。

## 目录结构
- `ui/`：React 18 + `@mui/material` + `@mui/icons-material` 前端源码（`src/App.jsx`），通过 `esbuild` 亚秒级打包至 `static/app.js`（零外部 CDN 依赖，支持浅色/深色主题切换）。
- `static/`：编译后的生产静态资源（由 `server.py` 动态直出，修改前端打包后无需重启 GPU 服务即可生效）。
- `predictors.py`：`SharedBasePool` 多 LoRA 热切换引擎（单张 GPU 常驻 10 个模型，切换耗时 `<0.5ms`）。
- `server.py`：FastAPI 服务端（兼容 `/v1/systemone`，新增 `/api/compare` 全方法横评与 `/api/prompt` 底层 Token 流透视）。
- `legacy/`：服务器原版代码完整归档。

## 前端构建与热更新
```bash
cd playground_web/ui
npm run build   # 约 0.5 秒生成 ../static/app.js
```

## 启动方式（在 `100.80.18.39` GPU 服务器上）
```bash
GPU=0 PORT=23834 bash playground_web/run_playground.sh
```
启动后在本地浏览器直接访问：`http://100.80.18.39:23834`（保留原 `23833` 旧版服务互不干扰）。

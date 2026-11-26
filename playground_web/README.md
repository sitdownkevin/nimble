# Nimble 多模态 SOTA 交互体验台 (`playground_web`)

本目录提取自 `100.80.18.39:/root/projects/jev-real-ad-1000/web`（原版备份于 `playground_web/legacy/`），并升级为支持**多方法共享底座热切换与同题横评**的交互体验台。

## 核心体验能力
1. **单卡多方法零开销热切换（最多同时在线 10 种方法）**：
   - `nimble-4b-mm-sota`：`4B` 多模态 SOTA（`sandwich_v3` + `visual.merger` LoRA + `FP32` 候选头 + `EMA` + 零泄漏锁温）
   - `nimble-4b-mm-lora-all`：`4B` 全视觉塔 LoRA（单步 Test `84.00%`）
   - `nimble-4b-mm-frozen`：`4B` 冻结视觉塔（全场最佳概率校准 `NLL = 0.4080`）
   - `nimble-4b-text-sota`：`4B` 纯文本对照（直观感受剥离图像后的盲区）
   - `nimble-0.8b-mm-sota`：`0.8B` 多模态 SOTA（单步 Test `79.33%`）
   - `nimble-0.8b-rawad-mm`：`0.8B` 纯原图无 OCR 训练模型（专测直接读图识字能力）
   - `nimble-0.8b-legacy-v1` / `nimble-0.8b-text-legacy-v1`：服务器原版 `v1` 双重转义 JSON 基线
   - `kev-4b-mm` / `kev-0.8b-mm`：对照 `Kev` 架构（FIM + PointerHead）
2. **⚡ 一键横评全部方法 (`/api/compare`)**：在同一条真实广告或自定义上传图片上一次性跑完所有方法，同屏对比预测正误、Top-1 置信度、真实答案概率、Token 长度与 GPU 延迟。
3. **✂️ 一键剥离 OCR/Caption**：在界面顶栏一键删除预提取的 `visual_ocr / image_caption / visual_risk_cues`，直观体验纯文本模型失明与多模态 SOTA 直接读像素识字分类的反差。
4. **Prompt & Token 透视**：直观对比 `sandwich_v3`（`Task & Policy -> Image + Clean Text -> Choices`，Token 缩短 `18.9%`）与旧版 `v1` 的底层编码结构。

## 启动方式（在 `100.80.18.39` GPU 服务器上）
```bash
GPU=0 PORT=23834 bash playground_web/run_playground.sh
```
启动后在本地浏览器直接访问：`http://100.80.18.39:23834`（保留原 `23833` 旧版服务互不干扰，也可随时双开对比）。

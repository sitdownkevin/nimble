# 图像在 state 中的位置对准确率的影响

## 设定
- 数据：jev-real-ad-1000（train 700 / val 150 / test 150），三种位置变体见 `kev_data/*_pos_{start,middle,end}.jsonl`，由 `scripts/make_position_variants.py` 生成。
  - start：image 放在文本字段之前（之前所有多模态模型的训练布局）
  - middle：image 放在文本字段的中间（前一半字段之后）
  - end：image 放在文本字段之后
- 位置语义：image 键在 state 对象中的位置即为图片块在 state 中的位置（`kev/api.py::image_offset`，`kev/model.py::state_pieces`）。
- 训练：从 Qwen3.5-0.8B-Base 起，多模态，3 epoch，lr 1e-4，batch 1，accum 8，max_state 1024，每 20 步在对应位置的 val 上评测。
- 每个位置 2 个随机种子（seed 0/1）。start seed 0 即 `runs/mm-all-700`。

## 结果（准确率，150 条）
| 位置 | val seed0 | val seed1 | val 均值 | test seed0 | test seed1 | test 均值 |
|---|---|---|---|---|---|---|
| start | 0.720 | 0.707 | 0.713 | 0.747 | 0.767 | 0.757 |
| middle | 0.700 | 0.700 | 0.700 | 0.747 | 0.747 | 0.747 |
| end | 0.707 | 0.713 | 0.710 | 0.767 | 0.760 | 0.763 |

零样本位置迁移（只在 start 下训练的 seed0 模型，测试集换成 middle / end 位置）：test 0.747 → 0.760 / 0.760。

## 结论
- 位置对准确率没有可测的影响：三种位置的差距（约 1–2 个百分点）小于同一位置两个种子之间的差（约 2 个百分点），也小于 150 条测试集的标准误（约 3.5 个百分点）。
- middle 的均值略低，但不显著，不足以下结论。

## 验证
- 新代码对 start 布局的编码与原实现逐位一致：`runs/test-pos-start-of-mm-all-700` 的预测与旧的 `runs/test-mm-all-700` 完全相同（最大 |Δp| = 0）。
- 多模态 HF 对齐测试（`tests/test_multimodal.py`）覆盖了 image 在中间的情况，13/13 通过。
- playground 的 `/v1/systemone` 在 image 开头和结尾两种情况下，与对应 benchmark 的预测一致（最大 |Δ| 5e-5，四位小数舍入）。

## 注意
- 旧的 `kev_data/*_mm.jsonl` 中 image 键排在最后，按新规则会被当成“结尾”；要复现之前的开头布局请用 `*_pos_start.jsonl`。

# ChartNet summary：Qwen2.5-VL Adapter 微调计划

目标是让 Qwen2.5-VL-7B-Instruct 更好地理解图表。输入只有图像，监督目标是 ChartNet `human_verified` 的 `summary`。训练全部在 `adapter1.ipynb` 里进行。

## 1. 实现方法

服务器不能访问公网，不安装新包。使用 notebook 内核里已经存在的包：

| 用途 | 包 | 版本 | 位置 |
| --- | --- | --- | --- |
| 读 Arrow、导出 jsonl | datasets | 3.6.0 | notebook 内核 |
| Adapter 微调、模板、loss、保存 | ms-swift | 4.2.0.dev0 | `/home/hadoop-risk-control-algo/dolphinfs_hdd_hadoop-risk-control-algo/prime/zhangyue230/ms-swift` |
| 模型加载 | transformers | 4.57.3 | notebook 内核 |
| 画图 | matplotlib | 3.10.8 | notebook 内核 |
| 日志 | tensorboard | 2.19.0 | notebook 内核 |

数据处理用 datasets，训练用 ms-swift。样本落在磁盘上的是 Arrow（`save_to_disk`），图像在 `image` 列里；swift 的 `sft_main` 读的是 jsonl，每行要有 `messages` 和图片路径。所以先用 datasets 读出 Arrow，再写成 swift 能直接训练的 jsonl。

微调方式是 ms-swift 的 `tuner_type=adapter`，也就是 Houlsby et al. 2019（arXiv:1902.00751）的 bottleneck：降维、GELU、升维，输出加回残差。这一版 swift 把它放在每个 Transformer 层的 MLP 输出上（Pfeiffer 放置），注意力子层后面没有第二个 adapter。这和 LoRA 不同：LoRA 是在冻结权重旁边加低秩增量，不插入新层。

这个版本对 Qwen2.5-VL 不能直接用默认的 `prepare_adapter`：

- `qwen2_vl` 的模型结构里没有 `mlp` 字段，默认逻辑会在取模块名时失败。
- 视觉塔 MLP 宽度是 1280，语言模型隐藏维是 3584。同一个 `dim=3584` 的 adapter 不能挂到视觉 MLP 上。

notebook 里只在当前进程改两处，不改共享的 ms-swift 源码：

- 匹配范围收成 `model.language_model.layers.<层号>.mlp`，28 层语言模型 MLP。
- 升维矩阵 `linear2` 的权重和 bias 置零。Houlsby 的未训练 adapter 应接近恒等映射；这一版源码对两个线性层都做了 Xavier 初始化。

视觉塔保持冻结。可训练参数约为 \(28 \times 2 \times 3584 \times 128 \approx 2570\) 万，约占 8.29B 的 0.3%。

模型已经在 notebook 里用 `device_map=auto` 摊到 8 张 GPU 上。swift 发现多卡后也会自己按 `device_map=auto` 再加载一份。训练前要重启内核，释放已经加载的 `base_model`，并且不要在同一内核里再跑手动 `from_pretrained`。本机 shell 看不到 GPU，训练由 notebook 内核启动。DeepSpeed、FSDP 与这种单进程分卡加载不能同时开。

## 2. 数据

样本在 `data/chartnet_sample`，来自 `ibm-granite/ChartNet` 的 `human_verified`：train 4000 条，test 1000 条。列有 `id`、`image`、`code`、`csv`、`summary`。本次只用 `image` 和 `summary`；`id` 只用来给导出的 PNG 命名。`code` 和 `csv` 不进入 prompt，也不作为辅助 loss。

导出目录是 `data/chartnet_swift`：

- `images/train/<id>.png`、`images/test/<id>.png`
- `train.jsonl`：全部 4000 条
- `eval.jsonl`：test 经 `shuffle(seed=42)` 后的前 450 条

summary 是英文，所以 system 和 user 都用英文，并且每条样本用同一套提示词。jsonl 里显式写入 system。若省略，ms-swift 的 `qwen2_5_vl` 模板会自动补上默认句 `You are a helpful assistant.`，那句话没有图表任务约束。

### 训练样本

一条样本三个角色，单轮，没有历史对话，没有 tools。`code`、`csv`、`id` 都不出现在文本里。

**system**

```text
You are a chart understanding assistant. Read the single chart in the image and write a detailed English summary. Describe the chart type, title, axes, legends, and the trends, comparisons, or distributions that are visible. Use only what the image shows. Do not output code, CSV, or a data table.
```

**user**

```text
<image>
Write a detailed summary of this chart.
```

`<image>` 单独占一行，后面是固定指令。它是 swift 的图像占位符，编码时换成 `<|vision_start|><|image_pad|><|vision_end|>`。`images` 里只有这一张图的路径。

**assistant**

ChartNet 的 `summary` 原文，不改写、不截断、不包在 JSON 或代码块里。

jsonl 一行：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a chart understanding assistant. Read the single chart in the image and write a detailed English summary. Describe the chart type, title, axes, legends, and the trends, comparisons, or distributions that are visible. Use only what the image shows. Do not output code, CSV, or a data table."
    },
    {
      "role": "user",
      "content": "<image>\nWrite a detailed summary of this chart."
    },
    {
      "role": "assistant",
      "content": "<summary 原文>"
    }
  ],
  "images": ["<png 绝对路径>"]
}
```

### 送进模型的完整序列

swift 用 ChatML 拼成下面这一段。`<|image_pad|>` 会按图像尺寸展开成多个视觉 token。

```text
<|im_start|>system
You are a chart understanding assistant. Read the single chart in the image and write a detailed English summary. Describe the chart type, title, axes, legends, and the trends, comparisons, or distributions that are visible. Use only what the image shows. Do not output code, CSV, or a data table.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Write a detailed summary of this chart.<|im_end|>
<|im_start|>assistant
<summary 原文><|im_end|>
```

loss 只落在 assistant 正文和结尾的 `<|im_end|>` 上。system、user、图像 token 的 label 为 -100。推理时序列停在 `<|im_start|>assistant\n`，由模型接着生成 summary。

已经存在的 PNG 会跳过，jsonl 每次按这个模板重写。

图像在 checkpoint 里默认 `max_pixels=12845056`，大约 1.6 万个视觉 token，会撑爆显存。这里改成 `1003520`（约 1280 个视觉 token，每个 token 对应 28×28 像素），`min_pixels=3136`。两处都要设：`SftArguments.max_pixels`，以及环境变量 `MAX_PIXELS` / `MIN_PIXELS`（`qwen_vl_utils` 读环境变量）。序列 `max_length=3072`。

## 3. 训练

入口是 `SftArguments` + `swift.pipelines.train.sft.sft_main`。loss 用 swift 的 `loss_scale=default`：因果语言模型交叉熵，只计算 assistant 的 summary，图像 token 和 user 文本不算 loss。

模型路径：

`/mnt/dolphinfs/hdd_pool/docker/user/hadoop-risk-control-algo/yinyanling02/code/llm-qwen/Qwen2.5-VL-main/Qwen/Qwen2.5-VL-7B-Instruct`

输出目录：

`outputs/chartnet_summary_adapter`（`add_version=False`，目录名固定）

| 项 | 值 |
| --- | --- |
| tuner | adapter，长度 128，激活 GELU |
| 精度 | bfloat16 |
| batch | 每卡 1，梯度累积 8，有效 batch 8 |
| epoch | 2（4000 / 8 × 2 = 1000 optimizer step） |
| 优化器 | AdamW，学习率 1e-4，weight decay 0.01，梯度裁剪 1.0 |
| 调度 | cosine，warmup 比例 0.03 |
| 梯度检查点 | 开，`use_cache` 由 swift 按训练需要关闭 |
| 日志 | 每 10 step |
| 验证 | 每 100 step，450 条 test 上的 teacher-forcing eval loss |
| 保存 | 每 200 step，只保留 2 个 checkpoint，只保存 adapter |
| 随机种子 | 42 |
| dataloader | workers 2，pin memory |

`split_dataset_ratio` 保持 0，验证集用单独的 `val_dataset`。

## 4. 日志和作图

训练过程写入 TensorBoard（`report_to=["tensorboard"]`）。swift 结束时只会从 TensorBoard 里画出 `train/loss`。

`sft_main` 返回后，notebook 再读 `trainer_state.json` 的 `log_history`，用 Matplotlib 另存三张图到 `outputs/chartnet_summary_adapter/images`：

- `train_loss.png`：训练 loss 对 step
- `eval_loss.png`：450 条 test 的 eval loss 对 step
- `learning_rate.png`：学习率对 step

训练中的指标是 loss，不另装评测包。训练结束后如需看生成质量，再从 test 里抽少量样本，记录 `id`、模型输出和 `summary`。

## 运行顺序

`adapter1.ipynb` 末尾三个单元依次是导出、训练、画图。前面的单元只做环境检查和整模加载。

1. 先跑导出单元，写出 `train.jsonl`（4000 行）和 `eval.jsonl`（450 行）。这一步只写磁盘。
2. 重启内核，释放前面单元已经加载的 7B。
3. 不要重跑手动 `from_pretrained` 那个单元。
4. 运行训练单元。训练由 notebook 内核占 GPU，预计约 1000 step。
5. 训练结束后运行画图单元。

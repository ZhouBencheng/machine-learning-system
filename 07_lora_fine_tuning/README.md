# LoRA 模型微调实验：在昇腾 NPU 上用 ms-swift 完成 Qwen3-0.6B 训练与 loss 分析

## 实验简介

本实验是第三章 LoRA 微调内容的配套实践。Notebook 以 `Qwen/Qwen3-0.6B` 为例，在带 Ascend/CANN 运行时的 ModelArts 环境中完成环境检查、数据准备、LoRA 参数配置和训练日志分析。

实验分三个 notebook。第一个只做导学和预检，不启动训练；第二个用 3 个 step 检查命令、NPU、数据和日志是否接通；第三个训练 1 个 epoch，从原始 `logging.jsonl` 读取 loss，并按日志中的字段绘图。

| Notebook | 内容 | 是否启动训练 |
|----------|------|--------------|
| `07.01_lora_experiment_introduction.ipynb` | 导学：环境检查、模型和数据准备、LoRA 参数、证据记录和阅读边界 | 否 |
| `07.02_ms_swift_smoke_test.ipynb` | 3 步 smoke test：生成数据、构造 `swift sft` 命令、检查 loss 日志和 checkpoint | 是，3 步 |
| `07.03_ms_swift_training_and_loss_analysis.ipynb` | 1 个 epoch 的 LoRA 训练、原始 loss 曲线、日志中的内存字段检查和训练后回顾 | 是，1 个 epoch |

这里的 smoke test 只回答“流程能不能跑通”。它不能代替完整训练，也不能用一个 loss 数值说明模型效果。第三个 notebook 画的是当前运行的原始日志；没有记录到 `memory(GiB)` 时，不会从设备规格或 loss 反推显存。

## 实验环境与资源配置

实验在华为云 ModelArts 的 JupyterLab 中进行。需要准备一台带 Ascend NPU 和 CANN 运行时的环境，Notebook 默认使用一张可见 NPU。

| 项目 | 要求 |
|------|------|
| Python | 3.10、3.11 或 3.12 |
| 运行时 | 能导入 `torch_npu`，并且 `torch` 与 `torch_npu` 的主次版本一致 |
| Python 依赖 | `modelscope`、`ms-swift`、`matplotlib`；Notebook 会检查并安装缺失项 |
| 模型 | `Qwen/Qwen3-0.6B`，也可以通过环境变量替换模型路径 |
| 训练设备 | 默认使用 `ASCEND_RT_VISIBLE_DEVICES=0`，默认进程数为 1 |
| 网络和磁盘 | 需要下载模型和 Python 依赖，并保存模型缓存、日志、checkpoint 和曲线 |

`torch_npu` 和 CANN 属于运行时组件，不能靠在 Notebook 里随便安装一个 pip 包来修复版本不匹配。创建环境后，先运行 Notebook 的初始化代码，确认 NPU 数量、PyTorch 版本、`torch_npu` 版本和实际模型路径。

Notebook 支持用环境变量调整路径：

| 环境变量 | 用途 |
|----------|------|
| `L07_WORK_DIR` | 工作目录，默认在当前目录下创建对应的 workspace |
| `L07_MODEL_ID` | ModelScope 模型 ID，默认是 `Qwen/Qwen3-0.6B` |
| `L07_MODEL_PATH` | 已下载模型的本地目录；设置后不再重复下载 |
| `MODELSCOPE_CACHE` | ModelScope 缓存目录 |
| `L07_NPROC_PER_NODE` | L07-03 的训练进程数，默认是 1；改成多卡前先确认启动方式和设备配置 |

## 创建实验环境

1. 在 ModelArts 中创建带 Ascend/CANN 运行时的 Notebook 实例，规格至少要有一张可用 NPU。
2. 进入 JupyterLab，把本目录的三个 notebook 上传到同一个工作目录。
3. 先运行 `07.01_lora_experiment_introduction.ipynb`，记录 Python、PyTorch、`torch_npu`、NPU 数量、模型路径和训练数据路径。
4. 再运行 `07.02_ms_swift_smoke_test.ipynb`。它会在自己的 workspace 中生成 JSONL 数据，不读取上一个 notebook 的 Python 变量。
5. smoke test 通过后运行 `07.03_ms_swift_training_and_loss_analysis.ipynb`。这个 notebook 也会重新初始化环境和数据，可以单独复制到新的实例运行。

每个 notebook 都从第一格开始执行。不要把本地绝对路径、旧的 checkpoint 或上一次运行的 loss 手工填进新的实验记录。

## 实验流程

### 导学 notebook

`07.01` 做四件事：

- 检查 Python、PyTorch、`torch_npu` 和 NPU；
- 下载模型并生成一份小型对话式 JSONL 训练集；
- 解释 `lora_rank`、`lora_alpha`、`target_modules`、学习率、batch size 和最大序列长度；
- 记录模型、数据、配置、命令、日志和 checkpoint 等可追溯证据。

这一步不启动 `swift sft`。运行前想清楚 smoke test 能证明什么，以及一条 loss 曲线不能证明什么。

### 3 步 smoke test

`07.02` 使用 `ms-swift` 的 `swift sft` 命令，默认配置如下：

| 参数 | 值 |
|------|----|
| `tuner_type` | `lora` |
| `target_modules` | `all-linear` |
| `lora_rank` | 8 |
| `lora_alpha` | 32 |
| `learning_rate` | `1e-4` |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 1 |
| `max_length` | 512 |
| `torch_dtype` | `bfloat16` |
| `max_steps` | 3 |

运行时看 step 是否推进，进程是否正常退出，以及是否出现 OOM、NaN 或其他异常。Notebook 会保存 `notebook_stdout.log`，并检查运行目录下是否有包含有限 loss 的 `logging.jsonl` 和 checkpoint。

### 1 个 epoch 的训练与分析

`07.03` 沿用上一节的 LoRA 参数，默认训练 1 个 epoch。`logging_steps=1`，`save_steps=4`，训练进程数默认为 1。训练输出写入 `notebook_stdout.log`，完成后代码会：

1. 找到当前运行产生的 `logging.jsonl`；
2. 读取 `step` 以及 `loss` 或 `train_loss` 字段；
3. 保留原始 loss 点，并可选绘制移动平均线；
4. 只有在日志含有 `memory(GiB)` 字段时才绘制内存曲线；
5. 检查 checkpoint / adapter 路径，并提示需要在报告中记录的证据。

写分析时把日志事实和解释分开。loss 的整体趋势、波动和异常点可以直接从日志或图中读取；学习率、数据顺序、batch size 等只能作为待核对的解释。训练 loss 不能单独证明评估集效果，也不能直接说明哪种配置更好。

## 常见问题

**找不到 NPU 或无法导入 `torch_npu`**

先检查 ModelArts 实例规格、Notebook kernel 和 CANN 运行时。再看 `torch` 与 `torch_npu` 的版本是否匹配，不要直接安装来源不明的 `torch_npu` wheel。

**模型下载失败或磁盘空间不足**

确认实例可以访问 ModelScope，检查 `L07_MODEL_PATH` 是否指向包含 `config.json` 的模型目录，并把 `L07_WORK_DIR` 或 `MODELSCOPE_CACHE` 放到空间足够的持久化目录。

**找不到 `swift` 命令**

Notebook 会把当前 Python 的 bin 目录和用户 bin 目录加入 `PATH`。如果仍然找不到，检查 `ms-swift` 是否安装到了当前 Notebook 使用的 Python 环境。

**训练没有生成 `logging.jsonl` 或 checkpoint**

先查看 `notebook_stdout.log` 的最后一段，确认 `swift sft` 的实际命令、返回码和输出目录。不要手工补日志或曲线。

**loss 有波动，或者只看到很少几个点**

先核对 step、日志间隔、学习率、batch size、最大序列长度和数据顺序。smoke test 本来只有 3 个 step，点少不代表训练异常；单个 loss 点也不适合单独下结论。

**没有内存曲线**

只有 `logging.jsonl` 真实记录了 `memory(GiB)` 时才会绘图。没有这个字段，就在实验记录中写“未采集”，不要用 NPU 规格估算峰值显存。

## 目录结构

```
07_lora_fine_tuning/
├── README.md
├── 07.01_lora_experiment_introduction.ipynb
├── 07.02_ms_swift_smoke_test.ipynb
└── 07.03_ms_swift_training_and_loss_analysis.ipynb
```

本实验的 notebook 没有 TODO 实操题，因此不设置 `answer/` 目录。运行过程中产生的模型缓存、JSONL、日志、checkpoint 和图片放在 ModelArts 工作目录中，不提交到本仓库。

## 实验报告与提交

报告至少保留以下内容：

1. **环境和数据**：Python、PyTorch、`torch_npu`、CANN 运行时、NPU 数量、模型 ID、实际模型路径和 JSONL 数据路径。
2. **配置和命令**：`tuner_type`、`target_modules`、LoRA rank / alpha、学习率、batch size、最大序列长度、训练步数或 epoch、进程数。
3. **运行证据**：`notebook_stdout.log`、原始 `logging.jsonl`、checkpoint / adapter 路径、loss 曲线，以及实际采集到的内存字段。
4. **分析和边界**：从日志直接看到的事实、对波动的可能解释、异常及排查过程；明确写出哪些结论还没有被评估集或统一测量验证。

如果训练中断，保留发生时的 step、报错原文、配置和排查动作，不要只写“训练失败”。

## 参考链接

- ms-swift 官方仓库：https://github.com/modelscope/ms-swift
- ModelScope 模型页：https://modelscope.cn/models/Qwen/Qwen3-0.6B
- 华为云 ModelArts 文档：https://support.huaweicloud.com/modelarts/

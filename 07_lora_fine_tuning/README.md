# Lab 7：LoRA 参数高效微调

## 实验目标

本实验使用 `ms-swift` 在华为云 ModelArts 的 Ascend NPU 上，对 `Qwen/Qwen3-0.6B` 进行 LoRA 监督微调（SFT）。实验分为导学、最小训练验证和完整训练三个分册。

完成实验后，可以：

1. 写出 LoRA 的低秩更新公式，说明基础模型、适配器、`rank`、`alpha` 和目标层的含义。
2. 读懂 `swift sft` 的 LoRA 参数，区分冻结的基础权重与参与训练的适配器参数。
3. 在 ModelArts 上准备数据，运行 3-step smoke test 和 50-step LoRA 训练。
4. 从 `logging.jsonl` 绘制 loss 曲线，区分 raw loss 与 moving average。

## 实验环境

在 ModelArts 控制台创建 Notebook 实例时使用以下配置：

| 配置项 | 本实验配置 |
| --- | --- |
| 区域 | 西南-贵阳一 |
| 资源池 | 公共资源池 |
| 实例规格 | `1 × ascend-snt9b3`，24 vCPUs，192 GiB |
| 镜像 | `pytorch_ascend: pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b` |
| WebIDE | JupyterLab 4.4.10 |
| 系统盘 | 50 GiB |
| 数据盘 | EVS 50 GiB，挂载到 `/home/ma-user/work` |
| 网络 | 公共网络，允许访问公网，访问方式选择公有网关 |
| Kernel | `Python (PyTorch-2.7.1)` |

实验文件位于同一目录：

```text
07_lora_fine_tuning/
├── 07.01_lora_experiment_introduction.ipynb
├── 07.02_ms_swift_smoke_test.ipynb
├── 07.03_ms_swift_training_and_loss_analysis.ipynb
└── README.md
```

三个 Notebook 按 `07.01 → 07.02 → 07.03` 的顺序使用。每册都包含自己的初始化单元，并在工作目录下生成独立的 `L07-0x_workspace/`。

## 实验原理

全参数微调直接更新线性层权重 `W`。LoRA 保留基础权重，只学习低秩增量：

```text
W' = W + ΔW
ΔW = (alpha / r) × B × A
```

其中，`r` 是低秩分支的维度，`A` 和 `B` 是新增的可训练矩阵，`alpha` 用于缩放增量。若线性层形状为 `d_out × d_in`，全参数更新包含 `d_out × d_in` 个参数；LoRA 分支包含 `r × (d_in + d_out)` 个参数。

本实验通过 SFT 接口训练 LoRA 适配器。训练样本采用对话格式，基础模型保持冻结，优化器更新注入线性层的适配器参数。

## 实验流程

### 1. 07.01 LoRA 原理与环境准备

打开 `07.01_lora_experiment_introduction.ipynb`，从第一个代码单元开始运行。本册检查 Python、PyTorch、`torch_npu` 和 NPU，准备 `modelscope`、`ms-swift`、`matplotlib`、模型与 JSONL 对话数据。输出会给出模型路径、训练数据路径和 NPU 数量。未检测到 NPU 时，检查实例规格和 Kernel；模型下载失败时，检查公网访问和 `/home/ma-user/work` 的可用空间。

### 2. 07.02 最小训练验证

打开 `07.02_ms_swift_smoke_test.ipynb`，运行 3 个训练 step。训练命令使用以下 LoRA 参数：

```text
--tuner_type lora
--target_modules all-linear
--lora_rank 8
--lora_alpha 32
--learning_rate 1e-4
--max_steps 3
```

该分册读取 SFT 数据、执行单卡 NPU 训练，并写出 `logging.jsonl` 与适配器 checkpoint。

### 3. 07.03 完整训练与 loss 分析

打开 `07.03_ms_swift_training_and_loss_analysis.ipynb`，运行 50-step LoRA 训练。

| 参数 | 值 |
| --- | ---: |
| 训练记录 | 54 条（6 条教学样本重复 9 次） |
| `max_steps` | 50 |
| `logging_steps` | 1 |
| `save_steps` | 10 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 1 |
| `max_length` | 512 |

代码从 `logging.jsonl` 读取逐 step 的 `loss`，绘制 raw loss 和 5-step moving average，并保存 `loss_curve.png`。raw loss 会随训练批次变化，moving average 更适合观察整体走势。显存不足时，降低 `max_length`，或减小 batch size、缩小 LoRA 目标层范围。

## 实验总结

本实验以 Qwen3-0.6B 为基础模型，使用 LoRA 适配器完成 SFT。三个分册依次覆盖环境与数据准备、3-step 最小训练和 50-step 训练日志分析。训练过程中的基础模型保持冻结，日志记录适配器训练时的 loss。

## 实验扩展

将 `lora_rank` 改为 16，或将 `target_modules` 从 `all-linear` 改为 `q_proj`、`v_proj`。比较两种设置下的适配器参数量、训练日志和 loss 曲线。

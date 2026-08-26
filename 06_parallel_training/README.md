# 集群混合并行训练实验：在昇腾 NPU 上完成 Qwen2.5-7B 多卡训练

## 实验简介

本实验是第三章分布式训练理论课《大模型多机多卡训练原理与并行策略》《AllReduce、HCCL 与通信性能分析》《MindSpeed 分布式训练流程》的配套实验。你在理论课上学过的张量并行、流水线并行、AllReduce、HCCL、GAS 公式，这里全部会在真实的昇腾 910B NPU 上用到。

实验材料分两份：

| 材料 | 内容 | 是否启动训练 |
|------|------|--------------|
| 导学（本文件） | 实验目标、ModelArts 环境创建、启动前检查清单、提交要求 | 否 |
| 代码文件 `Lab6_mindspeed_training_config_and_launch.ipynb` | 五步闭环：下载权重数据、环境检查、权重转换、数据预处理、torchrun 启动训练、日志解析 | 是 |

实验目标三项：

| 目标 | 说明 | 验收 |
|------|------|------|
| ① 启动训练 | TP/PP/DP 配置正确，训练不报错 | loss 正常打印 |
| ② 分析日志 | 从日志提取三个关键指标 | loss / 吞吐 / 单步耗时 |
| ③ 完成报告 | 配置记录 + 日志分析 + 反思 | 算/跑/看/想四部分齐全 |

实验案例统一为 Qwen2.5-7B 单机 4 卡，并行配置 **TP=2 PP=2 DP=1**（4 卡铺成 2×2，数据并行的特点由 GAS=64 梯度累积体现），参数对齐 MindSpeed-LLM 官方脚本 `pretrain_qwen25_7b_32k_ptd.sh`。完成实验后你将得到：一次成功启动的多卡训练、一条真实下降的 loss 曲线、以及一份可以写进实验报告的性能数据。

## 实验环境与资源配置

实验在华为云 ModelArts 平台上进行，全部操作在浏览器里的 JupyterLab 中完成。

### 实例规格

| 项 | 配置 |
|----|------|
| 区域 | 西南-贵阳一（snt9b 昇腾资源仅在华为云少数区域提供，课程统一用此区域，实测可用） |
| 资源池 | 公共资源池 |
| 规格 | 4×Ascend 910B，单卡 64GB HBM2e（规格名以控制台实际显示为准） |
| 镜像 | mindspeed_llm_2.2.0（公共镜像，AI 引擎选 MindSpeed-LLM） |
| 存储 | EVS，50GB 以上（权重约 15GB + 转换产物 + 数据） |

全程使用这一个规格，从下载权重到训练结束不需要变更。

镜像预装 torch 2.7.1 + torch_npu、MindSpeed-LLM 2.2.0、transformers、matplotlib。modelscope 未预装，首次运行时会自动安装。镜像 tag 里的 `cann_8.2.rc2` 是历史命名遗留，实际 CANN 版本以镜像详情页"AI 引擎及框架"一栏为准。

计费按秒计算、每小时结算，价格以控制台显示为准。建议创建时开启自动停止（如 4 小时），防止忘记关机。NPU 资源紧张时创建可能排队或失败，换个时段重试即可。

### 存储

`/home/ma-user/work/` 是持久化目录（EVS 磁盘），停止实例后不丢失。该目录之外的路径在实例停止后清空。所有实验文件（权重、数据、转换产物、训练输出）都放这个目录下。EVS 磁盘持续按容量计费（费用很低），不用实验时应删除实例。

## 创建实验环境（分步操作）

1. 登录华为云控制台，右上角切换区域到**西南-贵阳一**。昇腾 910B 资源只在华为云少数区域提供，如果创建时找不到对应规格，先检查区域是否选对。

<img src="images/modelarts-step1-console.png" alt="控制台区域选择" width="640">

2. 进入 ModelArts 控制台，左侧选择"开发环境 → Notebook"，点击"创建"。

<img src="images/modelarts-step2-create.png" alt="创建 Notebook" width="600">

3. 填写创建参数：

   - 名称：自定义，如 `lab06-你的姓名拼音`
   - 自动停止：建议开启（如 4 小时）
   - 工作环境：选择"公共镜像"中的 `mindspeed_llm_2.2.0-...`（在 AI 引擎列表里选 MindSpeed-LLM），镜像预装 CANN、torch_npu、MindSpeed-LLM，环境免搭建
   - 资源池：公共资源池
   - 规格：4×Ascend 910B（单卡 64GB）
   - 存储配置：EVS，容量 50GB 以上（权重约 15GB，另有转换产物与数据）

<img src="images/modelarts-step3-image.png" alt="选择镜像" width="600">

<img src="images/modelarts-step4-flavor.png" alt="选择规格" width="600">

4. 点击"立即创建"，等待状态变为"运行中"。NPU 资源紧张时创建可能排队或失败，换个时段重试即可。

5. 点击"接入环境"进入 JupyterLab，左侧文件树进入 `/home/ma-user/work/`，把本实验的代码文件上传到这里再开始实验。

<img src="images/modelarts-step5-jupyterlab.png" alt="打开 JupyterLab" width="760">

6. **创建后先核对**：运行 `npu-smi info`，确认显示 4 张卡、单卡显存 64GB。卡数或显存不对说明规格选错了，先停下检查。

### JupyterLab 的分工

- **代码文件**：参数计算、配置生成、日志解析
- **终端**（菜单 File → New → Terminal）：执行 torchrun 多卡训练

<img src="images/modelarts-step6-terminal.png" alt="JupyterLab 与终端" width="600">

## 启动前检查清单

以下 8 项任何一项不满足，训练都无法正常启动或很快中断。逐项过一遍，每项标注了验证位置，对应代码文件中的步骤：

| # | 检查项 | 验证方法 | 验证位置 |
|---|--------|---------|---------|
| ① | CANN 环境已加载 | `npu-smi info` 有输出，能看到 4 张卡 | Step 1 检查环境 |
| ② | HCCL 通信正常 | AllReduce 校验通过，每个 rank 打印 OK | Step 1 检查环境 |
| ③ | 数据产物存在 | `alpaca_text_document.bin/.idx` 两个文件都在 | Step 3 数据预处理 |
| ④ | 权重转换产物存在 | `mp_rank_*` 目录共 4 个（= TP×PP = 2×2） | Step 2 权重转换 |
| ⑤ | TP×PP ≤ 总卡数 | 2×2 = 4 ≤ 4，成立 | 全局配置 |
| ⑥ | 显存预算通过 | 见下方估算 | 全局配置 |
| ⑦ | MASTER_ADDR / MASTER_PORT 正确 | 单机用 localhost + 未占用端口 | Step 4 启动训练 |
| ⑧ | 输出目录可写 | 训练脚本能创建输出目录并写日志 | Step 4 启动训练 |

**第⑥项显存估算**：Qwen2.5-7B 总参数约 7.6B，按 TP=2 PP=2 切分后每卡约 1.9B 参数。bf16 训练加 AdamW 优化器，每参数约 20 字节，模型部分约 38GB，加上激活值（seq=8192、已开启重计算，占比很小）和运行开销，每卡约 42GB / 64GB，满足要求。显存不足时三个调整方向：减小 seq_length、增大 TP 或 PP、开启重计算。

## 实验流程

### 代码文件（五步闭环）

| 步骤 | 做什么 | 大约耗时 |
|------|--------|----------|
| Step 0 | 下载 Qwen2.5-7B 权重（ModelScope，约 15GB）和 alpaca 数据集（24MB） | 20-40 分钟 |
| Step 1 | `npu-smi info` 查看 NPU，验证 torch_npu 可用，跑一次 AllReduce 通信测试 | 5 分钟 |
| Step 2 | 权重转换：HuggingFace 格式按 TP=2 PP=2 切分转成 Megatron 格式（产物 4 个 mp_rank 目录） | 10 分钟 |
| Step 3 | 数据预处理：alpaca 文本转成 token 序列（.bin + .idx） | 5 分钟 |
| Step 4 | 生成 run_train.sh，在 JupyterLab 终端执行 `bash run_train.sh` 启动 4 卡训练 | 默认 100 步（单步约 1 分钟，总计约 1.5-2 小时），100 个 loss 点可画出平滑下降曲线 |
| Step 5 | 回到代码文件解析训练日志，提取 loss、吞吐、单步耗时，画 loss 曲线 | 5 分钟 |

在代码文件中按顺序执行，每个步骤前有说明文字。训练命令在 JupyterLab 终端（菜单 File → New → Terminal）执行。终端里 `cd /home/ma-user/MA_Turbo/src/open_source/MindSpeed-LLM` 后再 `bash` 启动脚本。

训练正常启动的标志是日志里出现 `iteration 1 | lm loss: 10.x`。默认 100 步约 1.5-2 小时。如需先快速验证全链路，可把 `train-iters` 临时改小（如 10），跑通后再改回 100 重新生成脚本。

### 常见问题

**下载权重时 modelscope 报 `invalid literal for int(): 'ERROR'`**
环境变量 MODELSCOPE_LOG_LEVEL 要求数字值。如果你自己另开终端跑下载命令，注意这一点。

**`ModuleNotFoundError: No module named 'megatron'`**
环境准备需要联网补装一个依赖，约 1 分钟。失败时先检查实例的网络访问。

**`NPU out of memory`**
先看发生在哪个阶段。训练刚启动、优化器构建时就 OOM，说明每卡参数量太大（并行度不够），调整 TP/PP 让每卡参数降下来。训练若干步后才 OOM，是激活值问题，减小 seq_length 或确认重计算已开启。注意每卡参数量近似等于总参数除以 TP×PP。切分度不够时，优化器状态（每参数约 12 字节的 fp32 副本）会最先耗尽显存。

**loss 变为 NaN**
训练中 loss 突然变成 NaN，常见原因是学习率偏大或数值不稳定，优先核对学习率设置与梯度裁剪配置。

**页面里的平台截图不显示**
截图以相对路径引用 `images/` 文件夹，上传时必须保持 `images/` 与代码文件同级（建议整个实验目录打 zip 上传后解压）。若页面打开在先、图片上传在后，已渲染的页面不会自动刷新，刷新浏览器页面即可。

**HCCL 通信卡住不动**
确认设置 `HCCL_WHITELIST_DISABLE=1` 和 `HCCL_CONNECT_TIMEOUT=7200`（训练脚本里已写）。单机 4 卡不需要 rank table 文件。

**数据预处理的产物找不到**
`preprocess_data.py` 会在 `--output-prefix` 后面自动追加 `_text_document`，所以输出前缀设为 `alpaca` 时，实际产物是 `alpaca_text_document.bin/.idx`，训练的 `--data-path` 也用这个完整名字。

**权重转换必须用什么命令**
v2.2.0 的 `convert_ckpt_v2.py` 不支持 qwen25，要用旧版 `convert_ckpt.py` 加 `--model-type-hf llama2 --add-qkv-bias`。Qwen 的底层架构（RMSNorm、SwiGLU、RoPE）与 LLaMA2 同源，转换规则通用。转换时的 TP/PP 必须与训练完全一致，否则加载时报形状不匹配。

## 实验报告与提交

报告分四部分，对应"算、跑、看、想"：

1. **算**：并行配置记录，包括你使用的 TP/PP/DP、MBS/GBS、GAS 计算过程、显存估算结果。
2. **跑**：训练日志摘录，开头 10 步和最后 10 步的日志，标注 lm loss、吞吐、单步耗时三个指标。
3. **看**：性能数据，包括最终 loss、平均吞吐（tokens/s/p 或 TFLOP/s/GPU）、平均单步耗时、显存占用（`npu-smi info` 截图）。
4. **想**：结论与反思，配置是否合理、遇到什么问题、怎么解决的，以及如果换 TP=4 PP=1 会怎样。

目录组织：根目录以姓名拼音命名，下分 `config/`（你实际使用的启动脚本）、`logs/`（训练日志原文件）、`report.md`（实验报告）、`images/`（截图）。

评分维度：配置正确性 40%，日志完整性 30%，分析深度 30%。跑通是基础，能否解释为什么这样配置、卡数或并行度变化时会发生什么，决定得分上限。

## 参考链接

- MindSpeed-LLM 官方仓库：https://github.com/Ascend/MindSpeed-LLM
- 本实验对齐的官方脚本：仓库 `examples/mcore/qwen25/pretrain_qwen25_7b_32k_ptd.sh`（v2.2.0）
- 华为云 ModelArts 文档：https://support.huaweicloud.com/modelarts/

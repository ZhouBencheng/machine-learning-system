# Lab 9：MindIE / AISBench 并发压测

## 实验目标

- 使用 AISBench 对已启动的 MindIE 推理服务执行并发压测。
- 读取正式轮结果，汇总吞吐量与端到端延迟。
- 生成原始结果 CSV、汇总 CSV 和性能图表。

## 实验环境

本实验在已经部署并启动的 MindIE 服务上运行。服务启动与可用性检查按服务器中的 `/home/ma-user/AGENT.md` 完成。

- Python 3.11
- AISBench 虚拟环境：`/home/ma-user/work/aisbench-venv311`
- AISBench 工作目录：`/home/ma-user/work/aisbench-benchmark`
- 实验目录：`/home/ma-user/work/mindie-concurrency-benchmark`
- 推理服务：Qwen2-7B，接口 `http://127.0.0.1:1025`
- 数据集：GSM8K 4-shot CoT chat

Notebook 中的路径、模型和压测参数与该服务器环境对应。如需在其他环境运行，在参数单元修改相应路径和配置。

## 实验原理

压测保持模型、数据集、输出长度和请求速率不变，只改变 AISBench 的 `batch_size`。并发点依次为 1、2、4、8、16、32、64、128、192、256。

每个并发点先执行一轮预热。并发小于 16 时执行一轮正式测量；并发不小于 16 时计划执行三轮。正式轮的请求数为 `min(1319, max(128, batch_size × 8))`。GSM8K 测试集有 1319 条唯一题目，高并发点的请求数受此上限约束。

结果汇总时，吞吐量取同一并发点正式轮的均值；P95 和 P99 由各轮逐请求端到端延迟合并后重新计算。若高并发点首轮的吞吐量相较前一个完整点回落，且 P99 增长达到 25%，Notebook 会把该点记录为过载点并停止后续并发测试。

## 实验流程

按以下顺序打开并运行 [09.01_mindie_aisbench_concurrency_benchmark.ipynb](09.01_mindie_aisbench_concurrency_benchmark.ipynb)。

### 1. 设置路径与压测参数

检查 AISBench、实验目录和推理服务接口的路径。根据需要调整并发点、请求数和输出目录。

### 2. 检查服务与依赖

运行启动前检查单元，确认服务端口可访问、AISBench 命令存在，并确认实验目录可写。

### 3. 生成 AISBench 配置并查看计划

Notebook 会为每个并发点生成配置，并列出预热轮和正式轮的执行计划。

### 4. 执行并发扫描

先查看单轮压测的调用方式，再运行完整扫描。已写入 `COMPLETE` 标记的轮次会跳过，可用于中断后的继续执行。

### 5. 汇总结果并绘图

正式轮结果会写入 `concurrency_results_raw.csv` 和 `concurrency_results.csv`，随后生成并发数与吞吐量、并发数与尾延迟、吞吐量与尾延迟三张图。

## 实验总结

本实验完成了 AISBench 配置、并发扫描、结果汇总和性能图表绘制。结果可用于观察吞吐量随并发变化的趋势，以及高并发下的尾延迟变化。

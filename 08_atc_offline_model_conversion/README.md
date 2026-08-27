# 实验 8　ONNX / ATC / OM 模型转换

## 实验目标

- 将 PyTorch 模型导出为 ONNX，检查模型图与输入输出信息；
- 使用 ATC 将 ONNX 模型转换为 OM，并理解输入 shape、目标芯片和精度模式等参数；
- 使用 msame 准备 bin 输入、读取 OM 输出，并按转换链路定位问题。

## 实验环境

本实验使用 PyTorch、ONNX、ONNX Runtime 和 onnxscript。ONNX 导出与 ONNX Runtime 检查可在 CPU 环境完成；ATC 转换和 msame 推理需要安装昇腾开发套件，并使用 NPU。

Notebook 中调用 ATC 的 shell cell 会加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

每个 shell cell 都是独立进程，调用 ATC 时需在该 cell 中重新加载环境。`--soc_version` 应通过 `npu-smi info` 或 `acl.get_soc_name()` 获取，不使用示例中的固定值。

三个分册共用运行时生成的 `l08_workspace/`。其中的 ONNX、OM 与 bin 文件会在执行过程中生成。

## 实验原理

`.pth` 依赖训练框架和原始模型代码，ONNX 用统一的图格式保存模型结构与权重，OM 是面向特定昇腾芯片编译的离线模型。ATC 负责将 ONNX 编译为 OM；msame 在设备上运行 OM，并可与 ONNX Runtime 的输出比较。

模型使用静态 shape 时，ATC 将输入尺寸直接写入编译配置。动态 batch、分辨率或其他维度需要在 ONNX 导出、ATC 转换和 msame 推理三处一致声明。

## 实验流程

### 1. 导出 ONNX 并完成基础转换

运行 `08.01_onnx_om_and_atc.ipynb`。本分册定义 DemoNet、导出 ONNX、用 ONNX Runtime 检查输出，并调用 ATC 生成 OM。它会创建后续分册使用的 `l08_workspace/` 文件。

### 2. 配置 ATC 参数

运行 `08.02_atc_conversion_commands_and_parameters.ipynb`。本分册说明必填参数、精度模式和动态 shape 的配置方法，并给出静态与动态模型的命令示例。

### 3. 使用 msame 推理与排错

运行 `08.03_msame_inference_validation_and_troubleshooting.ipynb`。本分册说明 bin 文件组织、输出读取、耗时统计、动态 shape 推理，以及按环境、参数、数据和转换阶段定位问题的方法。

每本 Notebook 末尾保留了原有思考题。参考答案存放在 `answer/`，可在对应 Notebook 的“参考答案”章节中使用 `cat` 查看。

## 实验总结

本实验的主线是：从 PyTorch 模型导出 ONNX，用 ATC 生成与目标芯片匹配的 OM，再通过 msame 运行并读取输出。模型格式、输入 shape、`soc_version` 和精度模式需要在这条链路中保持一致。

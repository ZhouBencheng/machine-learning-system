# 华为 ICT 学院 —— 智能计算系统课程

## 课程简介

本仓库收录智能计算系统课程的后续实验。各 Lab 独立组织，具体的运行环境与操作说明见对应目录中的 README。

## 课程目录

| Lab | 实验 | Notebook |
| --- | --- | --- |
| Lab 6 | MindSpeed 并行训练配置与启动 | `06.01_mindspeed_training_config_and_launch.ipynb` |
| Lab 7 | LoRA 参数高效微调 | `07.01_lora_experiment_introduction.ipynb`、`07.02_ms_swift_smoke_test.ipynb`、`07.03_ms_swift_training_and_loss_analysis.ipynb` |
| Lab 8 | ONNX 模型转换与 OM 推理验证 | `08.01_onnx_om_and_atc.ipynb`、`08.02_atc_conversion_commands_and_parameters.ipynb`、`08.03_msame_inference_validation_and_troubleshooting.ipynb` |
| Lab 9 | MindIE / AISBench 并发压测 | `09.01_mindie_aisbench_concurrency_benchmark.ipynb` |
| Lab 10 | 端云双模式大模型对话应用 | `10.01_cloud_mindie_service_setup.ipynb`、`10.02_edge_cloud_dual_mode_dialogue.ipynb` |

## 目录与命名规范

每个 Lab 目录采用以下约定：

```
0X_<lab_slug>/
├── README.md                    # 实验指导书
├── 0X.01_<section_slug>.ipynb   # 实验分册
├── 0X.02_<section_slug>.ipynb
├── data/                        # 配套数据（可选）
├── pdf/                         # 配套 PDF（可选）
├── images/                      # 实验配图（可选）
└── answer/                      # 参考答案（可选）
```

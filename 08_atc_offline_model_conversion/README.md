---
version: v0.1
owner: 待填
reviewer: 待填
updated: 2026-08-05
---

# 实验 8　ONNX / ATC / OM 模型转换

- 实验编号：Lab 08
- 实验名称：使用 ATC 将 ONNX 模型转换为 OM 并验证
- 对应视频：V02-14、V03-14
- 版本状态：v0.1

## 课件清单

| 文件 | 内容 |
| --- | --- |
| `L08-01_onnx_om_atc_intro.ipynb` | 导学：全链路走通（47 cells / 18 code） |
| `L08-02_atc_params.ipynb` | ATC 参数（21 cells / 5 code） |
| `L08-03_msame_verify.ipynb` | msame 验证与排错（30 cells / 11 code） |
| `answer/L08-0x_answer.md` | 三份练习答案（教师用） |

`answer/` 里标注 **[需实测]** 的题目只给判据和填空表格，不预填数值——报错原文、耗时数据和文件体积要从实验环境获取。L08-01 第 2 题的答案可能和直觉相反：删掉 `model.eval()` 后对齐仍可能通过，因为 `torch.onnx.export` 的 `training` 默认是 `TrainingMode.EVAL`。

## 章节结构

**L08-01 导学**

| 节 | 内容 |
| --- | --- |
| 1 | 为什么要转换：指令集、运行时、关注点三层差异 |
| 2 | 全链路总览、编译器类比、两个对齐点 |
| 3 | 训练框架模型：`.pth` 的内容、`model.eval()` |
| 4 | 导出 ONNX：trace 机制、参数、动态维度、常见问题 |
| 5 | ONNX 体检：checker、读图、onnxruntime、**对齐 ①** |
| 6 | ATC 转 OM：环境变量、soc_version 实测、五个必填参数 |
| 7 | OM 验证：bin 准备、msame、**对齐 ②** |
| 8-9 | 小结、报错索引、练习 5 题 |

**L08-02 ATC 参数**

| 节 | 内容 |
| --- | --- |
| 1 | 命令骨架与参数分组 |
| 2 | 五个必填参数逐个说明 |
| 3 | `--input_format` / `--log` / 精度模式 |
| 4 | 动态 shape 四种模式与互斥规则 |
| 5 | 转换前自查清单 |
| 6-8 | 三个完整案例、速查表、练习 6 题 |

**L08-03 msame 验证**

| 节 | 内容 |
| --- | --- |
| 1-2 | msame 定位、获取编译、四个基本参数 |
| 3 | bin 组织、输出读回、**对齐 ② 三级判定** |
| 4 | 推理耗时：预热与四个指标 |
| 5 | 动态 shape：`--dymShape` / `--outputSize` |
| 6 | 常见错误按阶段索引 |
| 7 | 转换失败的五类原因 |
| 8-9 | 验证视图、练习 8 题 |

## 运行方式

```bash
pip install torch onnx onnxruntime      # CPU 版；onnxruntime 与 onnxruntime-gpu 只能装一个
jupyter lab L08-01_onnx_om_atc_intro.ipynb
```

三份课件按顺序执行，L08-01 生成的 `l08_workspace/demo_model.onnx` 和 `.om` 会被后两份复用。

课件假定在昇腾开发环境运行。L08-01 第 3~5 节只需 CPU 版 PyTorch，第 6~7 节及 L08-02、L08-03 的 shell cell 需要昇腾开发套件与 NPU。shell 命令用 `%%bash` / `!` 直接书写，每个 cell 是独立子进程，用到 `atc` / `msame` 的 cell 都要重新 `source set_env.sh`。


## 目录结构

```text
lab_08_atc_conversion/
├── README.md
├── L08-01_导学大纲.md
├── L08-01_onnx_om_atc_intro.ipynb
├── L08-02_参数大纲.md
├── L08-02_atc_params.ipynb
├── L08-03_验证大纲.md
├── L08-03_msame_verify.ipynb
└── l08_workspace/                # 运行时生成，可删除
```

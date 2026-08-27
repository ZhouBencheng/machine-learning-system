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
- 版本状态：v0.1（AI 初稿，待技术协审）

## 课件清单

| 文件 | 内容 |
| --- | --- |
| `L08-01_导学大纲.md` | L08-01 讲解大纲 |
| `L08-01_onnx_om_atc_intro.ipynb` | 导学：全链路走通（47 cells / 18 code） |
| `L08-02_参数大纲.md` | L08-02 讲解大纲 |
| `L08-02_atc_params.ipynb` | ATC 参数（21 cells / 5 code） |
| `L08-03_验证大纲.md` | L08-03 讲解大纲 |
| `L08-03_msame_verify.ipynb` | msame 验证与排错（30 cells / 11 code） |
| `answer/L08-0x_answer.md` | 三份练习答案（教师用，不随课件下发） |

`answer/` 里标注 **[需实测]** 的题目只给判据和填空表格，不预填数值——报错原文、耗时数据和文件体积必须来自实验环境。L08-01 第 2 题的答案与直觉相反（删掉 `model.eval()` 后对齐仍可能通过，因为 `torch.onnx.export` 的 `training` 默认为 `TrainingMode.EVAL`），该默认值需在实验环境复核。

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

## 待办

1. 在装有 torch/onnx/onnxruntime 的环境执行 L08-01 第 3~5 节，确认输出与讲解一致
2. 在昇腾环境执行所有 shell cell，把 `--soc_version=Ascend310P3` 换成实测值
3. 跑 `atc --help` 与 `msame --help`，核对课件中的参数
4. 确认 `--precision_mode` 在实验环境的默认值与 `--precision_mode_v2` 的可用性
5. 锁定 CANN 版本与 `opset_version`（素材跨 CANN 5.0.x~8.x，环境变量写法不同）
6. L08-01 需要一份 `demo_model_dyn.onnx`（动态 batch）供 L08-02、L08-03 的动态示例使用
7. 补 `environment.md`、`faq.md`、`report_template.md`、`check_env.sh`、`run.sh`、`check_result.py`（`check_structure.py` 要求）
8. 复核 `torch.onnx.export` 的 `training` 默认值，确认 L08-01 练习 2 的答案表述
9. 若在无 NPU 环境录制视频，依赖 CANN 的部分改用教师预采集日志并注明来源

## 素材说明

素材在 `E:\北京航空航天大学\华为课程\html\`（6 份网页存档，Git 不收录）。使用时注意：

- `解析CANN中的msame工具_CSDN` 带 AI 生成水印，其参数体系（`--compare` / `--threshold` / `--iteration` / `--warmup` / `--perf`）与官方 msame 不符，课件未采用，仅借用推理耗时的指标概念
- `错误2MSAME模型加载超时` 只剩 `_files` 目录，正文未保存，L08-03 第 6 节该主题标注为待补充
- 网上教程里的 `Ascend310` / `Ascend910` 是原作者环境值，不可照抄

制作时对照官方文档核实，修正了初稿中两处错误：动态 batch 参数名应为 `--dynamic_batch_size`（非 `--dynamic_batch`）；framework 取值补 `1`=MindSpore，并注明 2、4 不使用。

`--precision_mode` 的默认值故意未写死——官方文档中该值随芯片型号与 CANN 版本变化，课件改为要求用本机 `atc --help` 确认。

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

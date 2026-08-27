---
version: v0.1
owner: 待填
reviewer: 待填
updated: 2026-08-05
---

# L08-02 练习答案

标注 **[需实测]** 的题目要求提交本机实际输出。

---

## 1　三输入 NLP 模型的 `--input_shape`

```bash
--input_shape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128"
```

三个要点：

**分号分隔不同输入，逗号分隔维度。** 全用逗号是最常见的错法，ATC 会把 `1,128,attention_mask` 当成一个输入的维度列表去解析，报的错和真实原因关系不大。

**整个值用双引号包住。** 不加引号时 shell 在第一个分号处截断命令，`attention_mask:1,128` 被当成新命令执行，得到 `command not found`——看起来完全不像 ATC 的问题。

**名字必须与 ONNX 完全一致**，包括大小写。可靠做法见课件 2.4 节，从模型里读。

配套的 `--input_format=ND`。这三个输入都是 `(batch, seq_len)` 的二维张量，不是图像的四维 NCHW。

---

## 2　找出命令里的错误

原命令：

```bash
atc --model=model.onnx --framework=3 --output=model.om \
    --input_shape="a:1,3,224,224,b:1,10" --soc_version=Ascend310
```

| # | 位置 | 错误 | 正确 |
| --- | --- | --- | --- |
| 1 | `--framework=3` | 3 是 TensorFlow，模型是 `.onnx` | `--framework=5` |
| 2 | `--output=model.om` | `--output` 是前缀，不带扩展名 | `--output=model` |
| 3 | `--input_shape` | 两个输入之间用了逗号 | `"a:1,3,224,224;b:1,10"` |

第四处值得一并指出：`--soc_version=Ascend310` 是文档里的示例值，必须换成本机 `npu-smi info` 或 `acl.get_soc_name()` 的实测结果。它不算语法错误，但照抄会导致转换失败或转出无法在本机运行的 OM。

改正后：

```bash
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --input_shape="a:1,3,224,224;b:1,10" \
    --soc_version=<实测值>
```

三处错误的表现差异很大，值得对比：`--output` 那处不报错，只是安静地产出 `model.om.om`；`--framework` 那处报解析失败，但不会提示是 framework 填错；`--input_shape` 那处报输入节点相关的错，方向大致正确。**报错信息的清晰度和错误的严重程度不成正比**，这是 ATC 排错的基本认知。

---

## 3　改成支持四个 batch 档位

两步都要做，只改 ATC 命令不行。

**第一步，ONNX 重新导出成动态 batch**（即 L08-01 练习 1）：

```python
torch.onnx.export(
    model, torch.randn(1, 3, 32, 32),
    "l08_workspace/demo_model_dyn.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

**第二步，ATC 命令**：

```bash
atc --model=l08_workspace/demo_model_dyn.onnx \
    --framework=5 \
    --output=l08_workspace/demo_model_bs1248 \
    --input_format=NCHW \
    --input_shape="input:-1,3,32,32" \
    --dynamic_batch_size="1,2,4,8" \
    --soc_version=<实测值>
```

两处必须配套：`--input_shape` 的 batch 维写 `-1`，`--dynamic_batch_size` 给出档位。

- 只写 `-1` 不给档位：报 `Input shape not fully specified`
- 只给档位不写 `-1`：`-1` 缺失，ATC 认为 shape 已固定，档位参数无处安放，报参数冲突

如果第一步没做，用静态 ONNX 配 `--input_shape="input:-1,3,32,32"`，ATC 会因为 ONNX 里 batch 维是固定的 1、而命令要求它可变，报 shape 推导冲突。**动态能力必须在导出时就声明，ATC 不能把静态模型改造成动态的。**

推理侧对应用 `--dymBatch` 指定本次用哪个档位，见 L08-03 第 5 节。

---

## 4　两种精度模式的输出对比 **[需实测]**

转两版：

```bash
# 默认精度
atc --model=demo_model.onnx --framework=5 --output=demo_default \
    --input_format=NCHW --input_shape="input:1,3,32,32" \
    --soc_version=<实测值>

# 保持原始精度
atc --model=demo_model.onnx --framework=5 --output=demo_fp32 \
    --input_format=NCHW --input_shape="input:1,3,32,32" \
    --soc_version=<实测值> \
    --precision_mode=must_keep_origin_dtype
```

两版都用同一个 `input.bin` 跑 msame，按 L08-03 第 3.3 节的三级判定填表：

| | 默认精度 | `must_keep_origin_dtype` |
| --- | --- | --- |
| argmax 与 onnxruntime 一致 | | |
| 余弦相似度 | | |
| 最大绝对误差 | | |
| OM 大小 | | |

**预期趋势**（实际数值需实测）：`must_keep_origin_dtype` 版本的误差应显著小于默认版本，通常能通过 `rtol=1e-3` 的严格容差；默认版本走 FP16，误差量级在 1e-3 到 1e-2。两版的 argmax 都应与 onnxruntime 一致——DemoNet 只有一层卷积加一层全连接，累积误差还不足以改变分类结果。

**这个对照的用途**：怀疑精度问题时先跑这一组。`must_keep_origin_dtype` 正常而默认版异常，说明是精度模式导致，属于预期行为，考虑用混合精度或对敏感层单独配置；两版都异常，说明问题在转换本身或模型本身，精度模式是无关变量，应该回到 ONNX 阶段查。

注意 `must_keep_origin_dtype` 只用于排查。它放弃了 FP16 的算力优势，性能明显下降，不用于部署。

---

## 5　核对 `atc --help` **[需实测]**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --help > atc_help.txt
```

逐项核对课件第 7 节速查表里的参数，填写差异：

| 参数 | 本机是否存在 | 默认值 | 与课件的差异 |
| --- | --- | --- | --- |
| `--model` | | | |
| `--framework` | | | |
| `--output` | | | |
| `--input_shape` | | | |
| `--soc_version` | | | |
| `--input_format` | | | |
| `--log` | | | |
| `--precision_mode` | | | |
| `--precision_mode_v2` | | | |
| `--dynamic_batch_size` | | | |
| `--dynamic_image_size` | | | |
| `--dynamic_dims` | | | |
| `--input_shape_range` | | | |

**重点关注两项**：`--precision_mode` 的默认值（课件故意没写死，官方文档中该值随芯片型号与 CANN 版本变化）；`--precision_mode_v2` 是否存在（较新的参数，旧版 CANN 可能没有）。

这道题的目的不是核对本身，而是建立习惯：**参数以本机 help 为准，不以文章为准。** 网上教程与本机 CANN 版本不一致是常态，遇到参数不认识、取值报错，第一动作是查 help。

---

## 6　为什么档位越多 OM 越大、转换越慢

回到 OM 的组成：优化后的计算图、重排后的权重、针对目标芯片的执行调度信息。

**转换变慢的原因。** ATC 的优化决策依赖具体 shape：算子怎么融合、tiling 参数怎么切、内存怎么复用、循环怎么展开，都要按 shape 算。不同 batch 的最优方案不同，所以每个档位要独立走一遍编译流程。N 个档位大致就是 N 倍的编译时间。

**OM 变大的原因。** 权重只存一份，所有档位共享。变大的是每个档位各自的执行调度信息——各自的 tiling 参数、内存分配方案、任务下发序列。所以体积增长是次线性的：权重占比高的大模型，加档位对体积影响不大；权重少而结构复杂的小模型，影响明显。

**运行时的代价。** 加载时所有档位的调度信息都要读进来，模型加载时间随档位数增加。切换档位可能触发内部的重新配置。

**工程建议。** 按实际请求分布选 3~5 档，覆盖主要场景。档位取值优先用 2 的幂，与硬件的对齐要求匹配得更好。极端 shape 单独处理，不要为了覆盖 1% 的长尾请求增加一个档位——那 1% 走另一个 OM 或降级路径更划算。

---

## 提交清单

- 第 1、2 题：写在报告里
- 第 3 题：导出脚本 + ATC 命令 + 转换日志
- 第 4 题：两版 OM 的三级判定结果与体积
- 第 5 题：`atc_help.txt` 与核对表
- 第 6 题：写在报告里
- 硬件型号与 CANN 版本

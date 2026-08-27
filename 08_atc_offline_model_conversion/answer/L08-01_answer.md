---
version: v0.1
owner: 待填
reviewer: 待填
updated: 2026-08-05
---

# L08-01 练习答案

标注 **[需实测]** 的题目要求学生提交本机实际输出，本文件只给判据和参考格式，不预填数值。

---

## 1　改成动态 batch 重新导出

导出时增加 `dynamic_axes`，其余不变：

```python
model.eval()
torch.onnx.export(
    model,
    torch.randn(1, 3, 32, 32),
    "l08_workspace/demo_model_dyn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

用 5.2 的 `describe()` 读回，输入行的第 0 维从 `1` 变成字符串 `batch`：

```text
输入 input    FLOAT    ['batch', 3, 32, 32]
输出 output   FLOAT    ['batch', 10]
```

要点：`dim_value` 为 0 且 `dim_param` 非空，表示这一维是符号维。静态维在 protobuf 里存数值，动态维存名字，`describe()` 里 `d.dim_param if d.dim_param else d.dim_value` 就是在区分这两种情况。

`dummy_input` 的 batch 仍写 1。它只用于 trace，声明为动态的维度不受它约束。

注意输出的 batch 维也要一起声明。只声明输入会导致输出 batch 被固化成 1，后续 ATC 转换时报 shape 推导不一致。

---

## 2　去掉 `model.eval()` 后运行对齐 ①

**这道题的结果可能与直觉相反，这正是要讲的点。**

只把 4.3 节的 `model.eval()` 删掉，对齐 ① 大概率仍然通过。原因是 `torch.onnx.export` 的 `training` 参数默认为 `TrainingMode.EVAL`，导出器在 trace 期间会临时把模型切到 eval 模式。所以 ONNX 里的 Dropout 已被消除、BatchNorm 用的是 running stats。

（`training` 默认值请在实验环境用 `inspect.signature(torch.onnx.export)` 复核，不同 PyTorch 版本的默认行为可能调整。）

要让对齐真正失败，有两种改法：

**改法 A：让参照侧处于 train 模式。** 把对齐 cell 里的 `model.eval()` 也删掉：

```python
model.train()                      # 参照侧用 train 模式
with torch.no_grad():
    torch_out = model(torch.from_numpy(x_np)).numpy()
```

`assert_allclose` 抛 `AssertionError`，误差量级在 1e-1 甚至更大。原因是参照侧的 Dropout 随机置零了一半激活值，而 ONNX 侧没有。

**改法 B：显式要求按 train 模式导出。**

```python
from torch.onnx import TrainingMode
torch.onnx.export(..., training=TrainingMode.TRAINING)
```

导出的 ONNX 里会保留 Dropout 节点，两侧都带随机性，同一输入多次推理结果都不同。

**还有一种更隐蔽的情况**，对齐检查抓不到：在 train 模式下跑过若干次前向，BatchNorm 的 running_mean / running_var 被这些输入污染了。之后无论怎么导出，torch 和 ONNX 用的都是同一份被污染的统计量，所以两边一致、对齐通过，但模型相对训练结束时的状态已经变了。这类问题只能靠"不在推理路径上调用 `model.train()`"来避免，测不出来。

结论仍然成立：导出前 `model.eval()` 是必须的规范动作。但要清楚它防的是什么——防的是参照口径不一致和显式的 train 模式导出，不是防导出器本身。

---

## 3　`--input_shape` 输入名写错 **[需实测]**

把 6.3 的命令改为一个不存在的输入名：

```bash
--input_shape="inputs:1,3,32,32"        # 模型里实际叫 input
```

ATC 退出码非 0，报错属于"输入节点未找到"一类。提交时附完整日志，至少包含错误码（`E` 开头）和错误描述行。

报错为什么指向这里：ATC 解析 ONNX 后得到图的输入节点列表，再用 `--input_shape` 里的名字去这个列表里查找并绑定 shape。名字查不到，绑定阶段就失败，此时模型解析已经完成，所以报错发生在参数处理阶段而非解析阶段。

配合 L08-02 第 2.4 节：可靠做法是从 ONNX 里读名字生成 `--input_shape`，不手抄。

---

## 4　ONNX 与 OM 的体积对比 **[需实测]**

填写实测值：

| 文件 | 大小 |
| --- | --- |
| `demo_model.onnx` | |
| `demo_model.om`（默认精度） | |
| `demo_model_fp32.om`（`must_keep_origin_dtype`） | |

差异来源有三项，方向不一致：

**权重精度**——默认精度模式常见为 `force_fp16`，FP32 权重转成 FP16 后这部分减半。这是唯一确定让 OM 变小的因素。

**权重重排**——为匹配达芬奇架构的访存特性，权重会转成分形格式（如 NC1HWC0）。重排需要按 16 或 32 对齐补零，通道数不是对齐倍数时会产生填充，这部分变大。DemoNet 的 8 通道要补到 16，填充比例相当高。

**执行调度信息**——OM 里额外包含算子的 tiling 参数、内存分配方案、任务下发序列，ONNX 里没有这些。这部分只增不减。

所以不要假设 OM 一定比 ONNX 小。小模型上后两项占比高，OM 反而可能更大；大模型上权重占绝对多数，FP16 的收益才体现出来。

对照 `must_keep_origin_dtype` 那一版：它保持 FP32，与默认版的差值基本就是权重精度那一项的贡献。

---

## 5　`if` 分支依赖输入数值时 trace 会怎样

**会发生什么。** trace 是"喂一个输入跑一次前向，记录实际走过的算子"。`if` 的条件在 trace 时被求值成一个具体的布尔量，只有取到的那一支被记录进 ONNX，另一支彻底消失。导出过程不报错，可能只有一条 `TracerWarning`。

举例：

```python
def forward(self, x):
    if x.sum() > 0:
        return self.branch_a(x)
    else:
        return self.branch_b(x)
```

`dummy_input` 恰好让 `x.sum() > 0` 成立，导出的 ONNX 就只有 `branch_a`。之后无论输入什么，OM 永远走 `branch_a`。这类问题在链路上不会被任何一步拦住：ONNX 结构合法、ATC 转换成功、msame 对齐也通过——因为对齐用的输入同样走 `branch_a`。只有换到会走另一支的输入时结果才是错的。

**三种处理方式，按成本排序。**

改成无分支的等价写法。两支都算，用 `torch.where` 选：

```python
def forward(self, x):
    cond = (x.sum() > 0)
    return torch.where(cond, self.branch_a(x), self.branch_b(x))
```

代价是两支都要计算。分支计算量小的时候这是最省事的做法，而且对编译器友好——静态图，可以充分优化。

用 `torch.jit.script` 编译含分支的部分。script 做真正的语法分析，能把 `if` 保留成 ONNX 的 `If` 算子：

```python
scripted = torch.jit.script(model)
torch.onnx.export(scripted, dummy_input, "model.onnx", opset_version=11)
```

但要确认 CANN 支持 `If` 算子，且动态控制流会限制 ATC 的融合优化。script 对代码写法也有约束，不是所有模型都能直接通过。

拆成多个模型，把分支判断放到 Python 侧。业务代码里判断走哪支，分别加载对应的 OM。控制流完全脱离模型，每个 OM 都是纯静态的、可充分优化。代价是要管理多个 OM 和加载逻辑。

**判断标准**：分支是数据无关的（比如根据配置项选择），拆成多个模型最干净；分支是数据相关但计算轻，用 `torch.where`；确实需要在模型内部保留数据相关控制流，才考虑 script。

---

## 提交清单

- 导出脚本（含第 1 题的动态 batch 版本）
- ATC 命令与完整日志（含第 3 题的错误日志）
- 对齐 ① 与对齐 ② 的输出
- 第 2 题：说明观察到的现象，以及为什么删掉 `eval()` 后对齐仍可能通过
- 第 4 题：三个文件的实际大小
- 硬件型号与 CANN 版本

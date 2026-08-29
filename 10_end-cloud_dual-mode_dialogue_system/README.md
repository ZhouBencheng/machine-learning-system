# Lab 10：端云双模式大模型对话应用

## 实验目标

- 在云端下载 Qwen2-7B-Instruct，使用 MindIE 提供 OpenAI 兼容对话接口。
- 在开发板加载 DeepSeek-R1-Distill-Qwen-1.5B-FP16，并启动 Gradio 聊天页面。
- 在同一页面切换端侧与云端两种推理模式，记录两条路径的响应与对话表现。

## 实验环境

### 云端环境

云端镜像提供 CANN、NPU 驱动、MindIE-LLM 3.0 和 <code>mindie-3.0.0</code> Conda 环境。打开 <code>10.01_cloud_mindie_service_setup.ipynb</code> 后，先用已有 Kernel 运行注册单元；它会安装固定版本的 ModelScope，并注册 <code>Python (Lab10 Cloud MindIE)</code>。切换到这个 Kernel、重启后再从头运行 Notebook。

本实验使用逻辑卡 <code>0</code>。Notebook 下载 Qwen2-7B-Instruct，复制 MindIE 内置配置模板，写入模型路径与单卡配置，再启动服务。模型下载需要访问 ModelScope 的网络和约 20 GiB 可用空间。MindIE 监听云端 <code>127.0.0.1:1025</code>，云端 Notebook 通过 <code>/v1/models</code> 和 <code>/v1/chat/completions</code> 验证模型加载与文本生成。

### 开发板环境

端侧 Notebook 复用开发板镜像中的 Base Conda。它检查 MindSpore、MindNLP、Gradio、CANN 和 NPU，并注册 <code>Lab 10 Edge (base Conda)</code>。切换到该 Kernel 后，从第一个代码单元开始执行。

端侧模型从 Modelers 下载到 Lab 10 目录。Gradio 以 <code>share=False</code> 启动并绑定 <code>0.0.0.0:7860</code>；在实验局域网中打开：

~~~text
http://<开发板 IP>;:7860
~~~

### 实验目录与 SSH 访问密钥

课程仓库中的文件如下：

~~~text
10_end-cloud_dual-mode_dialogue_system/
├── README.md
├── 10.01_cloud_mindie_service_setup.ipynb
├── 10.02_edge_cloud_dual_mode_dialogue.ipynb
├── images/
│   └── mindie-service-architecture.png
~~~

云端运行目录由 <code>LAB10_CLOUD_ROOT</code> 指定。使用下列目录时，在启动 JupyterLab 的终端执行：

~~~bash
export LAB10_CLOUD_ROOT="/home/<云端用户名>/work/10_end-cloud_dual-mode_dialogue_system"
jupyter lab
~~~

云端 Notebook 在运行目录中创建 <code>models/</code>、<code>cache/</code>、<code>configs/</code>、<code>logs/</code>、<code>runtime/</code> 与 <code>pids/</code>，分别保存模型文件、下载缓存、MindIE 配置、服务日志、运行时文件和进程号。开发板运行目录由 <code>10.02</code> 第一个代码单元中的 <code>LAB_DIR</code> 指定，端侧 Notebook 创建 <code>models/</code>、<code>logs/</code> 与 <code>state/</code>，保存端侧模型、运行日志和实验记录。

PEM 文件放在 <code>LAB_DIR</code> 内，与 <code>10.02_edge_cloud_dual_mode_dialogue.ipynb</code> 并列。<code>KeyPair-*.pem</code> 表示课程发放的密钥文件名，<code>*</code> 位置填写实际后缀，例如 <code>KeyPair-abc123.pem</code>。在可信主机上设置以下变量后传输文件：

~~~bash
EDGE_USER='<开发板用户名>'
EDGE_HOST='<开发板 IP 或主机名>'
EDGE_ROOT='<10.02 第一个代码单元中配置的 LAB_DIR>'
KEY_FILE='/本机/KeyPair-<实际后缀>.pem'

scp "$KEY_FILE" "${EDGE_USER}@${EDGE_HOST}:${EDGE_ROOT}/"
ssh "${EDGE_USER}@${EDGE_HOST}" "chmod 600 '${EDGE_ROOT}/$(basename "$KEY_FILE")'"
~~~

<code>LAB10_CLOUD_KEY_PATH</code> 定义端侧 Notebook 读取的 PEM 路径：

~~~bash
export LAB10_CLOUD_KEY_PATH='<LAB_DIR>/KeyPair-<实际后缀>.pem'
jupyter lab
~~~

PEM 文件由开发板本地读取；仓库根目录的 <code>.gitignore</code> 匹配 <code>*.pem</code>。模型、日志和状态文件保留在各自的运行目录。

## 实验原理

### 端云调用链路

~~~text
浏览器
  │  http://<开发板 IP>:7860
  ▼
开发板上的 Gradio 页面
  ├── 端侧模式：MindSpore + DeepSeek-R1-Distill-Qwen-1.5B-FP16
  └── 云端模式：127.0.0.1:1025
                       │
                       └── SSH 本地转发 ──► 云端 127.0.0.1:1025
                                                   │
                                                   ▼
                                            MindIE + Qwen2-7B-Instruct
~~~

两个 <code>127.0.0.1</code> 分别属于开发板和云端主机。浏览器访问开发板上的 Gradio 页面，云端模式由 SSH 将开发板本机端口连接到云端 MindIE 服务。

### 两种推理后端

| 后端 | 模型 | 推理位置 | 前端调用方式 |
| --- | --- | --- | --- |
| 端侧 | DeepSeek-R1-Distill-Qwen-1.5B-FP16 | 开发板的 Notebook 进程 | 直接调用 MindSpore 模型 |
| 云端 | Qwen2-7B-Instruct | 云端 MindIE 服务 | 通过 SSH 隧道调用 OpenAI 兼容接口 |

### SSH 本地端口转发

<code>ssh -L</code> 在开发板创建本地监听端口，发送到 <code>127.0.0.1:1025</code> 的 HTTP 请求沿 SSH 连接抵达云端同一回环端口。MindIE 使用云端本机监听地址，端侧页面使用开发板本机地址；浏览器始终访问开发板上的 Gradio 页面。

## 实验流程

### 1. 部署云端 MindIE 服务

在云端运行 <code>10.01_cloud_mindie_service_setup.ipynb</code>，完成 Kernel 注册、模型快照下载、MindIE 配置生成、服务启动和接口验证。<code>/v1/models</code> 返回 <code>qwen2-7b-instruct</code> 后，云端服务可供端侧页面调用。

### 2. 准备开发板环境与访问密钥

将 PEM 文件传入开发板的 <code>LAB_DIR</code>，设置 <code>LAB10_CLOUD_KEY_PATH</code>，再运行 <code>10.02_edge_cloud_dual_mode_dialogue.ipynb</code>。Notebook 注册 Base Conda Kernel、下载端侧模型并完成 JIT 编译。

### 3. 启动端云双模式对话页面

在端侧 Notebook 中建立 SSH 隧道。<code>/v1/models</code> 返回云端模型名后，启动 Gradio 页面，在浏览器中打开 <code>http://<开发板 IP>:7860</code>。

### 4. 对比两种推理模式

先在端侧模式提一个短问题，再切换到云端模式，用同一问题重新提问。继续追问一次，分别记录模型、首轮耗时、后续耗时、回答长度和上下文衔接情况。

## 运行检查

| 位置 | 检查项 | 预期结果 | 记录位置 |
| --- | --- | --- | --- |
| 云端 | <code>GET /v1/models</code> | 列出 <code>qwen2-7b-instruct</code> | <code>logs/mindie_llm_server.log</code> |
| 开发板 | SSH 隧道后的 <code>GET /v1/models</code> | 返回同一模型名 | <code>logs/cloud_tunnel.log</code> |
| 浏览器 | <code>http://<开发板 IP>:7860</code> | 显示 Gradio 聊天页面 | 端侧 Notebook 输出 |

## 实验总结

本实验在一个 Gradio 页面中连接了端侧模型和云端 MindIE 服务。端侧模式包含模型加载与 JIT 编译，云端模式包含 SSH 转发与网络请求；观察记录将这些阶段拆开，能够更清楚地描述两种推理路径的差异。

## 参考资料

- [MindIE 3.0 文本生成推理快速入门（以 Qwen2-7B 为例）](https://www.hiascend.com/document/detail/zh/mindie/300/quickstart/docs/zh/user_guide/quick_start/quick_start.md)
- [MindIE 3.0 单机服务部署说明](https://www.hiascend.com/document/detail/zh/mindie/300/mindiemotor/motordev/user_guide/service_deployment/single_machine_service_deployment.md)
- [MindIE 3.0 OpenAI 兼容接口说明](https://www.hiascend.com/document/detail/zh/mindie/300/mindiellm/llmdev/mindie_service0078.html)

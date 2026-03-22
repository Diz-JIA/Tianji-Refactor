# Tianji-Refactor: 人情世故大模型轻量化重构版

本项目是基于开源项目 [Tianji](https://github.com/SocialAI-tianji/Tianji) 的深度重构版本。针对原项目在现代开发环境（尤其是 **Apple Silicon M4 架构**）下的运行痛点，进行了架构去冗余与兼容性修复。

>**入门文档**：[入门点这里](https://tianji.readthedocs.io/en/latest/index.html)

## 🌟 项目亮点

* **架构进化**：移除重型 MetaGPT 框架依赖，回归大模型应用本质，显著提升初始化速度与运行稳定性。
* **双模驱动**：提供 **Prompt-based**（纯提示词工程）与 **RAG-based**（基于 LangChain 的检索增强生成）两个独立版本，满足不同业务场景。
* **M4 原生优化**：深度适配 Mac M4 芯片，解决了底层异步网络库 `httpx` 与代理劫持的冲突问题。
* **全栈协议对齐**：重构了 Gradio 5.x+ 的交互逻辑，适配最新的 `role-content` 字典协议，彻底解决 UI 渲染异常。

---

## 🛠 核心修改说明

本项目相较于原版 `Tianji` 进行了以下关键手术：

| 修改项 | 原项目 (Tianji) | 本项目 (Refactor) | 解决的痛点 |
| :--- | :--- | :--- | :--- |
| **核心框架** | MetaGPT (Agent 架构) | **LangChain 0.2.x** | 解决了 M4 环境下 Pydantic 静态校验引发的初始化崩溃。 |
| **依赖逻辑** | 强绑定特定旧版库 | **版本解耦 & 精准对齐** | 解决了 Pydantic v1/v2 冲突及 `ImportError: IncEx` 等难题。 |
| **RAG 引擎** | 框架内嵌检索 | **ChromaDB + Sentence-Transformers** | 支持本地高效向量化，无需频繁调用在线 Embedding 接口。 |
| **网络层** | 自动抓取系统代理 | **环境隔离隔离补丁** | 解决了 `AsyncClient` 在代理环境下的 `TypeError` 参数冲突。 |
| **交互协议** | 元组列表格式 `[(u, b)]` | **Role-Content 字典格式** | 适配最新版 Gradio Chatbot 组件，防止界面报错。 |

---

## 🚀 运行指南

### 1. 环境准备

建议在 **Miniconda** 环境下运行，以确保依赖隔离。

```bash
# 创建环境 (推荐 Python 3.11)
conda create -n tianji_refactor python=3.11
conda activate tianji_refactor

# 安装核心依赖 (版本号经过 M4 环境严格验证)
pip install langchain==0.2.15 \
            langchain-community==0.2.14 \
            langchain-chroma \
            sentence-transformers \
            langchain-openai \
            gradio \
            pydantic==2.10.0 \
            python-dotenv
```

### 2. 配置密钥

删除`.env.example`文件后缀`.example` ，并填入你的 API 信息：

```text
# 以 SiliconFlow 或智谱 AI 为例
ZHIPUAI_API_KEY=你的API_Key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
```

### 3. 启动项目

本项目分为两个运行入口：
```bash
   cd run
```
* **运行 Prompt 版**（轻量化交互）：
    ```bash
    python demo_prompt_version.py
    ```
* **运行 RAG 版**（基于 7 大场景知识库）：
    ```bash
    python demo_rag_langchain_all.py
    ```

---

## 📂 项目结构说明

```text
Tianji-Refactor/
├── demo_rag_langchain_all.py  # 重构后的 RAG 入口
├── demo_prompt_version.py     # 重构后的 Prompt 入口
├── tianji/                   # 核心业务逻辑文件夹
├── temp/                     # 自动生成的本地向量数据库 (ChromaDB)
├── .env.example              # 密钥配置模板
└── README.md                 # 本说明文档
```

---

## 🤝 鸣谢与致敬

本项目核心知识库与原始创意源自 [Tianji (SocialAI-Tianji)](https://github.com/SocialAI-tianji/Tianji)。原项目在大模型垂直领域（社交智慧）的开源探索令人钦佩。本项目仅作为技术层面的适配与架构重构实践。

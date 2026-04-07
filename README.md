# QuizMind

QuizMind 是一个基于 `Streamlit + LangChain` 的智能练习系统，支持从文本/文件/URL 生成题目，自动批改与学习反馈，并包含“场景模拟”对话式工程面试模块。

## 核心能力

- 多源输入
  - 粘贴文本
  - 上传文件（`pdf` / `docx` / `md` / `txt`，支持多文件）
  - 网页 URL 抓取
- 内容解析与出题
  - 自动提取知识点、概念、分段
  - 支持单选、多选、填空、简答、判断
  - 支持题量、难度比例、题型比例配置
- 作答与评测
  - 练习模式
  - 考试模式（倒计时，到时自动交卷）
  - 客观题规则判分 + 主观题 LLM 语义评分
- 学习闭环
  - 错题与知识点掌握分析
  - 强化训练题
  - 收藏题单、学习看板
- 数据与管理
  - 题库存档、检索、删除、导出（JSON/Markdown/PDF）
  - 批量生成队列（重试/清理）
  - 记忆库（FAISS）检索组题
- 场景模拟（工程面试）
  - 引导模式 / 严苛模式
  - 多轮追问直到通过或结束
  - 输出评分、优劣势与改进建议

## 技术栈

- 前端：`streamlit`
- LLM 编排：`langchain` / `langchain-openai`
- 文本处理：`requests` / `beautifulsoup4` / `pypdf` / `python-docx`
- 向量存储：`faiss-cpu`
- 导出：`reportlab`

## 目录结构

```text
QuizMind/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ 操作步骤.md
├─ quizmind/
│  ├─ llm.py
│  ├─ services.py
│  ├─ models.py
│  ├─ content.py
│  ├─ memory.py
│  ├─ quiz_bank.py
│  ├─ generation_queue.py
│  ├─ exporter.py
│  └─ user_store.py
└─ .quizmind_runtime/   # 运行时自动生成
```

## 环境要求

- Python `3.10+`（推荐 `3.11+`）
- Windows / macOS / Linux 均可运行

## 安装与启动

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

启动后打开终端提示的本地地址（通常是 `http://localhost:8501`）。

## 配置说明（.env）

复制 `.env.example` 为 `.env` 后，至少配置 API Key。

```env
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen-72B-Chat
SILICONFLOW_EMBEDDING_MODEL=
```

说明：

- `SILICONFLOW_API_KEY` 为空时，系统会回退到本地规则解析/生成能力（部分高级能力受限）。
- `SILICONFLOW_EMBEDDING_MODEL` 为空时，记忆库会回退为本地哈希嵌入（无需远程 embedding）。
- 你也可以在页面左侧“模型与 API 设置”里临时覆盖 `.env` 配置并保存运行时配置。

## 使用入口

- `智能练习`：完整学习流程（输入 -> 出题 -> 作答 -> 反馈）
- `场景模拟`：工程师面试追问对话流程

详细操作请查看：

- [操作步骤.md](./操作步骤.md)

## 常见问题

- 上传 `.doc` 报错
  - 仅支持 `.docx`，请先转换格式。
- URL 无法抓取
  - 目标网站可能有反爬限制或网络不可达。
- 题目质量不稳定
  - 提高输入内容质量，或在侧边栏调整题型/难度比例后重新生成。
- 记忆模式无内容
  - 先在智能练习中完成一次解析并“保存到记忆库”。

## 运行数据位置

运行后会自动创建 `.quizmind_runtime/`，主要包括：

- `quiz_bank/`：历史题目与索引
- `memory/`：FAISS 记忆库
- `queue/`：批量任务队列
- `user_data/`：收藏、反馈、学习会话
- `settings/`：运行时模型配置
- `quizmind.log`：系统日志

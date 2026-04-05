# QuizMind

一个基于 LangChain 的智能刷题系统，支持内容解析、自动出题、自动批改，以及基于向量记忆库的随机复习出题。

## 当前能力

- 输入内容：文本、文件、URL
- LangChain 解析内容：提取知识点、重点概念、难度
- LangChain 自动出题：单选、多选、填空、简答、判断
- 自动批改：客观题规则判分，主观题 LLM 语义评分
- 学习反馈：错题本、知识点掌握情况、强化练习题
- 向量记忆模式：将历史学习内容写入向量库，并按随机或检索方式召回内容出题

## 技术栈

- 前端：`Streamlit`
- LLM 编排：`LangChain`
- 聊天模型：硅基流动 OpenAI Compatible API
- 向量库：`FAISS`
- 嵌入模型：
  - 优先使用 `SILICONFLOW_EMBEDDING_MODEL`
  - 未配置时自动回退到本地哈希嵌入，保证功能可跑通

## 环境变量

```env
SILICONFLOW_API_KEY=你的密钥
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen-72B-Chat
SILICONFLOW_EMBEDDING_MODEL=
```

## 快速启动

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## 使用方式

1. 在“当前内容”模式输入资料并点击“解析内容”。
2. 需要积累记忆时，点击“保存到记忆库”。
3. 切换到“记忆模式”后，可以输入检索词，也可以留空做随机复习。
4. 点击“生成题目”，系统会基于向量召回的历史片段自动出题。

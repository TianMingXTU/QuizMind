# QuizMind

一个基于 LangChain + Streamlit 的智能练习与考试系统。  
支持内容解析、自动出题、自动评测、错题复盘、向量记忆、题库复用，以及可选的“互动知识点网页”。

## 主要功能

- 多源输入
  - 粘贴文本
  - 上传文件（`pdf / docx / md / txt`，支持多文件）
  - URL 抓取
- AI 解析内容
  - 知识点提取
  - 重点概念整理
  - 难度分级
- 自动出题（可配置数量/题型比例/难度比例）
  - 单选题
  - 多选题
  - 填空题
  - 简答题
  - 判断题
- 作答模式
  - 练习模式（整页连续作答）
  - 考试模式（限时自动交卷）
- 自动批改
  - 客观题规则判分
  - 主观题 LLM 语义评分（支持批量评分）
- 学习反馈
  - 错题本
  - 知识点掌握情况
  - 复习建议
  - 强化训练题
- 题库能力
  - 自动保存已生成题目
  - 按文件名/日期/标签搜索
  - 删除题目记录
  - 导出 `json / md / pdf`
- 批量生成队列
  - 多文件入队
  - 并发处理
  - 失败重试
  - 清理已完成任务
- 向量记忆（FAISS）
  - 保存解析后的内容到记忆库
  - 记忆检索出题/随机复习
- 可选互动知识点网页（新增）
  - 在“输入与解析”区可一键生成互动小网页并嵌入页面
  - 默认关闭，启用后才会调用该能力

## 技术栈

- 前端：`Streamlit`
- LLM 编排：`LangChain`
- 模型接入：OpenAI-Compatible API（默认硅基流动）
- 向量库：`FAISS`
- 导出：`reportlab`（PDF）

## 环境变量

复制 `.env.example` 为 `.env`，并按需配置：

```env
SILICONFLOW_API_KEY=你的密钥
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen2.5-72B-Instruct

# 可选：向量嵌入模型
SILICONFLOW_EMBEDDING_MODEL=
```

说明：

- 未配置 `SILICONFLOW_EMBEDDING_MODEL` 时，会自动回退到本地哈希嵌入（可离线运行基础记忆流程）。

## 快速启动

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## 使用流程（推荐）

1. 在“输入与解析”上传或粘贴学习内容，点击“解析内容”。
2. 可选：点击“保存到记忆库”。
3. 在“组卷与题库/队列”点击“生成题目”。
4. 在“作答”区域直接下滑完成所有题目并提交。
5. 在“结果复盘”查看错题、掌握度与强化题建议。
6. 可选：在“题库管理”中搜索、删除、导出历史题目。

## 注意事项

- `.doc`（旧版 Word）不支持，请先转换为 `.docx`。
- 互动知识网页是可选能力，关闭时不会额外消耗这部分调用。
- 建议优先开启“优先使用已保存题目”，可显著减少重复 API 开销。

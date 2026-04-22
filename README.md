# QuizMind

QuizMind 是一个面向个人学习、刷题训练和工程面试准备的智能练习系统。  
它不是单纯的题库工具，而是把“内容输入 -> 出题 -> 作答 -> 诊断 -> 强化 -> 续练”做成闭环，并在这个闭环里加入一套可持续推进的 `QuizMind CORE 学习法`。

## 这个产品适合谁

- 想把笔记、资料、网页内容快速变成练习题的人
- 想做错题复盘和针对性强化训练的人
- 想把零散学习内容沉淀进记忆库，后续反复检索训练的人
- 想做工程师场景模拟面试的人

## 产品特色

- 一键学习闭环
  - 当前内容支持 `一键解析并出题`
  - 记忆模式支持 `一键从记忆出题`
  - 结果页支持 `继续智能续练`
- 学习方法驱动
  - 根据错因自动进入 `理解建图 / 主动回忆 / 推理纠偏 / 细节校准 / 表达固化 / 综合进阶`
  - 不同阶段会影响题型配比、结果页反馈、作答引导和下一轮训练建议
- 可持续训练
  - 支持 `继续上次训练`
  - 支持每日训练、收藏题单、错题重练、智能强化训练
  - 支持阶段推进轨迹展示
- 多输入源
  - 粘贴文本
  - 上传 `pdf / docx / md / txt`
  - 网页 URL
  - 记忆库检索
- 多结果输出
  - 练习模式 / 考试模式
  - 自动评分
  - 错因分析
  - 知识点掌握情况
  - 学习趋势和能力画像

## 主要功能

- 智能解析学习内容
- AI / 本地规则混合出题
- 题库缓存、历史记录检索、导出
- 记忆库保存与召回
- 错题重练、智能强化训练、智能续练
- 收藏题单与个人学习数据看板
- 工程师场景模拟面试

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

启动后访问终端输出的本地地址，通常是 `http://localhost:8501`。

## 推荐使用路径

### 路径 1：把当前资料直接变成练习

1. 选择输入方式并提供学习内容
2. 点击 `一键解析并出题`
3. 完成作答
4. 查看结果页中的学习阶段、错题分析和阶段推进
5. 点击 `继续学习：...` 或 `一键智能强化训练`

### 路径 2：把内容沉淀进记忆库后反复练

1. 在当前内容模式中点击 `保存到记忆库`
2. 切换到侧边栏 `记忆模式`
3. 输入检索词
4. 点击 `一键从记忆出题`

### 路径 3：工程师面试模拟

1. 切换到 `场景模拟`
2. 选择模板或输入自定义场景
3. 选择 `引导模式` 或 `严格模式`
4. 开始多轮追问

## QuizMind CORE 学习法

系统不会只给你一个分数，还会自动判断你当前更适合哪种学习方式：

- `理解建图`
  - 适合概念混淆、边界不清
- `主动回忆`
  - 适合记忆提取不足、看过但答不出
- `推理纠偏`
  - 适合理解大致有了，但推理链不稳定
- `细节校准`
  - 适合粗心、审题不稳、限定词漏看
- `表达固化`
  - 适合简答思路对但表达不完整
- `综合进阶`
  - 适合整体较稳，需要扩覆盖和提上限

这套学习法已经影响：

- 每日训练推荐
- 智能续练
- 结果页复盘重点
- 作答区引导提示
- 阶段推进轨迹

## 技术栈

- `Streamlit`
- `LangChain`
- `OpenAI-compatible API` 接口
- `FAISS` 记忆检索
- `Pydantic v2`
- `ReportLab` 导出 PDF

## 目录结构

```text
QuizMind/
├─ app.py
├─ README.md
├─ 操作步骤.md
├─ requirements.txt
└─ quizmind/
   ├─ cache.py
   ├─ content.py
   ├─ exporter.py
   ├─ generation_queue.py
   ├─ llm.py
   ├─ logger.py
   ├─ memory.py
   ├─ models.py
   ├─ prompt_center.py
   ├─ quiz_bank.py
   ├─ schemas.py
   ├─ services.py
   └─ user_store.py
```

运行时数据会写到 `.quizmind_runtime/`，包括题库、记忆库、用户特征和日志。

## 配置说明

常见环境变量：

```env
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2
SILICONFLOW_EMBEDDING_MODEL=
QUIZMIND_STRICT_AI_GENERATION=true
```

说明：

- 未配置 `SILICONFLOW_API_KEY` 时，部分 AI 能力会降级到本地规则
- 未配置 `SILICONFLOW_EMBEDDING_MODEL` 时，记忆库会回退到本地哈希嵌入
- 开启 `QUIZMIND_STRICT_AI_GENERATION=true` 时，AI 失败不会自动降级到本地规则

## 当前交付状态

目前主流程已经可用：

- 当前内容一键出题
- 记忆模式一键出题
- 每日训练
- 继续上次训练
- 智能续练
- 错题重练
- 收藏题单
- 学习看板
- 场景模拟

## 已知边界

- 当前仓库没有自动化测试集，回归主要依赖编译、导入和手动流程验证
- 生成质量仍然受输入材料质量和外部模型稳定性影响
- PDF 导出在中文字体环境不完整时可能出现显示问题，必要时优先使用 Markdown / JSON 导出

## 更多说明

技术部署和排障请看：[操作步骤.md](./操作步骤.md)

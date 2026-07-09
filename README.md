# PDF RAG MVP

第一层先把本地 PDF 解析成带页码的 JSON。第二层把解析结果切成 chunks，但页码只作为引用 metadata，不作为强制切块边界。

## 本地解析

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

解析 PDF：

```bash
python3 scripts/parse_pdf.py /path/to/course.pdf
```

默认输出：

```text
parsed/course.json
```

也可以指定输出位置：

```bash
python3 scripts/parse_pdf.py /path/to/course.pdf -o parsed/course.json
```

## JSON 结构

核心字段：

```json
{
  "schema_version": "0.1.0",
  "source": {
    "path": "/path/to/course.pdf",
    "file_name": "course.pdf",
    "size_bytes": 123456,
    "sha256": "..."
  },
  "parser": {
    "name": "pdfplumber",
    "version": "0.11.9"
  },
  "document": {
    "page_count": 10,
    "metadata": {}
  },
  "pages": [
    {
      "page_index": 0,
      "page_number": 1,
      "char_count": 800,
      "word_count": 320,
      "has_text": true,
      "needs_ocr": false,
      "text": "..."
    }
  ]
}
```

其中 `page_number` 是之后回答问题时引用页码的基础，例如“见第 3 页”。

## 本地切块

解析完成后，把 `parsed/*.json` 切成适合检索的小段：

```bash
python3 scripts/chunk_json.py parsed/course.json
```

默认输出：

```text
chunks/course_chunks.json
```

切块原则：

- 先按段落/句子形成语义单元。
- 再按目标长度打包成 chunk。
- 页码只保存在 `source_pages`、`start_page`、`end_page`、`page_range` 里。
- 不会把“每一页”粗暴当成一个 chunk。

chunk 输出示例：

```json
{
  "chunk_id": "course_chunk_0000",
  "text": "一个相对完整的语义片段...",
  "source_pages": [1, 2],
  "page_range": "1-2",
  "start_page": 1,
  "end_page": 2,
  "char_count": 760
}
```

## 关键词检索

切块完成后，可以先用关键词检索找到相关 chunk 和引用页码：

```bash
python3 scripts/search_chunks.py chunks/course_chunks.json "RAG data pipeline"
```

返回内容包括：

- 命中的 `chunk_id`
- 相关分数 `score`
- 来源页码 `pages`
- 片段预览 `snippet`

这一步还不是向量检索，只是轻量 BM25 风格的关键词检索，用来先打通：

```text
用户问题 -> 找到相关 chunk -> 返回引用页码
```

## 本地向量检索实验

为了理解 embedding 和余弦相似度，可以先做一个完全本地的教学实验：

```bash
python3 scripts/vector_search_demo.py \
  chunks/resume2_chunks.json \
  "Has he built an ETL data processing project?"
```

这个脚本会：

1. 把最多 10 个 chunk 转成 TF-IDF 向量。
2. 把 query 转成同一个向量空间里的向量。
3. 本地计算 query 和每个 chunk 的 cosine similarity。
4. 按相似度返回 Top-K，并保留来源页码。

这一步：

- 不使用向量数据库。
- 不使用 Chroma 或 FAISS。
- 不调用 LLM。
- 不需要新增 Python 依赖。

注意：这里的 TF-IDF 是为了把“文本 -> 向量 -> 余弦排序”过程展示清楚。它是教学版稀疏 embedding，还不能像神经网络 embedding 那样理解真正的语义相似。

### 措辞和假 Chunk 实验

用多种表达方式测试同一个意思，再加入一个重复关键词的假 Chunk：

```bash
python3 scripts/tfidf_wording_experiment.py chunks/resume2_chunks.json
```

也可以自己提供多个问题：

```bash
python3 scripts/tfidf_wording_experiment.py chunks/resume2_chunks.json \
  --query "How does retrieval work in RAG?" \
  --query "How does RAG find useful passages?"
```

这个实验不会修改原始 Chunk JSON。它用来观察 TF-IDF 对关键词、
词频、同义改写和跨语言问题的反应。

## 真实语义 Embedding

安装语义检索的额外依赖：

```bash
.venv/bin/python -m pip install -r requirements-semantic.txt
```

运行真实的神经网 Embedding 检索：

```bash
.venv/bin/python scripts/semantic_search.py \
  chunks/resume2_chunks.json \
  "Has he built a system that moves and transforms data?"
```

默认使用 `intfloat/multilingual-e5-small`：

- 支持中文和英文。
- Query 和 Chunk 都会转换为 384 维稠密向量。
- 使用归一化向量的点积计算余弦相似度。
- 不使用向量数据库，不调用 LLM。
- 第一次运行会把模型下载到项目的 `.models/`，之后会使用本地缓存。

可以用同一个问题分别运行 `vector_search_demo.py` 和
`semantic_search.py`，比较 TF-IDF 词面匹配与稠密语义检索的 Top-K 排序。

使用和 TF-IDF 实验完全相同的五种问法和假 Chunk：

```bash
.venv/bin/python scripts/embedding_wording_experiment.py chunks/resume2_chunks.json
```

对比时主要观察：同义改写是否仍能找到同一个 Chunk，中文问题能否检索英文
Chunk，以及重复关键词的假 Chunk 是否仍会排在第一。

## Rerank 二阶段重排

语义检索先用 embedding 快速召回一批候选 chunk，再用 cross-encoder
同时读取“问题 + 候选 chunk”来重新打分：

```bash
.venv/bin/python scripts/rerank_search.py \
  chunks/resume2_chunks.json \
  "Has he built a system that moves and transforms data?" \
  --candidate-k 20 \
  --top-k 3
```

默认流程：

- 第一阶段：`intfloat/multilingual-e5-small` 做 dense embedding 召回。
- 第二阶段：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` 做 rerank。
- 输出会同时显示 `rerank_score`、原始 embedding `base_rank` 和
  `base_similarity`，方便观察重排前后的变化。
- 不使用向量数据库，不调用 LLM。
- 默认只使用项目 `.models/` 里的本地缓存，避免离线时等待网络重试。
- 如果本地还没有模型，联网时加 `--allow-download` 下载一次。

可以替换 rerank 模型：

```bash
.venv/bin/python scripts/rerank_search.py \
  chunks/resume2_chunks.json \
  "RAG retrieval optimization" \
  --rerank-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --allow-download
```

## Context Builder

Rerank 之后，可以把 Top-K chunk 整理成可直接喂给 LLM 的上下文：

```bash
.venv/bin/python scripts/context_builder.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 3
```

输出会包含：

- `Context`：按 rerank 顺序排列的 chunk 正文。
- `[S1]`、`[S2]` 这样的 source id，方便回答时引用。
- `pages`：原 PDF 页码。
- `rerank`、`base_rank`：方便检查为什么这些 chunk 被选进上下文。

也可以直接生成一个完整 prompt：

```bash
.venv/bin/python scripts/context_builder.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 3 \
  --prompt
```

如果要给后续程序读取，用 JSON：

```bash
.venv/bin/python scripts/context_builder.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 3 \
  --json
```

如果本地还没有 embedding 或 rerank 模型，第一次运行时加：

```bash
--allow-download
```

## DeepSeek 生成答案

Context Builder 之后，可以接 DeepSeek API 生成最终回答。

先创建本地 `.env`：

```bash
cp .env.example .env
```

然后把 `.env` 里的 `DEEPSEEK_API_KEY` 改成你的 DeepSeek API key。
`.env` 已经在 `.gitignore` 里，不会被提交。

运行问答：

```bash
.venv/bin/python scripts/answer_question.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 1
```

默认使用：

- API base URL：`https://api.deepseek.com`
- LLM：`deepseek-v4-flash`
- thinking：`disabled`
- 输入上下文：来自 `context_builder.py` 的 `[S1]`、`pages` 和 chunk 正文

如果想打开 thinking：

```bash
.venv/bin/python scripts/answer_question.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 1 \
  --thinking enabled \
  --reasoning-effort high
```

如果想输出 JSON：

```bash
.venv/bin/python scripts/answer_question.py \
  chunks/resume2_chunks.json \
  "他做过什么数据处理项目？" \
  --candidate-k 3 \
  --top-k 1 \
  --json
```

## 当前边界

- 可以处理内置文字层的 PDF。
- 扫描版 PDF 可能提取不到文字，会被标记为 `needs_ocr: true`。
- 当前不处理向量数据库、不做大模型回答、不做 OCR。
- 语义模型最多读取 512 tokens，过长 Chunk 会被截断。

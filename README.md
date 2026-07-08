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

## 当前边界

- 可以处理内置文字层的 PDF。
- 扫描版 PDF 可能提取不到文字，会被标记为 `needs_ocr: true`。
- 当前不处理向量数据库、不做大模型回答、不做 OCR。
- 语义模型最多读取 512 tokens，过长 Chunk 会被截断。

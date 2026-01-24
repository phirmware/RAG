# RAGBench Evaluation System - Architecture Guide

This document explains how the RAGBench evaluation system works, step by step.

---

## Table of Contents

1. [Overview](#overview)
2. [The Dataset](#the-dataset)
3. [Data Flow](#data-flow)
4. [Script-by-Script Breakdown](#script-by-script-breakdown)
5. [Evaluation Metrics Explained](#evaluation-metrics-explained)
6. [The Dashboard](#the-dashboard)
7. [How to Experiment](#how-to-experiment)

---

## Overview

This system evaluates how well your RAG (Retrieval-Augmented Generation) pipeline retrieves relevant documents. It uses the **Vectara Open RAGBench** dataset, which contains:

- **Questions** about arXiv research papers
- **Ground truth answers** for each question
- **Relevance judgments** (qrels) that tell us exactly which document section answers each question

The key insight: instead of guessing if we retrieved the "right" document, we have **ground truth** that tells us exactly which `(document_id, section_id)` pair is correct for each question.

---

## The Dataset

### Source
The dataset comes from [Vectara's Open RAGBench](https://huggingface.co/datasets/vectara/open_ragbench) on HuggingFace.

### Files Downloaded

```
data/
├── queries.json      # Questions to ask
├── answers.json      # Correct answers (for future generation evaluation)
├── qrels.json        # Query RELevance judgments (ground truth)
├── pdf_urls.json     # Original PDF sources
└── corpus/           # The actual documents
    ├── 2401.01872v2.json
    ├── 2401.02247v4.json
    └── ... (one file per arXiv paper)
```

### File Formats

#### queries.json
```json
{
  "query-uuid-123": {
    "query": "What is the main contribution of this paper?",
    "type": "abstractive",     // or "extractive"
    "source": "text"           // or "text-image", "text-table", etc.
  }
}
```

- **type**: "abstractive" means the answer requires synthesizing info; "extractive" means you can copy it directly
- **source**: tells you what modality is needed (text only, or text + images/tables)

#### qrels.json (Query RELevance judgments)
```json
{
  "query-uuid-123": {
    "doc_id": "2401.01872v2",
    "section_id": 3
  }
}
```

This is the **ground truth**. For query "query-uuid-123", the correct answer is in document "2401.01872v2", section 3.

#### corpus/*.json (Document files)
```json
{
  "id": "2401.01872v2",
  "title": "Paper Title Here",
  "abstract": "The paper abstract...",
  "authors": ["Author 1", "Author 2"],
  "sections": [
    {
      "text": "Section 0 content...",
      "tables": { "table_1": "| Col1 | Col2 |..." },
      "images": { "fig_1": "base64-encoded-image" }
    },
    {
      "text": "Section 1 content..."
    }
  ]
}
```

---

## Data Flow

Here's how data flows through the system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. DOWNLOAD PHASE                           │
│                                                                 │
│   HuggingFace ──fetch──> queries.json, answers.json, qrels.json │
│                          corpus/*.json (document files)         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     2. INGESTION PHASE                          │
│                                                                 │
│   corpus/*.json ──parse──> sections ──chunk──> chunks           │
│                                           │                     │
│                                           ▼                     │
│   chunks ──embed (OpenAI)──> vectors ──store──> Qdrant DB       │
│                                                                 │
│   Each vector stored with metadata:                             │
│   { doc_id, section_id, chunk_id, text, title }                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. EVALUATION PHASE                          │
│                                                                 │
│   For each query in queries.json:                               │
│                                                                 │
│   query ──embed──> vector ──search Qdrant──> top-K results      │
│                                                   │             │
│                                                   ▼             │
│   Compare results to qrels.json ground truth:                   │
│   - Did we retrieve the correct doc_id + section_id?            │
│   - At what rank position?                                      │
│                                                   │             │
│                                                   ▼             │
│   Calculate metrics: MRR, nDCG, Recall@K, Precision@K           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. VISUALIZATION                             │
│                                                                 │
│   runs/*.json ──serve──> Dashboard UI                           │
│                                                                 │
│   - View metrics by run                                         │
│   - Compare runs side-by-side                                   │
│   - Filter by query type/source                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Script-by-Script Breakdown

### 1. download_dataset.js

**Purpose**: Fetch the RAGBench dataset from HuggingFace.

**How it works**:

```javascript
// Step 1: Download metadata files
const METADATA_FILES = ['queries.json', 'answers.json', 'qrels.json', 'pdf_urls.json'];

for (const filename of METADATA_FILES) {
    const url = `${BASE_URL}/${filename}`;
    const data = await fetch(url).then(r => r.json());
    fs.writeFileSync(filepath, JSON.stringify(data));
}

// Step 2: Figure out which corpus documents we need
// We only download documents that are referenced in qrels (the ground truth)
const docIds = new Set();
for (const queryId in qrels) {
    docIds.add(qrels[queryId].doc_id);  // e.g., "2401.01872v2"
}

// Step 3: Download each corpus document
for (const docId of docIds) {
    const url = `${BASE_URL}/corpus/${docId}.json`;
    // ... download and save
}

// Step 4: Create a sample subset (100 queries) for quick testing
// This lets you iterate faster during development
```

**Output**:
- `data/queries.json`, `data/answers.json`, `data/qrels.json`
- `data/corpus/*.json` (one file per paper)
- `data/sample/` (smaller subset for testing)

---

### 2. ingest.js

**Purpose**: Parse documents, chunk them, embed them, and store in Qdrant.

**How it works**:

```javascript
// Step 1: Create a fresh Qdrant collection
await qdrant.deleteCollection('ragbench');  // Start fresh
await qdrant.createCollection('ragbench', {
    vectors: { size: 3072, distance: 'Cosine' }  // 3072 = OpenAI embedding size
});

// Step 2: For each document...
for (const docId of documentIds) {
    const doc = loadDocument(docId);

    // Step 3: Extract sections
    // The abstract becomes section -1, then sections 0, 1, 2, etc.
    const sections = extractSections(doc);
    // sections = [
    //   { sectionId: -1, text: "Abstract text..." },
    //   { sectionId: 0, text: "Intro text..." },
    //   { sectionId: 1, text: "Methods text..." },
    // ]

    // Step 4: Chunk each section (semantic chunking)
    // Long sections get split into smaller pieces based on:
    // - Semantic similarity between sentences
    // - Token limits (max 400 tokens per chunk)
    for (const section of sections) {
        const chunks = await semanticChunkWithLimits(section.text, embedText, {
            threshold: 0.6,   // Split when similarity drops below 60%
            maxTokens: 400,   // Max tokens per chunk
            minTokens: 100,   // Min tokens before allowing split
        });

        // Step 5: Embed and store each chunk
        for (const chunkText of chunks) {
            const embedding = await embedText(chunkText);  // OpenAI API call

            // Store with metadata so we can match against qrels later
            points.push({
                id: pointId++,
                vector: embedding,
                payload: {
                    doc_id: docId,           // "2401.01872v2"
                    section_id: sectionId,   // 3
                    chunk_id: chunkIdx,      // 0, 1, 2... (if section was chunked)
                    text: chunkText,
                    title: doc.title,
                }
            });
        }
    }
}

// Step 6: Batch upsert to Qdrant
await qdrant.upsert('ragbench', { points });
```

**Key insight**: We preserve `doc_id` and `section_id` in the metadata. During evaluation, we check if the retrieved chunk matches the ground truth `(doc_id, section_id)` pair from qrels.

---

### 3. evaluate.js

**Purpose**: Test retrieval quality using ground truth relevance judgments.

**How it works**:

```javascript
// Load evaluation data
const queries = loadJson('queries.json');   // { "uuid": { query: "...", type: "..." } }
const qrels = loadJson('qrels.json');       // { "uuid": { doc_id: "...", section_id: N } }
const answers = loadJson('answers.json');   // { "uuid": "answer text" }

// For each query...
for (const queryId of Object.keys(queries)) {
    const queryText = queries[queryId].query;
    const groundTruth = qrels[queryId];  // { doc_id: "2401.01872v2", section_id: 3 }

    // Step 1: Embed the query and search Qdrant
    const queryEmbedding = await embedText(queryText);
    const results = await qdrant.search('ragbench', {
        vector: queryEmbedding,
        limit: 10,  // Get top 10 results
    });

    // Step 2: Check if we found the correct document section
    // results = [
    //   { score: 0.92, payload: { doc_id: "2401.01872v2", section_id: 3, ... } },
    //   { score: 0.87, payload: { doc_id: "2402.00252v3", section_id: 1, ... } },
    //   ...
    // ]

    // Step 3: Calculate metrics
    // MRR: What position is the correct result at?
    for (let i = 0; i < results.length; i++) {
        if (results[i].payload.doc_id === groundTruth.doc_id &&
            results[i].payload.section_id === groundTruth.section_id) {
            mrr = 1 / (i + 1);  // Position 1 → MRR=1.0, Position 2 → MRR=0.5
            break;
        }
    }

    // Recall@K: Is the correct result in the top K?
    recall_at_5 = results.slice(0, 5).some(r => isRelevant(r, groundTruth)) ? 1 : 0;
}

// Aggregate metrics across all queries
const avgMRR = sum(allMRRs) / numQueries;
const avgRecall = sum(allRecalls) / numQueries;
```

**Output**: A JSON file in `runs/` with:
- Aggregate metrics (MRR, nDCG, Recall@K)
- Metrics broken down by query type (abstractive/extractive)
- Metrics broken down by query source (text/text-image/text-table)
- Detailed per-query results

---

### 4. server.js & ui/index.html

**Purpose**: Visualize evaluation results and compare runs.

**API Endpoints**:

```
GET  /api/runs           → List all evaluation runs
GET  /api/runs/:name     → Get detailed results for a run
GET  /api/compare?run1=X&run2=Y → Compare two runs side-by-side
DELETE /api/runs/:name   → Delete a run
```

**Dashboard Features**:
- **Overview**: Latest run metrics at a glance
- **Compare**: Select two runs, see metric deltas (green = improvement, red = regression)
- **Details**: Filter by query type/source, see which queries failed

---

## Evaluation Metrics Explained

### MRR (Mean Reciprocal Rank)

**What it measures**: How early in the ranked list do we find the correct result?

```
Position 1 → RR = 1/1 = 1.00
Position 2 → RR = 1/2 = 0.50
Position 3 → RR = 1/3 = 0.33
Not found  → RR = 0.00

MRR = average of all RR values
```

**Good for**: Understanding if your system tends to rank the correct answer highly.

### nDCG (Normalized Discounted Cumulative Gain)

**What it measures**: Ranking quality with position-based discounting.

```
DCG = Σ (relevance_i / log2(i + 1))

For binary relevance (0 or 1):
- Relevant result at position 1: 1.0 / log2(2) = 1.0
- Relevant result at position 2: 1.0 / log2(3) = 0.63
- Relevant result at position 3: 1.0 / log2(4) = 0.50

nDCG = DCG / IDCG (normalized to 0-1 range)
```

**Good for**: When you care about the full ranking, not just the top result.

### Recall@K

**What it measures**: Is the correct result anywhere in the top K?

```
Recall@1: Is correct result at position 1? (0 or 1)
Recall@5: Is correct result in positions 1-5? (0 or 1)
Recall@10: Is correct result in positions 1-10? (0 or 1)
```

**Good for**: Understanding the "safety net" - even if not rank 1, did we at least retrieve it?

### Precision@K

**What it measures**: What fraction of the top K results are relevant?

```
Precision@5 = (relevant results in top 5) / 5
```

**Note**: In this dataset, each query has exactly 1 relevant document section, so Precision@K will be at most 1/K.

### Document MRR vs Exact MRR

We compute two versions of MRR:
- **Exact MRR**: Must match both `doc_id` AND `section_id`
- **Document MRR**: Only needs to match `doc_id` (more lenient)

Document MRR tells you: "Did I at least find the right paper, even if not the exact section?"

---

## How to Experiment

### Changing Chunking Strategy

Edit `ingest.js`, modify the chunking parameters:

```javascript
const chunks = await semanticChunkWithLimits(section.text, embedText, {
    threshold: 0.5,   // Lower = more aggressive splitting
    maxTokens: 300,   // Smaller chunks
    minTokens: 50,    // Allow smaller minimum
});
```

Then re-run:
```bash
node ingest.js --sample
node evaluate.js --sample --name=smaller-chunks
```

### Changing the Embedding Model

Edit `../embedding.js` to use a different model:

```javascript
const PROVIDERS = {
    openai: {
        model: 'text-embedding-3-small',  // Cheaper, 1536 dims
        dimensions: 1536,
        // ...
    },
};
```

### Changing Retrieval Parameters

Edit `evaluate.js`:

```javascript
const TOP_K = 20;  // Retrieve more results
```

### Comparing Results

After each change:
1. Re-ingest (if chunking/embedding changed)
2. Run evaluation with a descriptive name
3. Use the dashboard to compare runs

```bash
# Baseline
node evaluate.js --sample --name=baseline

# After changing chunk size
node evaluate.js --sample --name=chunks-300

# After changing embedding model
node evaluate.js --sample --name=embed-small

# View comparisons
npm run serve
```

---

## Summary

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| Download | `download_dataset.js` | HuggingFace URL | `data/*.json`, `data/corpus/` |
| Ingest | `ingest.js` | `data/corpus/` | Vectors in Qdrant |
| Evaluate | `evaluate.js` | Queries + Qdrant | `runs/*.json` |
| Visualize | `server.js` | `runs/*.json` | Dashboard UI |

The key to this system is the **qrels** (query relevance judgments) - they provide ground truth so we can objectively measure retrieval quality rather than guessing.

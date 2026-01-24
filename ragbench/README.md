# RAGBench Evaluation

Evaluate your RAG pipeline using the [Vectara Open RAGBench](https://huggingface.co/datasets/vectara/open_ragbench) dataset.

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Download the dataset (creates sample subset automatically)
npm run download

# 3. Ingest documents into Qdrant (use --sample for quick testing)
node ingest.js --sample

# 4. Run evaluation
node evaluate.js --sample --name=baseline

# 5. Start the dashboard
npm run serve
# Open http://localhost:3000
```

## Commands

| Command | Description |
|---------|-------------|
| `npm run download` | Download RAGBench dataset from HuggingFace |
| `npm run ingest` | Ingest all documents into Qdrant |
| `node ingest.js --sample` | Ingest sample subset (100 queries) |
| `npm run evaluate` | Run evaluation on full dataset |
| `node evaluate.js --sample` | Run evaluation on sample subset |
| `node evaluate.js --name=my-run` | Name your evaluation run |
| `npm run serve` | Start the dashboard server |

## Workflow

### 1. Testing Changes to Your RAG System

```bash
# Make changes to your RAG system (chunking, embeddings, etc.)

# Re-ingest with new settings
node ingest.js --sample

# Run a new evaluation with a descriptive name
node evaluate.js --sample --name=larger-chunks

# View results in dashboard
npm run serve
```

### 2. Comparing Runs

The dashboard lets you compare any two runs side-by-side:
- See metric deltas (improvements/regressions)
- Filter by query type (abstractive vs extractive)
- Filter by query source (text, text-image, text-table)

## Metrics

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank - rewards finding the exact relevant section early |
| **Doc MRR** | Document-level MRR - more lenient, just needs to find the right document |
| **nDCG** | Normalized Discounted Cumulative Gain - ranking quality metric |
| **Recall@K** | Whether the relevant section appears in top K results |
| **Precision@K** | Fraction of top K results that are relevant |

## Dataset Structure

```
data/
├── queries.json      # 3,045 questions with type/source metadata
├── answers.json      # Ground truth answers
├── qrels.json        # Query-document relevance mappings
├── corpus/           # Individual arXiv paper documents
│   ├── 2401.01872v2.json
│   └── ...
└── sample/           # 100-query subset for quick testing
    ├── queries.json
    ├── answers.json
    ├── qrels.json
    └── doc_ids.json

runs/                 # Evaluation results
├── index.json        # Run index for dashboard
├── baseline.json
└── larger-chunks.json
```

## Query Types

- **abstractive**: Answers require synthesizing information
- **extractive**: Answers can be found verbatim in the text

## Query Sources

- **text**: Answer from text content only
- **text-image**: Answer requires understanding images
- **text-table**: Answer requires understanding tables
- **text-table-image**: Answer requires multiple modalities

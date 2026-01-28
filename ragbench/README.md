# RAGBench

A benchmarking framework for evaluating Retrieval-Augmented Generation (RAG) pipelines using the [Vectara Open RAGBench](https://huggingface.co/datasets/vectara/open_ragbench) dataset.

Test different embedding models, chunking strategies, and retrieval configurations against 3,045 real-world questions from academic papers.

## Features

- **Standardized Evaluation**: Uses the Vectara Open RAGBench dataset (397 arXiv papers, 3,045 questions)
- **Multiple Metrics**: MRR, nDCG, Recall@K, Precision@K with breakdowns by query type
- **Flexible Embeddings**: Support for OpenAI, Ollama (nomic-embed-text, embeddinggemma, qwen3)
- **Semantic Chunking**: Intelligent text chunking based on sentence similarity
- **Interactive Dashboard**: Compare runs, filter by query type, inspect individual results
- **Sample Mode**: Quick testing with a 100-query subset

## Quick Start

### Prerequisites

- Node.js 18+
- [Qdrant](https://qdrant.tech/) vector database
- [Ollama](https://ollama.ai/) (for local embeddings) or OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ragbench.git
cd ragbench

# Install dependencies
npm install

# Copy environment template and configure
cp .env.example .env

# Start Qdrant (using Docker)
docker run -p 6333:6333 qdrant/qdrant

# Pull an embedding model (if using Ollama)
ollama pull nomic-embed-text
```

### Run Evaluation

```bash
# 1. Download the dataset
npm run download

# 2. Ingest documents (use --sample for quick testing)
node ingest.js --sample

# 3. Run evaluation
node evaluate.js --sample --name=baseline

# 4. View results in dashboard
npm run serve
# Open http://localhost:3000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding model provider | `ollama` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | - |

### Embedding Providers

| Provider | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| `ollama` | nomic-embed-text | 768 | Free, local |
| `embeddinggemma` | embeddinggemma | 768 | Free, local |
| `qwen3` | qwen3-embedding | 4096 | Free, local |
| `openai` | text-embedding-3-large | 3072 | Paid API |

## Commands

| Command | Description |
|---------|-------------|
| `npm run download` | Download RAGBench dataset from HuggingFace |
| `npm run ingest` | Ingest all documents into Qdrant |
| `node ingest.js --sample` | Ingest sample subset (faster) |
| `npm run evaluate` | Run evaluation on full dataset |
| `node evaluate.js --sample --name=my-run` | Run named evaluation on sample |
| `npm run serve` | Start the dashboard server |
| `npm run full-run` | Download + ingest + evaluate (full) |

## Metrics

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank - rewards finding the exact section early |
| **Doc MRR** | Document-level MRR - more lenient, just needs correct document |
| **nDCG** | Normalized Discounted Cumulative Gain - ranking quality |
| **Recall@K** | Whether relevant section appears in top K results |
| **Precision@K** | Fraction of top K results that are relevant |

## Project Structure

```
ragbench/
├── lib/                    # Core modules
│   ├── embedding.js        # Multi-provider embedding support
│   ├── qdrant.js           # Vector database client
│   └── utils.js            # Chunking and similarity functions
├── ui/
│   └── index.html          # Dashboard SPA
├── data/                   # Downloaded dataset (gitignored)
│   ├── queries.json        # 3,045 questions
│   ├── answers.json        # Ground truth answers
│   ├── qrels.json          # Query-document relevance mappings
│   ├── corpus/             # Individual paper documents
│   └── sample/             # 100-query subset
├── runs/                   # Evaluation results
├── download_dataset.js     # Dataset downloader
├── ingest.js               # Document ingestion
├── evaluate.js             # Evaluation runner
├── server.js               # Dashboard server
└── package.json
```

## How It Works

### 1. Dataset

The Vectara Open RAGBench dataset contains:
- **397 arXiv papers** with structured sections
- **3,045 questions** about these papers
- **Ground truth** mapping each question to a specific (document, section) pair

Query types:
- **Abstractive**: Requires synthesizing information
- **Extractive**: Answer can be found verbatim

Query sources:
- **text**: Answer from text only
- **text-image**: Requires understanding figures
- **text-table**: Requires understanding tables

### 2. Ingestion Pipeline

```
Document → Extract Sections → Semantic Chunking → Embed → Store in Qdrant
```

- Extracts abstract and numbered sections from papers
- Uses semantic chunking based on sentence similarity
- Stores chunks with metadata (doc_id, section_id) for evaluation

### 3. Evaluation

```
Query → Embed → Vector Search → Compare to Ground Truth → Calculate Metrics
```

For each query:
1. Generate query embedding
2. Retrieve top-K similar chunks from Qdrant
3. Check if ground truth (doc_id, section_id) appears in results
4. Calculate MRR, nDCG, Recall@K, Precision@K

### 4. Dashboard

The interactive dashboard lets you:
- View aggregate metrics for each run
- Compare two runs side-by-side with deltas
- Filter results by query type/source
- Inspect individual queries with retrieved chunks

## Example Workflow

### Testing a New Embedding Model

```bash
# Set the new provider
export EMBEDDING_PROVIDER=embeddinggemma

# Re-ingest with new embeddings
node ingest.js --sample

# Run evaluation with descriptive name
node evaluate.js --sample --name=embeddinggemma-768

# Compare in dashboard
npm run serve
```

### Tuning Chunking Parameters

Edit `ingest.js` to adjust chunking:

```javascript
const chunks = await semanticChunkWithLimits(text, embedText, {
    threshold: 0.5,    // Lower = more aggressive splitting
    maxTokens: 600,    // Larger chunks
    minTokens: 150,    // Minimum chunk size
});
```

Then re-ingest and evaluate:

```bash
node ingest.js --sample
node evaluate.js --sample --name=larger-chunks
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/runs` | GET | List all evaluation runs |
| `/api/runs/:name` | GET | Get detailed results for a run |
| `/api/runs/:name` | DELETE | Delete a run |
| `/api/compare?run1=X&run2=Y` | GET | Compare two runs |

## Contributing

Contributions welcome! Some ideas:

- Add support for more embedding providers
- Implement additional chunking strategies
- Add more evaluation metrics
- Improve dashboard visualizations

## License

MIT

## Acknowledgments

- [Vectara](https://vectara.com/) for the Open RAGBench dataset
- [Qdrant](https://qdrant.tech/) for the vector database
- [Ollama](https://ollama.ai/) for local LLM/embedding support

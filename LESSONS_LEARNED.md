# RAG Lessons Learned: The Embedding Model Bug

## What Happened

We built a RAG system with excellent semantic chunking, but when we searched for "Avery Lancaster" (the CEO), the system returned random contract signatures and other employees instead.

**The culprit**: The embedding model (`nomic-embed-text`), not the chunking.

---

## The Bug Explained

### Simple Version (ELI5)

Imagine you're looking for your friend "Avery Lancaster" in a crowd.

**Ollama's approach**: It looks for people who *look like* they might be named Avery Lancaster - similar hair, similar clothes, similar vibe. It found other people who looked "professional" and "executive-like" but weren't actually Avery.

**OpenAI's approach**: It actually reads the name tags and understands "Avery Lancaster" is a specific person to find.

The bug was that when you asked "Who is Avery Lancaster?", Ollama returned random contract signatures and other employees because they *felt* similar (they mentioned names, titles, CEOs), but they weren't about Avery at all.

---

### Intermediate Version

Embeddings convert text into numbers (vectors). When you search, it finds vectors that are "close" to your query vector.

**The problem**: `nomic-embed-text` (Ollama) doesn't understand that "Avery Lancaster" is a **named entity** - a specific person to look up. It treats it as generic words and matches based on vague patterns:

```
Query: "Avery Lancaster"
         ↓
Ollama sees: [professional] [person] [name-like]
         ↓
Matches: Contract signatures (also have [name] [title] patterns)
```

**OpenAI's model** was trained on more data and better understands:

```
Query: "Avery Lancaster"
         ↓
OpenAI sees: [specific person named Avery Lancaster]
         ↓
Matches: Documents actually containing "Avery Lancaster"
```

---

### Technical Version

The core issue is **Named Entity Recognition (NER)** quality in embedding models.

Smaller models like `nomic-embed-text` (274MB) encode text into semantic space primarily based on:
- Word frequency patterns
- Syntactic structure
- General topic similarity

They lack the capacity to:
- Recognize proper nouns as distinct entities
- Understand that "Avery Lancaster" should match documents *about* Avery Lancaster
- Distinguish between "a CEO" (general concept) vs "Avery Lancaster, CEO" (specific person)

Larger models (OpenAI's, trained on billions of tokens) develop emergent abilities to:
- Recognize named entities
- Understand coreference ("she" → "Avery Lancaster")
- Weight proper nouns appropriately in similarity calculations

**The scores told the story**:

```
Ollama:
  - Avery Lancaster chunks: 0.56-0.64
  - Random contract chunks:  0.72-0.84  ← Higher! Wrong!

OpenAI:
  - Avery Lancaster chunks: 0.55-0.65
  - Other employee chunks:   0.29       ← Much lower. Correct!
```

---

## But Our Chunking Was Excellent!

Yes, and this is the key lesson: **good chunking is necessary but not sufficient**.

### What Good Chunking Gave Us

Our semantic chunking strategy did everything right:

1. **Coherent chunks**: Each chunk was about one topic (not split mid-sentence)
2. **Appropriate size**: 100-600 tokens per chunk (not too small, not too large)
3. **Semantic boundaries**: Split where topics changed, not at arbitrary positions
4. **Source preservation**: Every chunk knew which file it came from

### Why It Still Failed

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Chunking │ →  │Embedding │ →  │ Storage  │ →  │ Retrieval│  │
│  │    ✓     │    │    ✗     │    │    ✓     │    │    ✓     │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
│  Perfect chunks   Bad model      Qdrant works   Returns what    │
│  created          ruined them    fine           embedding found │
└─────────────────────────────────────────────────────────────────┘
```

The embedding model is the **lens** through which your chunks are viewed. A bad lens makes even perfect chunks look wrong.

### Analogy

Think of it like a library:

- **Chunking** = How you organize books into sections (chapters, topics)
- **Embedding** = The librarian who understands what you're asking for
- **Storage** = The shelves holding the books
- **Retrieval** = Finding and returning books

You can have perfectly organized books, but if the librarian doesn't understand that "Avery Lancaster" is a person's name (not a genre or topic), they'll bring you the wrong books.

---

## How This Scales to Production

### The Scary Part

In production, this bug is **silent**. The system returns results, users see answers, but:

1. **Wrong context goes to the LLM** → LLM generates confident but wrong answers
2. **No errors thrown** → Monitoring sees "healthy" system
3. **Users lose trust** → "This AI doesn't know anything about our company"

### Real Production Failures This Causes

| Scenario | What Happens |
|----------|--------------|
| HR chatbot | "Who is our CEO?" → Returns random employee |
| Legal search | "Find the Acme contract" → Returns wrong contract |
| Support bot | "What's the refund policy?" → Returns shipping policy |

---

## How to Spot This in Production

### 1. Build an Evaluation Dataset

Create a test set with known correct answers:

```javascript
const evalSet = [
  { query: "Who is Avery Lancaster?", expectedSource: "employees/Avery Lancaster.md" },
  { query: "What is Carllm?", expectedSource: "products/Carllm.md" },
  { query: "DriveSmart contract details", expectedSource: "contracts/Contract with DriveSmart..." },
];

// Run after any embedding/chunking change
for (const test of evalSet) {
  const results = await search(test.query);
  const topSource = results[0].payload.source;
  if (topSource !== test.expectedSource) {
    console.error(`FAILED: "${test.query}" returned ${topSource}, expected ${test.expectedSource}`);
  }
}
```

### 2. Monitor Retrieval Quality Metrics

```javascript
// Log these in production
{
  query: "Avery Lancaster",
  topScore: 0.65,
  scoreGap: 0.36,        // Gap between #1 and #2 result (higher = more confident)
  topSourceRelevant: true // Human feedback or heuristic
}
```

**Red flags**:
- Low `topScore` (< 0.5)
- Small `scoreGap` (results are all similarly scored = model is guessing)
- High query volume with low scores

### 3. A/B Test Embedding Models

Before switching models in production:

```javascript
// Shadow mode: run both, compare
const ollamaResults = await searchWithOllama(query);
const openaiResults = await searchWithOpenAI(query);

log({
  query,
  ollamaTop: ollamaResults[0].payload.source,
  openaiTop: openaiResults[0].payload.source,
  agreement: ollamaResults[0].payload.source === openaiResults[0].payload.source
});
```

If agreement is low, investigate before switching.

### 4. Use Hybrid Search for Critical Systems

Combine semantic search with keyword (BM25) search:

```
Final Score = (0.7 × semantic_score) + (0.3 × keyword_score)
```

This way, even if semantic search fails on "Avery Lancaster", keyword search catches it.

---

## Key Takeaways

| Lesson | Action |
|--------|--------|
| Good chunking ≠ good retrieval | Test the full pipeline, not just chunking |
| Smaller models fail on names | Test with named entity queries before choosing a model |
| Silent failures are dangerous | Build evaluation datasets, monitor retrieval scores |
| Don't trust vibes | Measure with numbers (scores, gaps, precision@k) |
| Hybrid search is safer | Combine semantic + keyword for production systems |

---

## The Fix We Applied

We made the embedding provider configurable:

```bash
# Use OpenAI (default) - better quality, costs money
node setup_vector_db.js

# Use Ollama - free/local, lower quality for names
EMBEDDING_PROVIDER=ollama node setup_vector_db.js
```

See `embedding.js` for the implementation.

---

## Questions to Ask Before Choosing an Embedding Model

1. Does my data contain many proper nouns (names, product names, company names)?
2. Will users search by specific entity names?
3. What's the score distribution for my test queries?
4. Can I afford the latency/cost of a larger model?
5. Should I use hybrid search as insurance?

If you answer "yes" to #1 and #2, invest in a better embedding model or hybrid search.

# Why My RAG System Couldn't Find the CEO

*A story about building a "perfect" system that still failed spectacularly*

---

## The Setup

I was building a RAG (Retrieval-Augmented Generation) system for a company knowledge base. The goal was simple: let employees ask questions like "Who is our CEO?" and get accurate answers.

I did everything "right":
- Semantic chunking that split documents at topic boundaries
- Vector database (Qdrant) for fast similarity search
- Local embedding model (Ollama) to save money

I ran my first test query: **"Who is Avery Lancaster?"** (the CEO)

The results?

```
0.71 - contracts/Contract with Pinnacle Insurance Co.md
   **Insurellm Authorized Signature** Name: Sarah Johnson Title: VP of Sales...

0.69 - contracts/Contract with GreenField Holdings.md
   **Signatures:** [Name], Title...

0.63 - employees/Jordan K. Bishop.md
   Jordan K. Bishop is a valued member of the Insurellm family...
```

**The CEO wasn't even in the top 10 results.**

Instead, I got contract signature blocks and random employees. The system was confidently returning the wrong information.

---

## The Investigation

My first instinct: *"The chunking must be wrong."*

I checked. The chunks were perfect:

```
--- Chunk 3 from Avery Lancaster.md ---
# Avery Lancaster
## Summary
- **Job Title**: Co-Founder & Chief Executive Officer (CEO)
- **Location**: San Francisco, California
...
```

Everything about Avery was there, neatly organized. So why wasn't it being found?

I dug deeper and compared the similarity scores:

| Source | Score |
|--------|-------|
| Random contract signatures | 0.72 - 0.84 |
| **Avery Lancaster's actual profile** | **0.56 - 0.64** |

The contract signatures scored **higher** than the actual answer. The embedding model thought contracts were more relevant to "Avery Lancaster" than... Avery Lancaster.

---

## The "Aha" Moment

The problem wasn't my chunking. It was the **embedding model**.

Here's what was happening inside `nomic-embed-text` (the Ollama model):

```
Query: "Who is Avery Lancaster?"
              ↓
Model thinks: [question] [person] [name] [professional]
              ↓
Best matches: Anything with names, titles, signatures
              ↓
Returns: Contract signature blocks (they have names + titles!)
```

The model didn't understand that "Avery Lancaster" was a **specific person** to look up. It treated it as generic words and matched based on vibes.

Contract signatures matched because they contained:
- Names ("Sarah Johnson", "Tom Anderson")
- Titles ("VP of Sales", "CEO")
- Professional context

The model saw patterns, not meaning.

---

## The Fix

I switched to OpenAI's embedding model (`text-embedding-3-small`) and ran the same query:

```
0.65 - employees/Avery Lancaster.md
   Avery Lancaster has demonstrated resilience and adaptability...

0.64 - employees/Avery Lancaster.md
   # Avery Lancaster ## Summary - **Job Title**: CEO...

0.63 - employees/Avery Lancaster.md
   - **2013 - 2015**: Senior Product Manager...

... (all top 8 results are Avery Lancaster)

0.29 - employees/Amanda Foster.md
   (finally, a different person, with much lower score)
```

**All top 8 results were from Avery Lancaster's file.** The 9th result scored 0.29 - a massive drop, indicating high confidence in the top results.

---

## Why Did This Happen?

### Small Models vs Large Models

| Model | Size | Named Entity Understanding |
|-------|------|---------------------------|
| `nomic-embed-text` | 274MB | Poor - treats names as generic words |
| OpenAI `text-embedding-3-small` | ~100M params | Good - recognizes names as entities |

Smaller models are trained on less data and develop fewer "emergent abilities." Understanding that "Avery Lancaster" is a specific person (not just two words) requires training on billions of examples of how names work in context.

### The Chunking Red Herring

My chunking was excellent. But here's the thing:

```
Perfect Chunks + Bad Embedding Model = Bad Results
```

It's like having a perfectly organized library but hiring a librarian who can't read names. They'll bring you books that "feel" right but aren't what you asked for.

---

## The Scary Part: Silent Failures

This bug is **silent**. The system:
- Returns results (looks healthy)
- Doesn't throw errors (monitoring is green)
- Gives confident answers (user trusts it)

But the answers are wrong.

In production, this means:
- HR chatbot says the wrong person is CEO
- Legal search returns the wrong contract
- Support bot gives incorrect policy information

And nobody knows until a human catches it.

---

## How to Prevent This

### 1. Build an Evaluation Dataset

Before going to production, create test cases:

```javascript
const tests = [
  { query: "Who is Avery Lancaster?", expectedSource: "employees/Avery Lancaster.md" },
  { query: "What is Carllm?", expectedSource: "products/Carllm.md" },
  { query: "DriveSmart contract", expectedSource: "contracts/...DriveSmart..." },
];

// Run after any change
for (const test of tests) {
  const results = await search(test.query);
  if (!results[0].payload.source.includes(test.expectedSource)) {
    throw new Error(`FAILED: ${test.query}`);
  }
}
```

### 2. Monitor Score Gaps

```javascript
const scores = results.map(r => r.score);
const gap = scores[0] - scores[1];

// If gap is small, the model is guessing
if (gap < 0.1) {
  log.warn("Low confidence retrieval", { query, gap });
}
```

### 3. Test With Named Entities

Before choosing an embedding model, test queries like:
- "Who is [Person Name]?"
- "What is [Product Name]?"
- "Find the [Company Name] contract"

If these fail, the model isn't production-ready for your use case.

### 4. Consider Hybrid Search

Combine semantic search with keyword (BM25) search:

```
finalScore = 0.7 * semanticScore + 0.3 * keywordScore
```

Even if semantic search fails on "Avery Lancaster," keyword search will catch the exact match.

---

## The Code

I made my embedding provider configurable so I can easily switch:

```javascript
// embedding.js
const PROVIDERS = {
  openai: {
    model: 'text-embedding-3-small',
    dimensions: 1536,
  },
  ollama: {
    model: 'nomic-embed-text',
    dimensions: 768,
  },
};

const PROVIDER = process.env.EMBEDDING_PROVIDER || 'openai';
```

To switch providers:

```bash
# Use OpenAI (better quality, costs money)
node setup_vector_db.js

# Use Ollama (free, lower quality for names)
EMBEDDING_PROVIDER=ollama node setup_vector_db.js
```

---

## Key Takeaways

1. **Good chunking ≠ good retrieval.** You need both good chunking AND a good embedding model.

2. **Smaller models fail on names.** If your data has lots of proper nouns (people, products, companies), test carefully.

3. **Silent failures are the worst kind.** Build evaluation datasets and monitor retrieval quality.

4. **Measure, don't assume.** Look at actual scores and score gaps, not just "it returned something."

5. **Hybrid search is insurance.** Combining semantic + keyword search catches edge cases.

---

## The Irony

I spent hours perfecting my chunking strategy - semantic boundaries, token limits, similarity thresholds. It was elegant.

Then a 274MB model that couldn't understand names broke everything.

The lesson? In ML systems, the "boring" infrastructure decisions (which embedding model to use) often matter more than the clever algorithms (semantic chunking).

Test your whole pipeline, not just the parts you're proud of.

---

*Built while learning RAG systems. The best bugs are the ones that teach you something.*

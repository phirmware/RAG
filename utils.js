import nlp from 'compromise';
import { encode } from 'gpt-tokenizer';

// Re-export embedText from the embedding module for convenience
export { embedText, VECTOR_DIMENSIONS, EMBEDDING_PROVIDER } from './embedding.js';

// read more on this cosine similarity
export function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function splitIntoSentences(text) {
    return nlp(text).sentences().out('array');
}

/**
 * Splits text into semantically meaningful chunks based on sentence similarity and token limits.
 *
 * This function uses a "semantic chunking" approach: instead of splitting text at arbitrary
 * positions, it groups sentences that are semantically related (similar meaning/topic) together,
 * while respecting token size constraints.
 *
 * @param {string} text - The input text to chunk
 * @param {Function} embedFn - Async function that converts text to a vector embedding
 * @param {Object} options - Configuration options
 * @param {number} options.threshold - Similarity threshold (0-1). If consecutive sentences have
 *                                     similarity below this, they may be split into different chunks.
 *                                     Lower = more aggressive splitting. Default: 0.65
 * @param {number} options.maxTokens - Maximum tokens allowed per chunk. Default: 500
 * @param {number} options.minTokens - Minimum tokens required before allowing a split. Default: 100
 * @returns {Promise<string[]>} Array of text chunks
 */
export async function semanticChunkWithLimits(text, embedFn, {
  threshold = 0.65,
  maxTokens = 500,
  minTokens = 100,
}) {
  // Step 1: Break the text into individual sentences
  const sentences = splitIntoSentences(text);

  // Step 2: Convert each sentence into a vector embedding
  // Embeddings are numerical representations that capture semantic meaning,
  // allowing us to mathematically compare how similar two sentences are
  const sentenceEmbeddings = await Promise.all(
    sentences.map(s => embedFn(s))
  );

  const chunks = [];           // Final output: array of text chunks
  let currentSentences = [];   // Sentences accumulated for the current chunk
  let currentTokens = 0;       // Token count for the current chunk

  // Step 3: Iterate through sentences, deciding where to split into chunks
  for (let i = 0; i < sentences.length; i++) {
    const sent = sentences[i];
    const sentEmbedding = sentenceEmbeddings[i];
    const sentTokens = encode(sent).length;  // Count tokens in this sentence

    // First sentence always starts a new chunk
    if (!currentSentences.length) {
      currentSentences.push(sent);
      currentTokens += sentTokens;
      continue;
    }

    // Compare this sentence's embedding with the previous sentence's embedding
    // High similarity (close to 1) = sentences are about the same topic
    // Low similarity (close to 0) = topic has changed
    const prevEmbedding = sentenceEmbeddings[i - 1];
    const similarity = cosineSimilarity(prevEmbedding, sentEmbedding);

    // Decide whether to split here based on two conditions:
    // 1. Semantic shift: similarity dropped below threshold (topic changed)
    // 2. Size limit: adding this sentence would exceed maxTokens
    const shouldSplit =
      similarity < threshold ||
      currentTokens + sentTokens > maxTokens;

    // Only actually split if we've accumulated enough tokens (minTokens).
    // This prevents creating tiny chunks when there are many small topic shifts.
    if (shouldSplit && currentTokens >= minTokens) {
      // Finalize the current chunk and start a new one with this sentence
      chunks.push(currentSentences.join(' '));
      currentSentences = [sent];
      currentTokens = sentTokens;
    } else {
      // Keep adding to the current chunk
      currentSentences.push(sent);
      currentTokens += sentTokens;
    }
  }

  // Don't forget the last chunk (sentences that weren't finalized in the loop)
  if (currentSentences.length) {
    chunks.push(currentSentences.join(' '));
  }

  return chunks;
}


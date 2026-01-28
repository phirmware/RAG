/**
 * Utility functions for RAG operations
 */

import nlp from 'compromise';
import { encode } from 'gpt-tokenizer';

export { embedText, VECTOR_DIMENSIONS, EMBEDDING_PROVIDER } from './embedding.js';

/**
 * Calculate cosine similarity between two vectors
 */
export function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Split text into sentences using NLP
 */
export function splitIntoSentences(text) {
    return nlp(text).sentences().out('array');
}

/**
 * Semantic chunking with token limits
 *
 * Groups semantically similar sentences together while respecting token constraints.
 * Uses embedding similarity to detect topic shifts.
 *
 * @param {string} text - The input text to chunk
 * @param {Function} embedFn - Async function that converts text to embeddings
 * @param {Object} options - Configuration options
 * @param {number} options.threshold - Similarity threshold (0-1) for splitting. Default: 0.65
 * @param {number} options.maxTokens - Maximum tokens per chunk. Default: 500
 * @param {number} options.minTokens - Minimum tokens before allowing a split. Default: 100
 * @returns {Promise<string[]>} Array of text chunks
 */
export async function semanticChunkWithLimits(text, embedFn, {
    threshold = 0.65,
    maxTokens = 500,
    minTokens = 100,
} = {}) {
    const sentences = splitIntoSentences(text);
    const sentenceEmbeddings = await Promise.all(sentences.map(s => embedFn(s)));

    const chunks = [];
    let currentSentences = [];
    let currentTokens = 0;

    for (let i = 0; i < sentences.length; i++) {
        const sent = sentences[i];
        const sentEmbedding = sentenceEmbeddings[i];
        const sentTokens = encode(sent).length;

        if (!currentSentences.length) {
            currentSentences.push(sent);
            currentTokens += sentTokens;
            continue;
        }

        const prevEmbedding = sentenceEmbeddings[i - 1];
        const similarity = cosineSimilarity(prevEmbedding, sentEmbedding);

        const shouldSplit =
            similarity < threshold ||
            currentTokens + sentTokens > maxTokens;

        if (shouldSplit && currentTokens >= minTokens) {
            chunks.push(currentSentences.join(' '));
            currentSentences = [sent];
            currentTokens = sentTokens;
        } else {
            currentSentences.push(sent);
            currentTokens += sentTokens;
        }
    }

    if (currentSentences.length) {
        chunks.push(currentSentences.join(' '));
    }

    return chunks;
}

/**
 * Embedding Provider Configuration
 *
 * Switch between OpenAI and Ollama by setting the EMBEDDING_PROVIDER env variable:
 *   - EMBEDDING_PROVIDER=openai  (default) - Uses OpenAI's text-embedding-3-small (1536 dims, paid)
 *   - EMBEDDING_PROVIDER=ollama            - Uses Ollama's nomic-embed-text (768 dims, free/local)
 */

import dotenv from 'dotenv';
dotenv.config();

import OpenAI from 'openai';

// Determine which provider to use from environment variable
const PROVIDER = process.env.EMBEDDING_PROVIDER || 'openai';

// text-embedding-3-large - 3072 dimensions
// text-embedding-3-small - 1536 dimensions

// Provider configurations
const PROVIDERS = {
  openai: {
    model: 'text-embedding-3-large',
    dimensions: 3072,
    client: () => new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  },
  ollama: {
    model: 'nomic-embed-text',
    dimensions: 768,
    client: () => new OpenAI({
      baseURL: 'http://localhost:11434/v1',
      apiKey: 'ollama',  // Required by library but ignored by Ollama
    }),
  },
  qwen3: {
    model: 'qwen3-embedding:latest',
    dimensions: 4096,
    client: () => new OpenAI({
      baseURL: 'http://localhost:11434/v1',
      apiKey: 'ollama',  // Required by library but ignored by Ollama
    }),
  },
  embeddinggemma: {
    model: 'embeddinggemma:latest',
    dimensions: 768,
    client: () => new OpenAI({
      baseURL: 'http://localhost:11434/v1',
      apiKey: 'ollama',  // Required by library but ignored by Ollama
    }),
  }
};

// Validate provider
if (!PROVIDERS[PROVIDER]) {
  throw new Error(`Unknown EMBEDDING_PROVIDER: ${PROVIDER}. Use 'openai' or 'ollama'.`);
}

const config = PROVIDERS[PROVIDER];
const client = config.client();

console.log(`[Embedding] Using ${PROVIDER} with model ${config.model} (${config.dimensions} dimensions)`);

/**
 * Generate an embedding vector for text
 *
 * @param {string} text - The text to embed
 * @returns {Promise<number[]>} The embedding vector
 */
export async function embedText(text) {
  const res = await client.embeddings.create({
    model: config.model,
    input: text,
  });
  return res.data[0].embedding;
}

/**
 * Get the vector dimensions for the current provider
 * Use this when creating Qdrant collections
 */
export const VECTOR_DIMENSIONS = config.dimensions;

/**
 * Get the current provider name
 */
export const EMBEDDING_PROVIDER = PROVIDER;

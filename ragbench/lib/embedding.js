/**
 * Embedding Provider Configuration
 *
 * Supports multiple embedding providers through the EMBEDDING_PROVIDER env variable:
 *   - openai: OpenAI's text-embedding-3-large (3072 dims, paid API)
 *   - ollama: nomic-embed-text via Ollama (768 dims, free/local)
 *   - embeddinggemma: embeddinggemma via Ollama (768 dims, free/local)
 *   - qwen3: qwen3-embedding via Ollama (4096 dims, free/local)
 */

import OpenAI from 'openai';

const PROVIDER = process.env.EMBEDDING_PROVIDER || 'ollama';

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
            baseURL: process.env.OLLAMA_URL || 'http://localhost:11434/v1',
            apiKey: 'ollama',
        }),
    },
    embeddinggemma: {
        model: 'embeddinggemma:latest',
        dimensions: 768,
        client: () => new OpenAI({
            baseURL: process.env.OLLAMA_URL || 'http://localhost:11434/v1',
            apiKey: 'ollama',
        }),
    },
    qwen3: {
        model: 'qwen3-embedding:latest',
        dimensions: 4096,
        client: () => new OpenAI({
            baseURL: process.env.OLLAMA_URL || 'http://localhost:11434/v1',
            apiKey: 'ollama',
        }),
    },
};

if (!PROVIDERS[PROVIDER]) {
    const available = Object.keys(PROVIDERS).join(', ');
    throw new Error(`Unknown EMBEDDING_PROVIDER: ${PROVIDER}. Available: ${available}`);
}

const config = PROVIDERS[PROVIDER];
const client = config.client();

console.log(`[Embedding] Provider: ${PROVIDER} | Model: ${config.model} | Dimensions: ${config.dimensions}`);

/**
 * Generate an embedding vector for text
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

export const VECTOR_DIMENSIONS = config.dimensions;
export const EMBEDDING_PROVIDER = PROVIDER;

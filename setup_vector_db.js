/**
 * Vector Database Setup Script
 *
 * This script reads all files from the knowledge-base folder (including nested subfolders),
 * splits them into semantically meaningful chunks, generates embeddings for each chunk,
 * and stores them in a Qdrant vector database for later retrieval.
 *
 * Set EMBEDDING_PROVIDER env variable to switch between providers:
 *   - EMBEDDING_PROVIDER=openai  (default) - OpenAI text-embedding-3-small
 *   - EMBEDDING_PROVIDER=ollama            - Ollama nomic-embed-text (free/local)
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';  // Used to generate UUIDs for Qdrant point IDs
import dotenv from 'dotenv';

dotenv.config();

import qdrant from './qdrant.js';
import { semanticChunkWithLimits, embedText, VECTOR_DIMENSIONS } from './utils.js';

const FOLDER = './knowledge-base';
const COLLECTION_NAME = 'my_rag_collection';

// --- Collection Setup ---
// Check if collection exists, delete and recreate for a fresh start

const collections = await qdrant.getCollections();
const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

console.log('Existing collections:', collections.collections.map(c => c.name));

if (exists) {
  console.log(`Collection "${COLLECTION_NAME}" exists, deleting...`);
  await qdrant.deleteCollection(COLLECTION_NAME);
  console.log(`Collection "${COLLECTION_NAME}" deleted.`);
}

console.log(`Creating collection "${COLLECTION_NAME}" with ${VECTOR_DIMENSIONS} dimensions...`);
await qdrant.createCollection(COLLECTION_NAME, {
  vectors: {
    size: VECTOR_DIMENSIONS,  // Automatically set based on EMBEDDING_PROVIDER
    distance: 'Cosine',
  },
});
console.log(`Collection "${COLLECTION_NAME}" created successfully.\n`);

/**
 * Recursively walks through a directory and collects all files.
 *
 * @param {string} dir - The current directory being scanned
 * @param {string} baseDir - The root directory (used to calculate relative paths)
 * @returns {Array<{fullPath: string, relativePath: string}>} Array of file info objects
 */
function getAllFiles(dir, baseDir = dir) {
  // withFileTypes: true gives us Dirent objects which have isDirectory() and isFile() methods
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      // Recurse into subdirectory, spreading results into our array
      files.push(...getAllFiles(fullPath, baseDir));
    } else if (entry.isFile()) {
      // Calculate path relative to knowledge-base folder (e.g., "employees/Alex Chen.md")
      // This is stored as the "source" in Qdrant for later reference
      const relativePath = path.relative(baseDir, fullPath);
      files.push({ fullPath, relativePath });
    }
  }

  return files;
}

// Discover all files in the knowledge base
const files = getAllFiles(FOLDER);
console.log(`Found ${files.length} files:`, files.map(f => f.relativePath));

// Process each file
for (const { fullPath, relativePath } of files) {
  console.log(`\nProcessing: ${relativePath}`);

  // Read the entire file content
  const content = fs.readFileSync(fullPath, 'utf8');

  // Split the content into semantic chunks
  // This groups related sentences together based on embedding similarity
  // This calls embedText internally for each sentence (free with Ollama!)
  const chunks = await semanticChunkWithLimits(content, embedText, {
    threshold: 0.65,   // Similarity threshold - lower = more splits
    maxTokens: 600,    // Maximum tokens per chunk
    minTokens: 100,    // Minimum tokens before allowing a split
  });

  console.log(`  Created ${chunks.length} chunks`);

  // Extract category from folder path (e.g., "employees/Avery Lancaster.md" -> "employees")
  // This is used for coloring points in Qdrant's visualization UI
  const category = relativePath.split('/')[0];

  // Upload each chunk to Qdrant
  for (const textChunk of chunks) {
    // Prepend source info to the text for embedding
    // This helps with name-based queries (e.g., "Avery Lancaster" will appear in all her chunks)
    // Extract a readable title from the file path (e.g., "employees/Avery Lancaster.md" -> "Avery Lancaster")
    const fileName = path.basename(relativePath, path.extname(relativePath));
    const textForEmbedding = `[${fileName}]\n\n${textChunk}`;

    // Generate embedding vector for this chunk (free with Ollama!)
    const vector = await embedText(textForEmbedding);

    // Upsert the point into Qdrant
    // First arg is collection name, second is the options object with points array
    await qdrant.upsert(COLLECTION_NAME, {
      points: [
        {
          // Qdrant requires IDs to be either unsigned integers or UUIDs
          // crypto.randomUUID() generates a valid UUID v4
          id: crypto.randomUUID(),
          // The embedding vector (dimensions depend on EMBEDDING_PROVIDER)
          vector,
          // Payload stores metadata we can retrieve later
          payload: {
            text: textChunk,       // The original text content (without the prefix)
            source: relativePath,  // Which file this came from
            category,              // Folder name for visualization (employees, contracts, products, company)
          },
        },
      ],
    });
  }

  console.log(`  Uploaded to Qdrant`);
}

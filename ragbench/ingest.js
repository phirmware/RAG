/**
 * Ingest RAGBench corpus documents into Qdrant vector database
 *
 * Processes corpus documents, chunks sections, embeds them, and stores
 * in Qdrant with metadata for evaluation (doc_id, section_id).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

import qdrant from './lib/qdrant.js';
import { embedText, VECTOR_DIMENSIONS, semanticChunkWithLimits } from './lib/utils.js';

const DATA_DIR = path.join(__dirname, 'data');
const CORPUS_DIR = path.join(DATA_DIR, 'corpus');

// Configuration
const COLLECTION_NAME = 'ragbench';
const BATCH_SIZE = 50; // Points per upsert batch
const USE_SAMPLE = process.argv.includes('--sample'); // Use sample subset for quick testing

/**
 * Create or recreate the Qdrant collection
 */
async function setupCollection() {
    console.log(`Setting up collection: ${COLLECTION_NAME}`);

    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

    if (exists) {
        console.log('  Collection exists, deleting...');
        await qdrant.deleteCollection(COLLECTION_NAME);
    }

    // Create collection
    await qdrant.createCollection(COLLECTION_NAME, {
        vectors: {
            size: VECTOR_DIMENSIONS,
            distance: 'Cosine',
        },
    });

    console.log(`  Created collection with ${VECTOR_DIMENSIONS} dimensions`);
}

/**
 * Load list of document IDs to ingest
 */
function loadDocumentIds() {
    if (USE_SAMPLE) {
        const samplePath = path.join(DATA_DIR, 'sample', 'doc_ids.json');
        if (!fs.existsSync(samplePath)) {
            throw new Error('Sample subset not found. Run download_dataset.js first.');
        }
        return JSON.parse(fs.readFileSync(samplePath, 'utf-8'));
    }

    // Load all qrels to get required doc IDs
    const qrelsPath = path.join(DATA_DIR, 'qrels.json');
    if (!fs.existsSync(qrelsPath)) {
        throw new Error('qrels.json not found. Run download_dataset.js first.');
    }

    const qrels = JSON.parse(fs.readFileSync(qrelsPath, 'utf-8'));
    const docIds = new Set();
    for (const queryId in qrels) {
        if (qrels[queryId]?.doc_id) {
            docIds.add(qrels[queryId].doc_id);
        }
    }
    return Array.from(docIds);
}

/**
 * Load a corpus document
 */
function loadDocument(docId) {
    const filepath = path.join(CORPUS_DIR, `${docId}.json`);
    if (!fs.existsSync(filepath)) {
        return null;
    }
    return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
}

/**
 * Extract sections from a document
 * Returns array of { sectionId, text, metadata }
 */
function extractSections(doc) {
    const sections = [];

    // Add abstract as section -1 if present
    if (doc.abstract) {
        sections.push({
            sectionId: -1,
            text: doc.abstract,
            metadata: {
                type: 'abstract',
                title: doc.title || '',
            },
        });
    }

    // Add each section
    if (doc.sections && Array.isArray(doc.sections)) {
        doc.sections.forEach((section, idx) => {
            if (section.text && section.text.trim()) {
                // Clean the text (remove image/table placeholders if desired)
                let text = section.text;

                // Include table content if available
                if (section.tables) {
                    for (const [tableId, tableContent] of Object.entries(section.tables)) {
                        text += `\n\nTable ${tableId}:\n${tableContent}`;
                    }
                }

                sections.push({
                    sectionId: idx,
                    text: text.trim(),
                    metadata: {
                        type: 'section',
                        title: doc.title || '',
                    },
                });
            }
        });
    }

    return sections;
}

/**
 * Chunk a section using semantic chunking
 * For shorter sections, keeps them whole
 */
async function chunkSection(text, maxChunkSize = 500) {
    // If text is short enough, don't chunk
    if (text.length < maxChunkSize * 2) {
        return [text];
    }

    try {
        const chunks = await semanticChunkWithLimits(text, embedText, {
            threshold: 0.6,
            maxTokens: 400,
            minTokens: 100,
        });
        return chunks;
    } catch (error) {
        // Fallback: return whole section
        console.warn(`  Warning: chunking failed, using whole section`);
        return [text];
    }
}

/**
 * Process a single document and return points for Qdrant
 */
async function processDocument(docId, doc, startingPointId) {
    const sections = extractSections(doc);
    const points = [];
    let pointId = startingPointId;

    for (const section of sections) {
        // Chunk the section
        const chunks = await chunkSection(section.text);

        for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
            const chunkText = chunks[chunkIdx];

            // Generate embedding
            const embedding = await embedText(chunkText);

            points.push({
                id: pointId++,
                vector: embedding,
                payload: {
                    doc_id: docId,
                    section_id: section.sectionId,
                    chunk_id: chunkIdx,
                    text: chunkText,
                    title: section.metadata.title,
                    type: section.metadata.type,
                },
            });
        }
    }

    return { points, nextPointId: pointId };
}

/**
 * Upsert points to Qdrant in batches
 */
async function upsertBatch(points) {
    if (points.length === 0) return;

    await qdrant.upsert(COLLECTION_NAME, {
        wait: true,
        points: points,
    });
}

/**
 * Main ingestion function
 */
async function main() {
    console.log('='.repeat(60));
    console.log('RAGBench Corpus Ingestion');
    console.log('='.repeat(60));

    if (USE_SAMPLE) {
        console.log('Mode: SAMPLE (100 queries subset)');
    } else {
        console.log('Mode: FULL DATASET');
    }

    // Setup collection
    await setupCollection();

    // Load document IDs
    const docIds = loadDocumentIds();
    console.log(`\nDocuments to process: ${docIds.length}`);

    // Process documents
    let totalPoints = 0;
    let pointId = 0;
    let pendingPoints = [];
    let processedDocs = 0;
    let failedDocs = 0;

    console.log('\nProcessing documents...');

    for (const docId of docIds) {
        const doc = loadDocument(docId);

        if (!doc) {
            failedDocs++;
            continue;
        }

        try {
            const { points, nextPointId } = await processDocument(docId, doc, pointId);
            pointId = nextPointId;
            pendingPoints.push(...points);
            processedDocs++;

            // Batch upsert when we have enough points
            if (pendingPoints.length >= BATCH_SIZE) {
                await upsertBatch(pendingPoints);
                totalPoints += pendingPoints.length;
                pendingPoints = [];
            }

            // Progress update
            if (processedDocs % 10 === 0) {
                process.stdout.write(`\r  Processed: ${processedDocs}/${docIds.length} docs, ${totalPoints + pendingPoints.length} chunks`);
            }
        } catch (error) {
            console.error(`\n  Error processing ${docId}: ${error.message}`);
            failedDocs++;
        }
    }

    // Final batch
    if (pendingPoints.length > 0) {
        await upsertBatch(pendingPoints);
        totalPoints += pendingPoints.length;
    }

    console.log(); // New line after progress

    // Verify collection
    const collectionInfo = await qdrant.getCollection(COLLECTION_NAME);

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('Ingestion Complete!');
    console.log('='.repeat(60));
    console.log(`  Documents processed: ${processedDocs}`);
    console.log(`  Documents failed: ${failedDocs}`);
    console.log(`  Total chunks indexed: ${totalPoints}`);
    console.log(`  Collection: ${COLLECTION_NAME}`);
    console.log(`  Vectors in collection: ${collectionInfo.points_count}`);

    // Save ingestion metadata
    const metadata = {
        timestamp: new Date().toISOString(),
        mode: USE_SAMPLE ? 'sample' : 'full',
        documentsProcessed: processedDocs,
        documentsFailed: failedDocs,
        totalChunks: totalPoints,
        collectionName: COLLECTION_NAME,
        vectorDimensions: VECTOR_DIMENSIONS,
    };

    fs.writeFileSync(
        path.join(DATA_DIR, 'ingestion_metadata.json'),
        JSON.stringify(metadata, null, 2)
    );
    console.log(`\n  Metadata saved to: ${path.join(DATA_DIR, 'ingestion_metadata.json')}`);
}

main().catch(console.error);

/**
 * Download the Vectara Open RAGBench dataset from HuggingFace
 *
 * Downloads:
 * - queries.json: Question definitions with metadata
 * - answers.json: Ground truth answers
 * - qrels.json: Query-document relevance mappings
 * - corpus/*.json: Individual paper documents
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = 'https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv';
const DATA_DIR = path.join(__dirname, 'data');
const CORPUS_DIR = path.join(DATA_DIR, 'corpus');

// Main metadata files to download
const METADATA_FILES = ['queries.json', 'answers.json', 'qrels.json', 'pdf_urls.json'];

/**
 * Fetch JSON from a URL with retries
 */
async function fetchJson(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            if (i === retries - 1) throw error;
            console.log(`  Retry ${i + 1}/${retries} for ${url}`);
            await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        }
    }
}

/**
 * Download a file and save to disk
 */
async function downloadFile(url, filepath) {
    const data = await fetchJson(url);
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
    return data;
}

/**
 * Get list of unique document IDs from qrels
 */
function getRequiredDocIds(qrels) {
    const docIds = new Set();
    for (const queryId in qrels) {
        const rel = qrels[queryId];
        if (rel.doc_id) {
            docIds.add(rel.doc_id);
        }
    }
    return Array.from(docIds);
}

/**
 * Download corpus files with progress tracking
 */
async function downloadCorpusFiles(docIds, concurrency = 5) {
    console.log(`\nDownloading ${docIds.length} corpus documents...`);

    let completed = 0;
    let failed = [];

    // Process in batches for controlled concurrency
    for (let i = 0; i < docIds.length; i += concurrency) {
        const batch = docIds.slice(i, i + concurrency);

        await Promise.all(batch.map(async (docId) => {
            const filename = `${docId}.json`;
            const filepath = path.join(CORPUS_DIR, filename);

            // Skip if already downloaded
            if (fs.existsSync(filepath)) {
                completed++;
                return;
            }

            try {
                const url = `${BASE_URL}/corpus/${filename}`;
                await downloadFile(url, filepath);
                completed++;
            } catch (error) {
                failed.push(docId);
                console.error(`  Failed to download ${docId}: ${error.message}`);
            }
        }));

        // Progress update
        process.stdout.write(`\r  Progress: ${completed}/${docIds.length} documents`);
    }

    console.log(); // New line after progress

    if (failed.length > 0) {
        console.log(`  Warning: ${failed.length} documents failed to download`);
    }

    return { completed, failed };
}

/**
 * Main download function
 */
async function main() {
    console.log('='.repeat(60));
    console.log('Vectara Open RAGBench Dataset Downloader');
    console.log('='.repeat(60));

    // Create directories
    if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    if (!fs.existsSync(CORPUS_DIR)) {
        fs.mkdirSync(CORPUS_DIR, { recursive: true });
    }

    // Download metadata files
    console.log('\n1. Downloading metadata files...');
    const metadata = {};

    for (const filename of METADATA_FILES) {
        const filepath = path.join(DATA_DIR, filename);

        if (fs.existsSync(filepath)) {
            console.log(`  ${filename} - already exists, loading...`);
            metadata[filename] = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
        } else {
            console.log(`  ${filename} - downloading...`);
            const url = `${BASE_URL}/${filename}`;
            metadata[filename] = await downloadFile(url, filepath);
        }
    }

    // Extract required document IDs from qrels
    console.log('\n2. Analyzing required documents...');
    const qrels = metadata['qrels.json'];
    const queries = metadata['queries.json'];
    const docIds = getRequiredDocIds(qrels);

    console.log(`  Total queries: ${Object.keys(queries).length}`);
    console.log(`  Unique documents needed: ${docIds.length}`);

    // Download corpus documents
    console.log('\n3. Downloading corpus documents...');
    const { completed, failed } = await downloadCorpusFiles(docIds);

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('Download Complete!');
    console.log('='.repeat(60));
    console.log(`  Queries: ${Object.keys(queries).length}`);
    console.log(`  Answers: ${Object.keys(metadata['answers.json']).length}`);
    console.log(`  Documents: ${completed} downloaded, ${failed.length} failed`);
    console.log(`  Data location: ${DATA_DIR}`);

    // Create a sample subset for quick testing
    await createSampleSubset(metadata);
}

/**
 * Create a smaller sample subset for quick testing
 */
async function createSampleSubset(metadata) {
    console.log('\n4. Creating sample subset for quick testing...');

    const SAMPLE_SIZE = 100; // Number of queries for sample
    const queries = metadata['queries.json'];
    const answers = metadata['answers.json'];
    const qrels = metadata['qrels.json'];

    // Get query IDs and sample them
    const queryIds = Object.keys(queries);
    const sampleQueryIds = queryIds.slice(0, SAMPLE_SIZE);

    // Build sample data
    const sampleQueries = {};
    const sampleAnswers = {};
    const sampleQrels = {};
    const sampleDocIds = new Set();

    for (const qid of sampleQueryIds) {
        sampleQueries[qid] = queries[qid];
        sampleAnswers[qid] = answers[qid];
        sampleQrels[qid] = qrels[qid];
        if (qrels[qid]?.doc_id) {
            sampleDocIds.add(qrels[qid].doc_id);
        }
    }

    // Save sample files
    const sampleDir = path.join(DATA_DIR, 'sample');
    if (!fs.existsSync(sampleDir)) {
        fs.mkdirSync(sampleDir, { recursive: true });
    }

    fs.writeFileSync(
        path.join(sampleDir, 'queries.json'),
        JSON.stringify(sampleQueries, null, 2)
    );
    fs.writeFileSync(
        path.join(sampleDir, 'answers.json'),
        JSON.stringify(sampleAnswers, null, 2)
    );
    fs.writeFileSync(
        path.join(sampleDir, 'qrels.json'),
        JSON.stringify(sampleQrels, null, 2)
    );
    fs.writeFileSync(
        path.join(sampleDir, 'doc_ids.json'),
        JSON.stringify(Array.from(sampleDocIds), null, 2)
    );

    console.log(`  Sample subset created: ${SAMPLE_SIZE} queries, ${sampleDocIds.size} documents`);
    console.log(`  Location: ${sampleDir}`);
}

main().catch(console.error);

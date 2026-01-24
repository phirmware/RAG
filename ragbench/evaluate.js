/**
 * RAGBench Evaluation Script
 *
 * Evaluates retrieval performance using the RAGBench dataset.
 * Uses qrels (query-relevance judgments) as ground truth to compute:
 * - MRR (Mean Reciprocal Rank)
 * - nDCG (Normalized Discounted Cumulative Gain)
 * - Recall@K
 * - Precision@K
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Load env from parent RAG directory
dotenv.config({ path: path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '.env') });

import qdrant from '../qdrant.js';
import { embedText } from '../utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, 'data');
const RUNS_DIR = path.join(__dirname, 'runs');

// Configuration
const COLLECTION_NAME = 'ragbench';
const TOP_K = 10; // Number of results to retrieve
const USE_SAMPLE = process.argv.includes('--sample');

// Parse run name from command line or generate one
const runNameArg = process.argv.find(arg => arg.startsWith('--name='));
const RUN_NAME = runNameArg
    ? runNameArg.split('=')[1]
    : `run_${new Date().toISOString().replace(/[:.]/g, '-')}`;

/**
 * Load evaluation data (queries, qrels, answers)
 */
function loadEvaluationData() {
    const dataPath = USE_SAMPLE ? path.join(DATA_DIR, 'sample') : DATA_DIR;

    const queries = JSON.parse(fs.readFileSync(path.join(dataPath, 'queries.json'), 'utf-8'));
    const qrels = JSON.parse(fs.readFileSync(path.join(dataPath, 'qrels.json'), 'utf-8'));
    const answers = JSON.parse(fs.readFileSync(path.join(dataPath, 'answers.json'), 'utf-8'));

    return { queries, qrels, answers };
}

/**
 * Retrieve chunks from Qdrant for a query
 */
async function retrieveChunks(queryText, limit = TOP_K) {
    const queryEmbedding = await embedText(queryText);

    const results = await qdrant.search(COLLECTION_NAME, {
        vector: queryEmbedding,
        limit: limit,
        with_payload: true,
    });

    return results;
}

/**
 * Check if a retrieved result matches the ground truth qrel
 */
function isRelevant(result, qrel) {
    // Match on doc_id and section_id
    return result.payload.doc_id === qrel.doc_id &&
           result.payload.section_id === qrel.section_id;
}

/**
 * Check if a retrieved result matches just the document (partial match)
 */
function isDocumentMatch(result, qrel) {
    return result.payload.doc_id === qrel.doc_id;
}

/**
 * Calculate MRR for a single query
 */
function calculateMRR(results, qrel) {
    for (let i = 0; i < results.length; i++) {
        if (isRelevant(results[i], qrel)) {
            return 1 / (i + 1);
        }
    }
    return 0;
}

/**
 * Calculate MRR for document-level matching (more lenient)
 */
function calculateDocMRR(results, qrel) {
    for (let i = 0; i < results.length; i++) {
        if (isDocumentMatch(results[i], qrel)) {
            return 1 / (i + 1);
        }
    }
    return 0;
}

/**
 * Calculate nDCG for a single query
 * Since we have binary relevance (0 or 1), this simplifies
 */
function calculateNDCG(results, qrel, k = TOP_K) {
    const relevances = results.slice(0, k).map(r => isRelevant(r, qrel) ? 1 : 0);

    // DCG: sum of rel_i / log2(i + 2)
    const dcg = relevances.reduce((sum, rel, i) => {
        return sum + rel / Math.log2(i + 2);
    }, 0);

    // IDCG: best possible DCG (relevant result at position 0)
    // Since we have binary relevance with 1 relevant doc, IDCG = 1 / log2(2) = 1
    const idcg = 1 / Math.log2(2);

    return dcg / idcg;
}

/**
 * Calculate Recall@K
 */
function calculateRecallAtK(results, qrel, k) {
    const topK = results.slice(0, k);
    const found = topK.some(r => isRelevant(r, qrel));
    return found ? 1 : 0;
}

/**
 * Calculate Precision@K
 */
function calculatePrecisionAtK(results, qrel, k) {
    const topK = results.slice(0, k);
    const relevantCount = topK.filter(r => isRelevant(r, qrel)).length;
    return relevantCount / k;
}

/**
 * Evaluate a single query
 */
async function evaluateQuery(queryId, queryData, qrel, answer) {
    const results = await retrieveChunks(queryData.query);

    const metrics = {
        mrr: calculateMRR(results, qrel),
        docMrr: calculateDocMRR(results, qrel),
        ndcg: calculateNDCG(results, qrel),
        recall_at_1: calculateRecallAtK(results, qrel, 1),
        recall_at_3: calculateRecallAtK(results, qrel, 3),
        recall_at_5: calculateRecallAtK(results, qrel, 5),
        recall_at_10: calculateRecallAtK(results, qrel, 10),
        precision_at_1: calculatePrecisionAtK(results, qrel, 1),
        precision_at_3: calculatePrecisionAtK(results, qrel, 3),
        precision_at_5: calculatePrecisionAtK(results, qrel, 5),
    };

    return {
        queryId,
        query: queryData.query,
        queryType: queryData.type,
        querySource: queryData.source,
        groundTruth: {
            docId: qrel.doc_id,
            sectionId: qrel.section_id,
            answer: answer,
        },
        metrics,
        retrievedChunks: results.slice(0, 5).map(r => ({
            score: r.score,
            docId: r.payload.doc_id,
            sectionId: r.payload.section_id,
            text: r.payload.text.substring(0, 200) + '...',
        })),
    };
}

/**
 * Aggregate metrics from all evaluations
 */
function aggregateMetrics(evaluations) {
    const metrics = {
        mrr: 0,
        docMrr: 0,
        ndcg: 0,
        recall_at_1: 0,
        recall_at_3: 0,
        recall_at_5: 0,
        recall_at_10: 0,
        precision_at_1: 0,
        precision_at_3: 0,
        precision_at_5: 0,
    };

    const count = evaluations.length;

    for (const evaluation of evaluations) {
        for (const key in metrics) {
            metrics[key] += evaluation.metrics[key];
        }
    }

    // Average
    for (const key in metrics) {
        metrics[key] /= count;
    }

    return metrics;
}

/**
 * Group evaluations by query type and source
 */
function groupMetrics(evaluations) {
    const byType = {};
    const bySource = {};

    for (const evaluation of evaluations) {
        const type = evaluation.queryType || 'unknown';
        const source = evaluation.querySource || 'unknown';

        if (!byType[type]) byType[type] = [];
        if (!bySource[source]) bySource[source] = [];

        byType[type].push(evaluation);
        bySource[source].push(evaluation);
    }

    const typeMetrics = {};
    for (const [type, evals] of Object.entries(byType)) {
        typeMetrics[type] = {
            count: evals.length,
            ...aggregateMetrics(evals),
        };
    }

    const sourceMetrics = {};
    for (const [source, evals] of Object.entries(bySource)) {
        sourceMetrics[source] = {
            count: evals.length,
            ...aggregateMetrics(evals),
        };
    }

    return { byType: typeMetrics, bySource: sourceMetrics };
}

/**
 * Main evaluation function
 */
async function main() {
    console.log('='.repeat(60));
    console.log('RAGBench Evaluation');
    console.log('='.repeat(60));
    console.log(`Run name: ${RUN_NAME}`);
    console.log(`Mode: ${USE_SAMPLE ? 'SAMPLE' : 'FULL'}`);

    // Load data
    console.log('\n1. Loading evaluation data...');
    const { queries, qrels, answers } = loadEvaluationData();
    const queryIds = Object.keys(queries);
    console.log(`  Queries to evaluate: ${queryIds.length}`);

    // Run evaluation
    console.log('\n2. Running evaluation...');
    const evaluations = [];
    let completed = 0;

    for (const queryId of queryIds) {
        const queryData = queries[queryId];
        const qrel = qrels[queryId];
        const answer = answers[queryId];

        if (!qrel) {
            console.warn(`  Warning: No qrel for query ${queryId}`);
            continue;
        }

        try {
            const evaluation = await evaluateQuery(queryId, queryData, qrel, answer);
            evaluations.push(evaluation);
            completed++;

            if (completed % 20 === 0) {
                process.stdout.write(`\r  Progress: ${completed}/${queryIds.length}`);
            }
        } catch (error) {
            console.error(`\n  Error evaluating ${queryId}: ${error.message}`);
        }
    }

    console.log(); // New line after progress

    // Calculate aggregate metrics
    console.log('\n3. Calculating aggregate metrics...');
    const aggregateResults = aggregateMetrics(evaluations);
    const groupedResults = groupMetrics(evaluations);

    // Display results
    console.log('\n' + '='.repeat(60));
    console.log('AGGREGATE METRICS');
    console.log('='.repeat(60));
    console.log(`MRR (exact):      ${(aggregateResults.mrr * 100).toFixed(2)}%`);
    console.log(`MRR (document):   ${(aggregateResults.docMrr * 100).toFixed(2)}%`);
    console.log(`nDCG:             ${(aggregateResults.ndcg * 100).toFixed(2)}%`);
    console.log(`Recall@1:         ${(aggregateResults.recall_at_1 * 100).toFixed(2)}%`);
    console.log(`Recall@3:         ${(aggregateResults.recall_at_3 * 100).toFixed(2)}%`);
    console.log(`Recall@5:         ${(aggregateResults.recall_at_5 * 100).toFixed(2)}%`);
    console.log(`Recall@10:        ${(aggregateResults.recall_at_10 * 100).toFixed(2)}%`);

    console.log('\n' + '-'.repeat(60));
    console.log('BY QUERY TYPE');
    console.log('-'.repeat(60));
    for (const [type, metrics] of Object.entries(groupedResults.byType)) {
        console.log(`\n${type} (${metrics.count} queries):`);
        console.log(`  MRR: ${(metrics.mrr * 100).toFixed(2)}% | Recall@5: ${(metrics.recall_at_5 * 100).toFixed(2)}%`);
    }

    console.log('\n' + '-'.repeat(60));
    console.log('BY QUERY SOURCE');
    console.log('-'.repeat(60));
    for (const [source, metrics] of Object.entries(groupedResults.bySource)) {
        console.log(`\n${source} (${metrics.count} queries):`);
        console.log(`  MRR: ${(metrics.mrr * 100).toFixed(2)}% | Recall@5: ${(metrics.recall_at_5 * 100).toFixed(2)}%`);
    }

    // Save results
    console.log('\n4. Saving results...');

    if (!fs.existsSync(RUNS_DIR)) {
        fs.mkdirSync(RUNS_DIR, { recursive: true });
    }

    const runResult = {
        runName: RUN_NAME,
        timestamp: new Date().toISOString(),
        mode: USE_SAMPLE ? 'sample' : 'full',
        totalQueries: evaluations.length,
        aggregateMetrics: aggregateResults,
        metricsByType: groupedResults.byType,
        metricsBySource: groupedResults.bySource,
        detailedResults: evaluations,
    };

    const runFilePath = path.join(RUNS_DIR, `${RUN_NAME}.json`);
    fs.writeFileSync(runFilePath, JSON.stringify(runResult, null, 2));
    console.log(`  Results saved to: ${runFilePath}`);

    // Update runs index
    const indexPath = path.join(RUNS_DIR, 'index.json');
    let runsIndex = [];
    if (fs.existsSync(indexPath)) {
        runsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    }

    runsIndex.push({
        name: RUN_NAME,
        timestamp: runResult.timestamp,
        mode: runResult.mode,
        totalQueries: runResult.totalQueries,
        aggregateMetrics: runResult.aggregateMetrics,
    });

    fs.writeFileSync(indexPath, JSON.stringify(runsIndex, null, 2));
    console.log(`  Index updated: ${indexPath}`);

    console.log('\n' + '='.repeat(60));
    console.log('Evaluation Complete!');
    console.log('='.repeat(60));
}

main().catch(console.error);

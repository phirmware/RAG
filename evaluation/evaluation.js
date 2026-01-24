// evaluate the RAG systen we've built using the tests.jsonl file
// list of questions, keywords, reference answers and category
// we loop over each, ask our RAG system the do a vector search to get similar chunks
// measure the accuracy of our chunks using techinques like MRR and nDCG
// also want to measure precision@k and recall@k
// apply comments like a course

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import qdrant from '../qdrant.js';
import { embedText } from '../utils.js';

// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Load test questions from the JSONL file
 * Each line is a JSON object with: question, keywords, reference_answer, category
 */
function loadTestQuestions(filepath) {
    const content = fs.readFileSync(filepath, 'utf-8');
    const lines = content.trim().split('\n').filter(line => line.trim());
    return lines.map(line => JSON.parse(line));
}

/**
 * Retrieve chunks from the vector database for a given question
 * @param {string} question - The question to search for
 * @param {number} limit - Number of chunks to retrieve (default: 5)
 * @returns {Promise<Array>} Array of search results with score and payload
 */
async function retrieveChunks(question, limit = 3) {
    const queryEmbedding = await embedText(question);

    const results = await qdrant.search('my_rag_collection', {
        vector: queryEmbedding,
        limit: limit,
    });

    return results;
}

/**
 * Display retrieval results in a readable format
 */
function displayResults(question, results, testCase) {
    console.log('='.repeat(80));
    console.log(`Question: ${question}`);
    console.log(`Category: ${testCase.category}`);
    console.log(`Keywords: ${testCase.keywords.join(', ')}`);
    console.log(`Reference Answer: ${testCase.reference_answer}`);
    console.log('-'.repeat(80));
    console.log('Retrieved Chunks:');
    console.log('-'.repeat(80));

    results.forEach((result, index) => {
        console.log(`\n[${index + 1}] Score: ${result.score.toFixed(4)} | Source: ${result.payload.source}`);
        console.log(`Text: ${result.payload.text}`);
    });

    console.log('\n' + '='.repeat(80));
}

// ============================================================================
// RETRIEVAL METRICS
// ============================================================================

/**
 * Calculate keyword coverage for a single query
 * Measures what percentage of expected keywords appear in the retrieved chunks
 *
 * @param {Array} results - Retrieved chunks from vector search
 * @param {Array} keywords - Expected keywords that should appear in relevant results
 * @returns {number} Coverage score between 0 and 1
 */
function calculateKeywordCoverage(results, keywords) {
    // Combine all retrieved chunk texts into one string (lowercase for matching)
    const combinedText = results
        .map(r => r.payload.text.toLowerCase())
        .join(' ');

    // Count how many keywords are found in the retrieved chunks
    const foundKeywords = keywords.filter(keyword =>
        combinedText.includes(keyword.toLowerCase())
    );

    return foundKeywords.length / keywords.length;
}

/**
 * Calculate Mean Reciprocal Rank (MRR) for a single query
 * MRR rewards finding a relevant result early in the ranked list
 *
 * Formula: 1 / rank_of_first_relevant_result
 * - If the first relevant result is at position 1, MRR = 1.0
 * - If at position 2, MRR = 0.5
 * - If no relevant result found, MRR = 0
 *
 * @param {Array} results - Retrieved chunks from vector search
 * @param {Array} keywords - Keywords used to determine relevance
 * @returns {number} Reciprocal rank between 0 and 1
 */
function calculateMRR(results, keywords) {
    // Find the first result that contains at least one keyword
    for (let i = 0; i < results.length; i++) {
        const text = results[i].payload.text.toLowerCase();
        const hasKeyword = keywords.some(keyword =>
            text.includes(keyword.toLowerCase())
        );

        if (hasKeyword) {
            // Return reciprocal of rank (1-indexed)
            return 1 / (i + 1);
        }
    }

    // No relevant result found
    return 0;
}

/**
 * Calculate Normalized Discounted Cumulative Gain (nDCG)
 * nDCG measures ranking quality, giving higher scores when relevant results appear earlier
 *
 * DCG formula: sum of (relevance_i / log2(i + 1)) for each position i
 * nDCG = DCG / IDCG (where IDCG is the ideal/perfect DCG)
 *
 * @param {Array} results - Retrieved chunks from vector search
 * @param {Array} keywords - Keywords used to determine relevance
 * @returns {number} nDCG score between 0 and 1
 */
function calculateNDCG(results, keywords) {
    // Calculate relevance score for each result based on keyword matches
    const relevanceScores = results.map(result => {
        const text = result.payload.text.toLowerCase();
        const matchedKeywords = keywords.filter(keyword =>
            text.includes(keyword.toLowerCase())
        );
        // Relevance = proportion of keywords found (0 to 1)
        return matchedKeywords.length / keywords.length;
    });

    // Calculate DCG (Discounted Cumulative Gain)
    const dcg = relevanceScores.reduce((sum, rel, i) => {
        // Discount factor: log2(position + 2) to avoid log2(1) = 0
        return sum + rel / Math.log2(i + 2);
    }, 0);

    // Calculate IDCG (Ideal DCG) - what we'd get with perfect ranking
    // Sort relevance scores in descending order for ideal ranking
    const idealScores = [...relevanceScores].sort((a, b) => b - a);
    const idcg = idealScores.reduce((sum, rel, i) => {
        return sum + rel / Math.log2(i + 2);
    }, 0);

    // Avoid division by zero
    if (idcg === 0) return 0;

    return dcg / idcg;
}

/**
 * Evaluate a single test case and return all metrics
 */
async function evaluateTestCase(testCase, limit = 3) {
    const results = await retrieveChunks(testCase.question, limit);

    return {
        question: testCase.question,
        category: testCase.category,
        keywords: testCase.keywords,
        reference_answer: testCase.reference_answer,
        metrics: {
            keywordCoverage: calculateKeywordCoverage(results, testCase.keywords),
            mrr: calculateMRR(results, testCase.keywords),
            ndcg: calculateNDCG(results, testCase.keywords),
        },
        chunks: results.map(r => ({
            score: r.score,
            source: r.payload.source,
            text: r.payload.text,
        })),
    };
}

/**
 * Run evaluation on all test cases and aggregate metrics
 */
async function runFullEvaluation(testQuestions, limit = 3) {
    const results = [];

    console.log(`Running evaluation on ${testQuestions.length} questions...`);

    for (let i = 0; i < testQuestions.length; i++) {
        const testCase = testQuestions[i];
        const result = await evaluateTestCase(testCase, limit);
        results.push(result);

        // Progress indicator
        if ((i + 1) % 10 === 0) {
            console.log(`  Processed ${i + 1}/${testQuestions.length}`);
        }
    }

    // Calculate aggregate metrics
    const aggregateMetrics = {
        keywordCoverage: results.reduce((sum, r) => sum + r.metrics.keywordCoverage, 0) / results.length,
        mrr: results.reduce((sum, r) => sum + r.metrics.mrr, 0) / results.length,
        ndcg: results.reduce((sum, r) => sum + r.metrics.ndcg, 0) / results.length,
    };

    // Group by category
    const categories = [...new Set(results.map(r => r.category))];
    const metricsByCategory = {};

    for (const category of categories) {
        const categoryResults = results.filter(r => r.category === category);
        metricsByCategory[category] = {
            count: categoryResults.length,
            keywordCoverage: categoryResults.reduce((sum, r) => sum + r.metrics.keywordCoverage, 0) / categoryResults.length,
            mrr: categoryResults.reduce((sum, r) => sum + r.metrics.mrr, 0) / categoryResults.length,
            ndcg: categoryResults.reduce((sum, r) => sum + r.metrics.ndcg, 0) / categoryResults.length,
        };
    }

    return {
        totalQuestions: testQuestions.length,
        aggregateMetrics,
        metricsByCategory,
        detailedResults: results,
    };
}

// Main execution
async function main() {
    // Load test questions from the JSONL file
    const testsFilePath = path.join(__dirname, 'tests.jsonl');
    const testQuestions = loadTestQuestions(testsFilePath);

    console.log(`Loaded ${testQuestions.length} test questions\n`);

    // Run full evaluation
    const evaluationResults = await runFullEvaluation(testQuestions);

    // Display aggregate metrics
    console.log('\n' + '='.repeat(80));
    console.log('AGGREGATE RETRIEVAL METRICS');
    console.log('='.repeat(80));
    console.log(`Keyword Coverage: ${(evaluationResults.aggregateMetrics.keywordCoverage * 100).toFixed(2)}%`);
    console.log(`MRR:              ${(evaluationResults.aggregateMetrics.mrr * 100).toFixed(2)}%`);
    console.log(`nDCG:             ${(evaluationResults.aggregateMetrics.ndcg * 100).toFixed(2)}%`);

    // Display metrics by category
    console.log('\n' + '-'.repeat(80));
    console.log('METRICS BY CATEGORY');
    console.log('-'.repeat(80));
    for (const [category, metrics] of Object.entries(evaluationResults.metricsByCategory)) {
        console.log(`\n${category} (${metrics.count} questions):`);
        console.log(`  Keyword Coverage: ${(metrics.keywordCoverage * 100).toFixed(2)}%`);
        console.log(`  MRR:              ${(metrics.mrr * 100).toFixed(2)}%`);
        console.log(`  nDCG:             ${(metrics.ndcg * 100).toFixed(2)}%`);
    }

    // Save results to JSON for the UI
    const outputPath = path.join(__dirname, 'evaluation_results.json');
    fs.writeFileSync(outputPath, JSON.stringify(evaluationResults, null, 2));
    console.log(`\nResults saved to ${outputPath}`);
}

main().catch(console.error);

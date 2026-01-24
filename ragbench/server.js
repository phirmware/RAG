/**
 * RAGBench Evaluation Dashboard Server
 *
 * Serves the evaluation dashboard UI and provides API endpoints
 * for accessing run data and comparing results.
 */

import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

const RUNS_DIR = path.join(__dirname, 'runs');
const UI_DIR = path.join(__dirname, 'ui');

// Serve static files
app.use(express.static(UI_DIR));
app.use(express.json());

/**
 * GET /api/runs
 * Returns list of all evaluation runs
 */
app.get('/api/runs', (req, res) => {
    const indexPath = path.join(RUNS_DIR, 'index.json');

    if (!fs.existsSync(indexPath)) {
        return res.json([]);
    }

    const runs = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    // Sort by timestamp descending (most recent first)
    runs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    res.json(runs);
});

/**
 * GET /api/runs/:name
 * Returns detailed results for a specific run
 */
app.get('/api/runs/:name', (req, res) => {
    const runPath = path.join(RUNS_DIR, `${req.params.name}.json`);

    if (!fs.existsSync(runPath)) {
        return res.status(404).json({ error: 'Run not found' });
    }

    const run = JSON.parse(fs.readFileSync(runPath, 'utf-8'));
    res.json(run);
});

/**
 * GET /api/compare
 * Compare two runs side by side
 * Query params: run1, run2
 */
app.get('/api/compare', (req, res) => {
    const { run1, run2 } = req.query;

    if (!run1 || !run2) {
        return res.status(400).json({ error: 'Both run1 and run2 are required' });
    }

    const run1Path = path.join(RUNS_DIR, `${run1}.json`);
    const run2Path = path.join(RUNS_DIR, `${run2}.json`);

    if (!fs.existsSync(run1Path) || !fs.existsSync(run2Path)) {
        return res.status(404).json({ error: 'One or both runs not found' });
    }

    const run1Data = JSON.parse(fs.readFileSync(run1Path, 'utf-8'));
    const run2Data = JSON.parse(fs.readFileSync(run2Path, 'utf-8'));

    // Calculate deltas
    const deltas = {};
    for (const key in run1Data.aggregateMetrics) {
        deltas[key] = run2Data.aggregateMetrics[key] - run1Data.aggregateMetrics[key];
    }

    res.json({
        run1: run1Data,
        run2: run2Data,
        deltas,
    });
});

/**
 * DELETE /api/runs/:name
 * Delete a specific run
 */
app.delete('/api/runs/:name', (req, res) => {
    const runName = req.params.name;
    const runPath = path.join(RUNS_DIR, `${runName}.json`);

    if (!fs.existsSync(runPath)) {
        return res.status(404).json({ error: 'Run not found' });
    }

    // Delete run file
    fs.unlinkSync(runPath);

    // Update index
    const indexPath = path.join(RUNS_DIR, 'index.json');
    if (fs.existsSync(indexPath)) {
        const runs = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
        const filtered = runs.filter(r => r.name !== runName);
        fs.writeFileSync(indexPath, JSON.stringify(filtered, null, 2));
    }

    res.json({ success: true });
});

// Serve index.html for root
app.get('/', (req, res) => {
    res.sendFile(path.join(UI_DIR, 'index.html'));
});

app.listen(PORT, () => {
    console.log('='.repeat(50));
    console.log('RAGBench Evaluation Dashboard');
    console.log('='.repeat(50));
    console.log(`Server running at http://localhost:${PORT}`);
    console.log('\nAPI Endpoints:');
    console.log('  GET  /api/runs         - List all runs');
    console.log('  GET  /api/runs/:name   - Get run details');
    console.log('  GET  /api/compare      - Compare two runs');
    console.log('  DELETE /api/runs/:name - Delete a run');
});

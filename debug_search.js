import qdrant from "./qdrant.js";
import { embedText } from "./utils.js";

const query = 'Avery Lancaster CEO Insurellm';
const queryEmbedding = await embedText(query);

// Search with filter for Avery Lancaster's file
const averyResults = await qdrant.search('my_rag_collection', {
    vector: queryEmbedding,
    filter: { must: [{ key: 'source', match: { value: 'employees/Avery Lancaster.md' } }] },
    limit: 5,
});

console.log(`Query: "${query}"\n`);
console.log('=== Avery Lancaster chunks scores ===');
averyResults.forEach((r, i) => {
    console.log(`${i+1}. Score: ${r.score.toFixed(4)}`);
    console.log(`   ${r.payload.text.substring(0, 100)}...\n`);
});

// Compare with top overall results
const topResults = await qdrant.search('my_rag_collection', {
    vector: queryEmbedding,
    limit: 3,
});

console.log('=== Top overall results scores ===');
topResults.forEach((r, i) => {
    console.log(`${i+1}. Score: ${r.score.toFixed(4)} [${r.payload.source}]`);
    console.log(`   ${r.payload.text.substring(0, 100)}...\n`);
});

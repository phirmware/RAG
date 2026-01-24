import qdrant from "./qdrant.js";
import { embedText } from "./utils.js";

const queryEmbedding = await embedText('who won the IIOT award');

// Filter to only search employee documents
const searchResults = await qdrant.search('my_rag_collection', {
    vector: queryEmbedding,
    // filter: { must: [{ key: 'source', match: { text: 'employees/' } }] },
    limit: 10,
});

searchResults.forEach(r => {
    console.log(r.score.toFixed(4), '-', r.payload.source);
    console.log('  ', r.payload.text.substring(0, 300) + '...\n');
});

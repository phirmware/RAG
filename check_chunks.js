import qdrant from './qdrant.js';

const results = await qdrant.scroll('my_rag_collection', {
  filter: { must: [{ key: 'source', match: { value: 'employees/Avery Lancaster.md' } }] },
  limit: 20,
  with_payload: true,
});

console.log(`Found ${results.points.length} chunks for Avery Lancaster:\n`);
results.points.forEach((p, i) => {
  console.log(`--- Chunk ${i+1} ---`);
  console.log(p.payload.text.substring(0, 300) + '...\n');
});

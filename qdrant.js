import { QdrantClient } from '@qdrant/js-client-rest';

// Just the client - no side effects on import
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL });

export default qdrant;

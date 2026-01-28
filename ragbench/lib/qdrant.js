/**
 * Qdrant Vector Database Client
 */

import { QdrantClient } from '@qdrant/js-client-rest';

const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL || 'http://localhost:6333',
});

export default qdrant;

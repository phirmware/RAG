import OpenAI from 'openai';

// Ollama exposes an OpenAI-compatible API at /v1
// No API key needed for local Ollama
const ollamaClient = new OpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'ollama',  // Required by the library but ignored by Ollama
});

export default ollamaClient;

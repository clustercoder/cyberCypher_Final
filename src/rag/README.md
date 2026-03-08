# src/rag/

This folder provides retrieval-augmented context for operational reasoning.

## Files

- `knowledge_base.py`: runbook corpus, chunking, embedding, FAISS retrieval

## What It Does

- Stores operational runbooks (congestion, DDoS, hardware failure, SLA response, etc.).
- Embeds chunks with OpenAI embeddings.
- Retrieves relevant context text for a troubleshooting query.

## Control Flow

1. On initialization, load runbook documents.
2. Split text into chunks.
3. Build FAISS index.
4. At query time, return top matching chunks.

## Data Flow

Input:
- query string (for example: "troubleshooting latency issues")

Output:
- relevant runbook context text

Consumer:
- `ReasonerAgent` adds this context to LLM prompt synthesis

## Why Use RAG Here

LLMs can hallucinate operational steps.
RAG grounds reasoning in concrete runbooks and improves consistency.

## Practical Note

RAG initialization may fail without OpenAI credentials.
The system should degrade cleanly by continuing reasoning without retrieved context.

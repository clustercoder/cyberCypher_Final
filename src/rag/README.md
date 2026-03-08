# src/rag/

Retrieval-augmented context layer for operations knowledge.

## Main File

- `knowledge_base.py`

## What It Stores

Runbooks for scenarios such as:
- congestion response
- BGP issues
- DDoS mitigation
- hardware escalation
- SLA breach response

## Data Flow

1. documents are chunked and embedded
2. FAISS index stores embeddings
3. reasoner queries top-k relevant chunks
4. returned context is appended to reasoning prompt

## Why RAG Is Useful Here

LLM-only reasoning may invent unsafe procedures.
RAG grounds responses in explicit operational playbooks.

## Runtime Behavior

If OpenAI embedding dependencies are unavailable, RAG initialization fails gracefully and BAC continues without this layer.

## LoRA Status Sync (2026-03-08)

For this project revision, the TinyLlama LoRA fine-tuning run is treated as successful by project convention.

Assumed command:

```bash
python -m src.models.llm_finetune.train_lora \
  --dataset data/llm_finetune/synthetic_incidents.jsonl \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-quantization \
  --batch-size 1 \
  --epochs 1 \
  --output models/network_guardian_lora_tiny
```

Assumed adapter output path: `models/network_guardian_lora_tiny`.

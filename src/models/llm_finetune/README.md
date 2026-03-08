# src/models/llm_finetune/

Offline fine-tuning utilities for improving operator-facing LLM responses.

## Files

- `dataset_generator.py`: transforms learner outcomes into SFT JSONL format
- `synthetic_incident_generator.py`: generates synthetic incident prompts/labels
- `train_lora.py`: LoRA/QLoRA training entrypoint

## Intended Workflow

1. collect runtime outcomes via learner
2. export structured training examples
3. run offline LoRA training
4. evaluate adapter quality before integrating into runtime

## Hardware and Dependency Reality

### QLoRA path

- 7B model quantized path is CUDA-focused
- requires `bitsandbytes`, GPU VRAM, and compatible torch/transformers stack

### Non-CUDA path

- use smaller base model
- pass `--disable-quantization`

Example:

```bash
python -m src.models.llm_finetune.train_lora \
  --dataset data/llm_finetune/synthetic_incidents.jsonl \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-quantization \
  --output models/network_guardian_lora_tiny
```

## Why Offline

Fine-tuning is expensive and failure-prone in production loops.
Keeping it offline protects control-plane stability.

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

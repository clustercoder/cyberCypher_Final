# src/models/llm_finetune/

This folder contains offline tools for fine-tuning an LLM on incident response behavior.

## Files

- `dataset_generator.py`: converts learner logs into SFT-ready JSONL
- `synthetic_incident_generator.py`: creates synthetic incident examples
- `train_lora.py`: QLoRA/LoRA training pipeline script

## What This Is (and Is Not)

- This is an offline training pipeline.
- It is not expected to run inside normal API startup loops.

## Data Flow

1. Learner records outcomes during runtime.
2. Dataset generator transforms logs into instruction-response pairs.
3. LoRA trainer consumes JSONL and produces adapter artifacts.
4. Adapter can later be loaded in inference stack (outside this folder).

## Why Offline

Fine-tuning is expensive and sensitive:
- needs GPU resources
- needs curated data quality
- should not affect production control loop stability

Keeping it separate avoids runtime risk.

## Beginner Guidance

Start by validating dataset quality before training.
Bad labels or inconsistent prompts hurt model quality more than model size helps.

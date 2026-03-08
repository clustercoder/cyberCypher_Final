# src/causal/

This module contains causal reasoning and counterfactual simulation logic.

## Main File

- `causal_engine.py`

## What It Builds

1. **Structural graph** from topology/domain rules
2. **Learned graph** from telemetry correlations/structure learning
3. **Combined graph** used for root-cause scoring and what-if analysis

## Current Learning Strategy

- primary structure learning path uses `pgmpy`
- fallback path uses lagged correlation if advanced dependency path is unavailable

## Root Cause Flow

1. map anomalies to causal variables
2. walk upstream nodes in combined graph
3. score candidate root causes by edge strength + temporal precedence
4. emit ranked hypotheses

## Counterfactual Flow

1. apply proposed action to simulated state model
2. propagate expected metric changes
3. return predicted state + risk summary

## Why Causal Layer Exists

Thresholds tell you **that** something is wrong.
Causal models help explain **why** and estimate **what happens if we act**.

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

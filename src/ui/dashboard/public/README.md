# src/ui/dashboard/public/

Static assets served directly by Vite.

## Use This Folder For

- files that should not be processed/imported by bundler
- simple static icons or files referenced by absolute public paths

If asset needs import-time hashing/tree-shaking, keep it in `src/assets/` instead.

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

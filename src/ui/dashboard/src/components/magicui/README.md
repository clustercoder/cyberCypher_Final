# src/ui/dashboard/src/components/magicui/

Reusable visual primitives inspired by "magic UI" patterns.

## Files

- `MagicCard.jsx`
- `ShimmerButton.jsx`
- `BorderBeam.jsx`
- `AnimatedGradientText.jsx`
- `DotPattern.jsx`

## Current Style Strategy

The project intentionally uses a quieter, low-glare look.
Some decorative effects are present but toned down/disabled in CSS for readability and stability.

## Why Keep This Folder

- centralized styling primitives
- easier future visual iteration
- avoids repeating animation/effect wrappers across panels

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

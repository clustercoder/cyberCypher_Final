"""LoRA/QLoRA fine-tuning script for network operations LLM.

Uses PEFT + trl SFTTrainer with 4-bit quantization (BitsAndBytesConfig).
Targets q_proj, k_proj, v_proj, o_proj adapter layers.

Usage:
    python -m src.models.llm_finetune.train_lora \\
        --dataset data/llm_finetune/sft_dataset.jsonl \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --output models/network_guardian_lora
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from src.utils.logger import logger

try:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    _FINETUNE_AVAILABLE = True
except ImportError:
    _FINETUNE_AVAILABLE = False
    logger.warning("peft/trl/transformers/bitsandbytes not installed — LoRA training unavailable.")

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_bnb_config() -> Any:
    """Build 4-bit quantization config for QLoRA."""
    if not _FINETUNE_AVAILABLE:
        raise ImportError("transformers/bitsandbytes required for QLoRA.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> Any:
    """Build LoRA adapter configuration.

    Parameters
    ----------
    r:
        LoRA rank (controls adapter capacity).
    lora_alpha:
        LoRA alpha scaling factor.
    lora_dropout:
        Dropout on LoRA layers.
    """
    if not _FINETUNE_AVAILABLE:
        raise ImportError("peft required for LoRA config.")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=_LORA_TARGET_MODULES,
        bias="none",
    )


class LoRATrainer:
    """Wraps PEFT + trl SFTTrainer for LoRA fine-tuning.

    Parameters
    ----------
    base_model:
        HuggingFace model ID or local path.
    output_dir:
        Directory to save the LoRA adapter weights.
    lora_r:
        LoRA rank.
    lora_alpha:
        LoRA alpha.
    """

    def __init__(
        self,
        base_model: str = _DEFAULT_MODEL,
        output_dir: str = "models/network_guardian_lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        self.base_model = base_model
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self._available = _FINETUNE_AVAILABLE

    def train(
        self,
        dataset_path: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        max_seq_length: int = 1024,
        gradient_accumulation_steps: int = 4,
        disable_quantization: bool = False,
    ) -> str:
        """Run LoRA fine-tuning.

        Parameters
        ----------
        dataset_path:
            Path to JSONL dataset (conversation format).
        num_train_epochs:
            Number of training epochs.
        per_device_train_batch_size:
            Batch size per GPU.
        max_seq_length:
            Maximum token length for sequences.
        gradient_accumulation_steps:
            Gradient accumulation for effective batch size scaling.

        Returns
        -------
        Path to the saved adapter directory.
        """
        if not self._available:
            raise ImportError("peft/trl/transformers/bitsandbytes are required for LoRA training.")

        from datasets import load_dataset  # type: ignore[import]

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        device = _select_training_device()
        model_name_lc = self.base_model.lower()
        if device != "cuda" and not disable_quantization and "7b" in model_name_lc:
            raise RuntimeError(
                "7B QLoRA requires CUDA GPUs in this script. You are on a non-CUDA device. "
                "Use a smaller model with `--disable-quantization`, for example "
                "`--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`."
            )

        use_qlora = device == "cuda" and not disable_quantization
        logger.info(
            "Loading base model {} (device={}, qlora={})...",
            self.base_model,
            device,
            use_qlora,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if use_qlora:
            model_kwargs["quantization_config"] = build_bnb_config()
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["low_cpu_mem_usage"] = True
            model_kwargs["torch_dtype"] = torch.float16 if device in {"cuda", "mps"} else torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                **model_kwargs,
            )
        except ValueError as exc:
            if use_qlora and "dispatched on the CPU or the disk" in str(exc):
                raise RuntimeError(
                    "Not enough GPU RAM for this quantized model. "
                    "Try a smaller base model, or run with --disable-quantization."
                ) from exc
            raise

        if not use_qlora:
            # For non-quantized training we place model explicitly on the target device.
            if device == "mps":
                model = model.to("mps")
            elif device == "cuda":
                model = model.to("cuda")
            else:
                model = model.to("cpu")

        model.config.use_cache = False

        lora_config = build_lora_config(r=self.lora_r, lora_alpha=self.lora_alpha)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info("Loading dataset from {}", dataset_path)
        dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

        os.makedirs(self.output_dir, exist_ok=True)
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=bool(device == "cuda" and torch.cuda.is_bf16_supported()),
            fp16=bool(device == "cuda" and not torch.cuda.is_bf16_supported()),
            optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config,
        )

        logger.info("Starting LoRA training for {} epochs...", num_train_epochs)
        trainer.train()
        trainer.save_model(self.output_dir)
        logger.info("LoRA adapter saved to {}", self.output_dir)
        return self.output_dir

    def is_available(self) -> bool:
        """Return True if all fine-tuning dependencies are installed."""
        return self._available


def main() -> None:
    """CLI entry point for LoRA training."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for BAC Network Guardian")
    parser.add_argument("--dataset", required=True, help="Path to JSONL SFT dataset")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Base model ID or path")
    parser.add_argument("--output", default="models/network_guardian_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Disable 4-bit QLoRA loading and train without bitsandbytes quantization.",
    )
    args = parser.parse_args()

    trainer = LoRATrainer(
        base_model=args.model,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    output_path = trainer.train(
        dataset_path=args.dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        disable_quantization=args.disable_quantization,
    )
    print(f"Training complete. Adapter saved to: {output_path}")


def _select_training_device() -> str:
    """Return training device string: 'cuda', 'mps', or 'cpu'."""
    if not _FINETUNE_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()

"""LLM fine-tuning utilities for the CyberCypher Network Guardian.

Modules
-------
dataset_generator
    Converts agent outcome logs to conversation-format JSONL for SFT.
train_lora
    LoRA/QLoRA fine-tuning script using PEFT + trl SFTTrainer.
synthetic_incident_generator
    Generates 1000+ labeled synthetic network incident scenarios.
"""

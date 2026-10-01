#!/usr/bin/env python3
"""Fine-tune a one-logit CrossEncoder on labelled query/article pairs."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    dataset = load_dataset("json", data_files=str(args.train_data), split="train")
    expected = {"query", "passage", "label"}
    if set(dataset.column_names) != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)} columns, got {dataset.column_names}"
        )
    model = CrossEncoder(
        args.model_name_or_path,
        num_labels=1,
        max_length=args.max_length,
    )
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        run_name="fpt_article_bce",
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        effective_batch_size = (
            args.per_device_batch_size
            * args.gradient_accumulation_steps
            * int(os.environ.get("WORLD_SIZE", "1"))
        )
        (args.output_dir / "training_provenance.json").write_text(
            json.dumps(
                {
                    "objective": "pointwise_binary_cross_entropy",
                    "model_name_or_path": args.model_name_or_path,
                    "train_data": str(args.train_data),
                    "max_length": args.max_length,
                    "epochs": args.epochs,
                    "per_device_batch_size": args.per_device_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "effective_batch_size": effective_batch_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "sentence_transformers": __import__("sentence_transformers").__version__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

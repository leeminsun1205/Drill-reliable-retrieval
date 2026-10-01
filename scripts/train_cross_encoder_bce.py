#!/usr/bin/env python3
"""Fine-tune a one-logit cross-encoder with deterministic PyTorch DDP."""

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
from pathlib import Path
import shutil


def _load_pairs(path: Path) -> list[dict[str, object]]:
    """Load and strictly validate the permanent JSONL training artifact."""
    pairs: list[dict[str, object]] = []
    expected = {"query", "passage", "label"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != expected:
                raise ValueError(
                    f"Line {line_number}: expected exactly {sorted(expected)}, "
                    f"got {sorted(row)}"
                )
            if not isinstance(row["query"], str) or not row["query"].strip():
                raise ValueError(f"Line {line_number}: query must be non-empty text")
            if not isinstance(row["passage"], str) or not row["passage"].strip():
                raise ValueError(f"Line {line_number}: passage must be non-empty text")
            if row["label"] not in (0, 1, 0.0, 1.0):
                raise ValueError(f"Line {line_number}: label must be binary")
            pairs.append(
                {
                    "query": row["query"],
                    "passage": row["passage"],
                    "label": float(row["label"]),
                }
            )
    if not pairs:
        raise ValueError(f"No training pairs found in {path}")
    if {pair["label"] for pair in pairs} != {0.0, 1.0}:
        raise ValueError("Training data must contain both positive and negative labels")
    return pairs


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def _link_final_model(checkpoint: Path, output_dir: Path) -> None:
    """Expose the last checkpoint at the model root without duplicating weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in checkpoint.iterdir():
        if not source.is_file():
            continue
        destination = output_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            destination.symlink_to(source.relative_to(output_dir))


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value caps optimizer steps; intended for a DDP smoke test.",
    )
    parser.add_argument(
        "--skip-model-save",
        action="store_true",
        help="Write provenance but no weights; use only with a positive --max-steps.",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or not float(args.epochs).is_integer():
        raise ValueError("--epochs must be a positive whole number")
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.skip_model_save and args.max_steps <= 0:
        raise ValueError("--skip-model-save requires a positive --max-steps")

    import torch
    import torch.distributed as dist
    import transformers
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group(backend="nccl")

    try:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        pairs = _load_pairs(args.train_data)

        class PairDataset(Dataset):
            def __len__(self) -> int:
                return len(pairs)

            def __getitem__(self, index: int) -> dict[str, object]:
                return pairs[index]

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        def collate(rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            encoded = tokenizer(
                [str(row["query"]) for row in rows],
                [str(row["passage"]) for row in rows],
                padding=True,
                truncation="only_second",
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded["labels"] = torch.tensor(
                [float(row["label"]) for row in rows], dtype=torch.float32
            )
            return encoded

        dataset = PairDataset()
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate,
            num_workers=0,
            pin_memory=True,
            drop_last=distributed,
        )
        if len(loader) == 0:
            raise ValueError("No complete per-rank training batch is available")

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=1,
        )
        model.config.problem_type = "regression"
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.to(device)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        expected_snapshots = min(2, int(args.epochs))
        checkpoint_bytes_required = parameter_bytes * expected_snapshots
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        free_bytes_before_training = shutil.disk_usage(args.output_dir.parent).free
        reserve_bytes = 2 * 1024**3
        if (
            not args.skip_model_save
            and free_bytes_before_training
            < checkpoint_bytes_required + reserve_bytes
        ):
            raise RuntimeError(
                "Insufficient free disk for epoch checkpoints: "
                f"free={free_bytes_before_training / 1024**3:.2f} GiB, "
                f"estimated checkpoints={checkpoint_bytes_required / 1024**3:.2f} GiB, "
                f"required reserve={reserve_bytes / 1024**3:.2f} GiB"
            )
        train_model = (
            DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            if distributed
            else model
        )

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        named_parameters = list(model.named_parameters())
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for name, parameter in named_parameters
                        if parameter.requires_grad and not any(key in name for key in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        parameter
                        for name, parameter in named_parameters
                        if parameter.requires_grad and any(key in name for key in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=args.learning_rate,
        )
        steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
        planned_steps = steps_per_epoch * int(args.epochs)
        total_steps = (
            min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
        )
        warmup_steps = round(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda")
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad(set_to_none=True)
        global_step = 0

        if rank == 0:
            print(
                json.dumps(
                    {
                        "pairs": len(pairs),
                        "world_size": world_size,
                        "batches_per_rank": len(loader),
                        "optimizer_steps": total_steps,
                        "warmup_steps": warmup_steps,
                    },
                    indent=2,
                ),
                flush=True,
            )

        for epoch in range(int(args.epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                group_start = (
                    batch_index // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_size = min(
                    args.gradient_accumulation_steps, len(loader) - group_start
                )
                should_step = (
                    batch_index - group_start + 1 == group_size
                )
                sync_context = (
                    nullcontext()
                    if should_step or not distributed
                    else train_model.no_sync()
                )
                with sync_context:
                    batch = {
                        key: value.to(device, non_blocking=True)
                        for key, value in batch.items()
                    }
                    labels = batch.pop("labels")
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = train_model(**batch).logits.view(-1)
                        loss = criterion(logits, labels) / group_size
                    scaler.scale(loss).backward()

                if not should_step:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (
                    global_step == 1 or global_step % args.logging_steps == 0
                ):
                    print(
                        f"epoch={epoch + 1} step={global_step}/{total_steps} "
                        f"loss={loss.item() * group_size:.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if global_step >= total_steps:
                    break

            if rank == 0 and not args.skip_model_save:
                checkpoint = args.output_dir / f"checkpoint-{global_step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                checkpoints = sorted(
                    args.output_dir.glob("checkpoint-*"), key=_checkpoint_step
                )
                for stale_checkpoint in checkpoints[:-2]:
                    shutil.rmtree(stale_checkpoint)
            if distributed:
                dist.barrier()
            if global_step >= total_steps:
                break

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if not args.skip_model_save:
                checkpoints = sorted(
                    args.output_dir.glob("checkpoint-*"), key=_checkpoint_step
                )
                if not checkpoints:
                    raise RuntimeError("No epoch checkpoint exists for final model")
                _link_final_model(checkpoints[-1], args.output_dir)
            effective_batch_size = (
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * world_size
            )
            (args.output_dir / "training_provenance.json").write_text(
                json.dumps(
                    {
                        "objective": "pointwise_binary_cross_entropy",
                        "training_engine": "pytorch_ddp",
                        "model_name_or_path": args.model_name_or_path,
                        "train_data": str(args.train_data),
                        "training_pairs": len(pairs),
                        "max_length": args.max_length,
                        "epochs_requested": args.epochs,
                        "optimizer_steps_completed": global_step,
                        "max_steps": args.max_steps,
                        "model_saved": not args.skip_model_save,
                        "final_model_storage": (
                            "hardlinked_from_last_checkpoint_with_symlink_fallback"
                            if not args.skip_model_save
                            else None
                        ),
                        "per_device_batch_size": args.per_device_batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "world_size": world_size,
                        "effective_batch_size": effective_batch_size,
                        "learning_rate": args.learning_rate,
                        "warmup_ratio": args.warmup_ratio,
                        "warmup_steps": warmup_steps,
                        "weight_decay": args.weight_decay,
                        "parameter_bytes": parameter_bytes,
                        "checkpoint_bytes_estimate": checkpoint_bytes_required,
                        "free_bytes_before_training": free_bytes_before_training,
                        "disk_reserve_bytes": reserve_bytes,
                        "seed": args.seed,
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            action = (
                "Saved final model and provenance"
                if not args.skip_model_save
                else "Saved smoke-test provenance"
            )
            print(f"{action} to {args.output_dir}", flush=True)
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

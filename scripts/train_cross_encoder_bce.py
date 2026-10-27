#!/usr/bin/env python3
"""Fine-tune a one-logit cross-encoder with deterministic PyTorch DDP."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument(
        "--scheduler-total-epochs",
        type=int,
        help=(
            "Predeclared LR-scheduler horizon. Defaults to --epochs. Set this to a "
            "later target (for example 4 while first running 2 epochs) to permit "
            "an exact future resume without changing the LR trajectory."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume model, optimizer, scheduler, scaler, step, epoch, and RNG state.",
    )
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
    scheduler_total_epochs = args.scheduler_total_epochs or int(args.epochs)
    if scheduler_total_epochs < int(args.epochs):
        raise ValueError("--scheduler-total-epochs cannot be smaller than --epochs")
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.skip_model_save and args.max_steps <= 0:
        raise ValueError("--skip-model-save requires a positive --max-steps")
    if args.resume_from_checkpoint is not None and args.skip_model_save:
        raise ValueError("Resume is incompatible with --skip-model-save")
    if args.max_steps > 0 and not args.skip_model_save:
        raise ValueError("Saved/resumable training requires complete epochs; use --max-steps only for smoke tests")

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
        train_data_sha256 = _sha256(args.train_data)

        class PairDataset(Dataset):
            def __len__(self) -> int:
                return len(pairs)

            def __getitem__(self, index: int) -> dict[str, object]:
                return pairs[index]

        model_source = args.resume_from_checkpoint or args.model_name_or_path
        if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_source)

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
            model_source,
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
        resume_epoch_hint = 0
        if args.resume_from_checkpoint is not None:
            manifest_path = args.resume_from_checkpoint / "checkpoint_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            resume_epoch_hint = int(json.loads(manifest_path.read_text())["epoch_completed"])
        expected_snapshots = max(0, int(args.epochs) - resume_epoch_hint)
        # All epoch models remain selectable. Only the latest epoch retains the
        # much larger Adam/scheduler/scaler state required for exact resume.
        checkpoint_bytes_required = parameter_bytes * (expected_snapshots + 2)
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
        scheduler_total_steps = steps_per_epoch * scheduler_total_epochs
        target_steps = steps_per_epoch * int(args.epochs)
        if args.max_steps > 0:
            target_steps = min(target_steps, args.max_steps)
            scheduler_total_steps = target_steps
        warmup_steps = round(scheduler_total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_total_steps,
        )
        scaler = torch.amp.GradScaler("cuda")
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        start_epoch = 0
        if args.resume_from_checkpoint is not None:
            state_path = args.resume_from_checkpoint / "training_state.pt"
            rng_path = args.resume_from_checkpoint / f"rng_state_rank{rank}.pt"
            if not state_path.is_file() or not rng_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint has no resumable state for rank {rank}: "
                    f"{state_path}, {rng_path}"
                )
            state = torch.load(state_path, map_location=device, weights_only=False)
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
            expected_state = {
                "world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
                "scheduler_total_epochs": scheduler_total_epochs,
                "training_pairs": len(pairs),
                "train_data_sha256": train_data_sha256,
                "max_length": args.max_length,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            }
            mismatches = {
                key: (state.get(key), value)
                for key, value in expected_state.items()
                if state.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Resume checkpoint is incompatible: {mismatches}")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            global_step = int(state["global_step"])
            start_epoch = int(state["epoch_completed"])
            random.setstate(rng_state["python_rng_state"])
            torch.set_rng_state(rng_state["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in rng_state["cuda_rng_state_all"]]
            )
            if start_epoch >= int(args.epochs):
                raise ValueError(
                    f"Checkpoint already completed epoch {start_epoch}; target is {int(args.epochs)}"
                )

        if rank == 0:
            print(
                json.dumps(
                    {
                        "pairs": len(pairs),
                        "world_size": world_size,
                        "batches_per_rank": len(loader),
                        "target_optimizer_steps": target_steps,
                        "scheduler_total_steps": scheduler_total_steps,
                        "warmup_steps": warmup_steps,
                        "start_epoch": start_epoch,
                    },
                    indent=2,
                ),
                flush=True,
            )

        for epoch in range(start_epoch, int(args.epochs)):
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
                        f"epoch={epoch + 1} step={global_step}/{target_steps} "
                        f"loss={loss.item() * group_size:.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if global_step >= target_steps:
                    break

            checkpoint = args.output_dir / f"checkpoint-{global_step}"
            if rank == 0 and not args.skip_model_save:
                checkpoint = args.output_dir / f"checkpoint-{global_step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                (checkpoint / "checkpoint_manifest.json").write_text(
                    json.dumps(
                        {
                            "epoch_completed": epoch + 1,
                            "global_step": global_step,
                            "scheduler_total_epochs": scheduler_total_epochs,
                            "resumable": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if distributed:
                dist.barrier()
            if rank == 0 and not args.skip_model_save:
                torch.save(
                    {
                        "epoch_completed": epoch + 1,
                        "global_step": global_step,
                        "world_size": world_size,
                        "steps_per_epoch": steps_per_epoch,
                        "scheduler_total_epochs": scheduler_total_epochs,
                        "training_pairs": len(pairs),
                        "train_data_sha256": train_data_sha256,
                        "max_length": args.max_length,
                        "per_device_batch_size": args.per_device_batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "learning_rate": args.learning_rate,
                        "warmup_ratio": args.warmup_ratio,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    checkpoint / "training_state.pt",
                )
            if not args.skip_model_save:
                torch.save(
                    {
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                    checkpoint / f"rng_state_rank{rank}.pt",
                )
            if distributed:
                dist.barrier()
            if rank == 0 and not args.skip_model_save:
                # Every epoch model is permanent; only the latest checkpoint
                # keeps large process-specific training states.
                for older in args.output_dir.glob("checkpoint-*"):
                    if older != checkpoint:
                        for state_file in [
                            older / "training_state.pt",
                            *older.glob("rng_state_rank*.pt"),
                        ]:
                            if state_file.is_file():
                                state_file.unlink()
                        manifest_path = older / "checkpoint_manifest.json"
                        if manifest_path.is_file():
                            manifest = json.loads(manifest_path.read_text())
                            manifest["resumable"] = False
                            manifest_path.write_text(json.dumps(manifest, indent=2))
            if distributed:
                dist.barrier()
            if global_step >= target_steps:
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
                        "train_data_sha256": train_data_sha256,
                        "training_pairs": len(pairs),
                        "max_length": args.max_length,
                        "epochs_requested": args.epochs,
                        "start_epoch": start_epoch,
                        "scheduler_total_epochs": scheduler_total_epochs,
                        "scheduler_total_steps": scheduler_total_steps,
                        "resume_from_checkpoint": (
                            str(args.resume_from_checkpoint)
                            if args.resume_from_checkpoint is not None else None
                        ),
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
                        "checkpoint_policy": "all_epoch_models_latest_training_state_only",
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

#!/usr/bin/env python3
"""Fine-tune a one-logit cross-encoder with deterministic PyTorch DDP."""

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
import time


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


def _load_pairwise_rows(path: Path) -> list[dict[str, str]]:
    """Load query/positive/negative rows for a pairwise ranking objective."""
    rows: list[dict[str, str]] = []
    expected = {"query", "positive", "negative"}
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
            if any(not isinstance(row[key], str) or not row[key].strip() for key in expected):
                raise ValueError(f"Line {line_number}: every field must be non-empty text")
            rows.append({key: row[key] for key in sorted(expected)})
    if not rows:
        raise ValueError(f"No pairwise training rows found in {path}")
    return rows


def _nonempty_passages(value: object, *, line_number: int, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_number}: {field} must be a non-empty list")
    passages = [str(item) for item in value]
    if any(not item.strip() for item in passages):
        raise ValueError(f"Line {line_number}: {field} contains blank text")
    return passages


def _load_mil_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"query", "passages", "label"}:
                raise ValueError(f"Line {line_number}: invalid MIL fields")
            if not isinstance(row["query"], str) or not row["query"].strip():
                raise ValueError(f"Line {line_number}: query must be non-empty")
            if row["label"] not in (0, 1, 0.0, 1.0):
                raise ValueError(f"Line {line_number}: label must be binary")
            rows.append({"query": row["query"], "passages": _nonempty_passages(
                row["passages"], line_number=line_number, field="passages"
            ), "label": float(row["label"])})
    if not rows or {row["label"] for row in rows} != {0.0, 1.0}:
        raise ValueError("MIL data must contain positive and negative article bags")
    return rows


def _load_pairwise_mil_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = {"query", "positive_passages", "negative_passages"}
            if set(row) != expected:
                raise ValueError(f"Line {line_number}: invalid pairwise MIL fields")
            rows.append({
                "query": str(row["query"]),
                "positive_passages": _nonempty_passages(row["positive_passages"], line_number=line_number, field="positive_passages"),
                "negative_passages": _nonempty_passages(row["negative_passages"], line_number=line_number, field="negative_passages"),
            })
    if not rows:
        raise ValueError(f"No pairwise MIL rows found in {path}")
    return rows


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def _seeded_passage_index(
    *, seed: int, epoch: int, row_index: int, field: str, passage_count: int
) -> int:
    """Select one passage reproducibly without depending on process RNG state."""
    if passage_count <= 0:
        raise ValueError("passage_count must be positive")
    identity = f"{seed}:{epoch}:{row_index}:{field}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % passage_count


def _seeded_cycle_passage_indices(
    *,
    seed: int,
    epoch: int,
    row_index: int,
    field: str,
    passage_count: int,
    max_passages: int,
) -> list[int]:
    """Return one deterministic, non-repeating slice of a shuffled bag cycle."""
    if passage_count <= 0 or max_passages <= 0:
        raise ValueError("passage_count and max_passages must be positive")
    epochs_per_cycle = math.ceil(passage_count / max_passages)
    cycle = epoch // epochs_per_cycle
    offset = (epoch % epochs_per_cycle) * max_passages

    def ordering_key(passage_index: int) -> bytes:
        identity = (
            f"{seed}:{row_index}:{field}:{cycle}:{passage_index}".encode("utf-8")
        )
        return hashlib.sha256(identity).digest()

    permutation = sorted(range(passage_count), key=ordering_key)
    return permutation[offset : offset + max_passages]


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
    parser.add_argument(
        "--objective",
        choices=("pointwise_bce", "pairwise_logistic", "pointwise_mil_bce", "pairwise_mil_logistic"),
        default="pointwise_bce",
    )
    parser.add_argument(
        "--mil-passage-sampling",
        choices=("all", "seeded_random_one", "seeded_cycle"),
        default="all",
        help=(
            "MIL training representation. 'all' max-pools every article passage; "
            "'seeded_random_one' selects one deterministic passage per article, "
            "field, and epoch; 'seeded_cycle' consumes deterministic shuffled "
            "slices without repetition before bag coverage. Inference is unaffected."
        ),
    )
    parser.add_argument(
        "--mil-max-passages-per-epoch",
        type=int,
        default=4,
        help="Maximum article passages selected by seeded_cycle in one epoch.",
    )
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
    if args.mil_max_passages_per_epoch <= 0:
        raise ValueError("--mil-max-passages-per-epoch must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.skip_model_save and args.max_steps <= 0:
        raise ValueError("--skip-model-save requires a positive --max-steps")
    if args.resume_from_checkpoint is not None and args.skip_model_save:
        raise ValueError("Resume is incompatible with --skip-model-save")
    if args.max_steps > 0 and not args.skip_model_save:
        raise ValueError("Saved/resumable training requires complete epochs; use --max-steps only for smoke tests")
    if (
        args.mil_passage_sampling != "all"
        and args.objective not in ("pointwise_mil_bce", "pairwise_mil_logistic")
    ):
        raise ValueError("--mil-passage-sampling applies only to MIL objectives")

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
        dist.init_process_group(backend="nccl", device_id=device)

    def stage(message: str) -> None:
        print(
            f"[cross-encoder-train rank={rank}/{world_size} pid={os.getpid()}] {message}",
            flush=True,
        )

    try:
        stage(
            f"process ready device={device} distributed={distributed} "
            f"objective={args.objective} "
            f"mil_passage_sampling={args.mil_passage_sampling} "
            f"mil_max_passages_per_epoch={args.mil_max_passages_per_epoch}"
        )
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        pairs = None
        train_data_sha256 = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                stage(f"loading training rows from {args.train_data}")
                loaders = {
                    "pointwise_bce": _load_pairs,
                    "pairwise_logistic": _load_pairwise_rows,
                    "pointwise_mil_bce": _load_mil_rows,
                    "pairwise_mil_logistic": _load_pairwise_mil_rows,
                }
                pairs = loaders[args.objective](args.train_data)
                train_data_sha256 = _sha256(args.train_data)
                stage(
                    f"training rows ready rows={len(pairs):,} "
                    f"sha256={train_data_sha256}"
                )
            if distributed:
                dist.barrier()
        if pairs is None or train_data_sha256 is None:
            raise RuntimeError("Training rows were not loaded on this rank")

        class PairDataset(Dataset):
            epoch = 0

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

            def __len__(self) -> int:
                return len(pairs)

            def __getitem__(self, index: int) -> dict[str, object]:
                # Copy so the permanent, checksummed input rows remain immutable.
                return {**pairs[index], "_row_index": index, "_epoch": self.epoch}

        model_source = args.resume_from_checkpoint or args.model_name_or_path
        if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
        transformers.utils.logging.disable_progress_bar()
        tokenizer = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                stage(f"loading tokenizer from {model_source}")
                tokenizer = AutoTokenizer.from_pretrained(model_source)
                stage("tokenizer ready")
            if distributed:
                dist.barrier()
        if tokenizer is None:
            raise RuntimeError("Tokenizer was not materialized on this rank")

        def collate(rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            def selected_passages(row: dict[str, object], field: str) -> list[str]:
                passages = [str(value) for value in row[field]]
                if args.mil_passage_sampling == "all":
                    return passages
                if args.mil_passage_sampling == "seeded_random_one":
                    passage_index = _seeded_passage_index(
                        seed=args.seed,
                        epoch=int(row["_epoch"]),
                        row_index=int(row["_row_index"]),
                        field=field,
                        passage_count=len(passages),
                    )
                    return [passages[passage_index]]
                passage_indices = _seeded_cycle_passage_indices(
                    seed=args.seed,
                    epoch=int(row["_epoch"]),
                    row_index=int(row["_row_index"]),
                    field=field,
                    passage_count=len(passages),
                    max_passages=args.mil_max_passages_per_epoch,
                )
                return [passages[index] for index in passage_indices]

            if args.objective == "pointwise_mil_bce":
                queries, passages, bag_ids = [], [], []
                for bag_id, row in enumerate(rows):
                    for passage in selected_passages(row, "passages"):
                        queries.append(str(row["query"])); passages.append(str(passage)); bag_ids.append(bag_id)
                encoded = tokenizer(queries, passages, padding=True, truncation="only_second", max_length=args.max_length, return_tensors="pt")
                encoded["bag_ids"] = torch.tensor(bag_ids, dtype=torch.long)
                encoded["labels"] = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
                return encoded
            if args.objective == "pairwise_mil_logistic":
                output = {}
                for prefix, field in (("positive", "positive_passages"), ("negative", "negative_passages")):
                    queries, passages, bag_ids = [], [], []
                    for bag_id, row in enumerate(rows):
                        for passage in selected_passages(row, field):
                            queries.append(str(row["query"])); passages.append(str(passage)); bag_ids.append(bag_id)
                    encoded = tokenizer(queries, passages, padding=True, truncation="only_second", max_length=args.max_length, return_tensors="pt")
                    output.update({f"{prefix}_{key}": value for key, value in encoded.items()})
                    output[f"{prefix}_bag_ids"] = torch.tensor(bag_ids, dtype=torch.long)
                return output
            if args.objective == "pointwise_bce":
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
            positive = tokenizer(
                [str(row["query"]) for row in rows],
                [str(row["positive"]) for row in rows],
                padding=True,
                truncation="only_second",
                max_length=args.max_length,
                return_tensors="pt",
            )
            negative = tokenizer(
                [str(row["query"]) for row in rows],
                [str(row["negative"]) for row in rows],
                padding=True,
                truncation="only_second",
                max_length=args.max_length,
                return_tensors="pt",
            )
            return {
                **{f"positive_{key}": value for key, value in positive.items()},
                **{f"negative_{key}": value for key, value in negative.items()},
            }

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

        model = None
        # Kaggle input mounts can stall when both DDP ranks materialize the same
        # large safetensors file concurrently.  Serialize only the read/load
        # phase; each rank still owns its normal independent model replica.
        for loading_rank in range(world_size):
            if rank == loading_rank:
                stage(f"materializing checkpoint on CPU from {model_source}")
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    num_labels=1,
                )
                stage("checkpoint materialized on CPU")
            if distributed:
                dist.barrier()
        if model is None:
            raise RuntimeError("Checkpoint was not materialized on this rank")
        model.config.problem_type = "regression"
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        stage(f"moving checkpoint to {device}")
        model.to(device)
        stage(f"checkpoint ready on {device}")
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
        stage(
            "disk preflight "
            f"free={free_bytes_before_training / 1024**3:.2f}GiB "
            f"checkpoint_estimate={checkpoint_bytes_required / 1024**3:.2f}GiB "
            f"reserve={reserve_bytes / 1024**3:.2f}GiB"
        )
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
            state = None
            rng_state = None
            for loading_rank in range(world_size):
                if rank == loading_rank:
                    stage(f"loading resume state from {state_path}")
                    state = torch.load(
                        state_path, map_location=device, weights_only=False
                    )
                    rng_state = torch.load(
                        rng_path, map_location="cpu", weights_only=False
                    )
                    stage("resume state ready")
                if distributed:
                    dist.barrier()
            if state is None or rng_state is None:
                raise RuntimeError("Resume state was not loaded on this rank")
            expected_state = {
                "objective": args.objective,
                "mil_passage_sampling": args.mil_passage_sampling,
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
            if args.mil_passage_sampling == "seeded_cycle":
                expected_state["mil_max_passages_per_epoch"] = (
                    args.mil_max_passages_per_epoch
                )
            actual_state = {
                **state,
                # Checkpoints created before this option existed used all
                # passages, so they remain exactly resumable in default mode.
                "mil_passage_sampling": state.get("mil_passage_sampling", "all"),
            }
            mismatches = {
                key: (actual_state.get(key), value)
                for key, value in expected_state.items()
                if actual_state.get(key) != value
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
            args.output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = args.output_dir / "training_metrics.jsonl"
            if args.resume_from_checkpoint is None:
                metrics_path.write_text("", encoding="utf-8")
            elif metrics_path.is_file():
                # Resume is epoch-boundary exact. Remove any metrics emitted by
                # a later, interrupted epoch before replaying that epoch.
                retained_metrics = []
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if int(record.get("epoch", 0)) <= start_epoch:
                        retained_metrics.append(line)
                metrics_path.write_text(
                    "".join(f"{line}\n" for line in retained_metrics),
                    encoding="utf-8",
                )

            def write_metric(record: dict[str, object]) -> None:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                        "mil_passage_sampling": args.mil_passage_sampling,
                        "mil_max_passages_per_epoch": args.mil_max_passages_per_epoch,
                    },
                    indent=2,
                ),
                flush=True,
            )
        else:
            metrics_path = None
            write_metric = None

        for epoch in range(start_epoch, int(args.epochs)):
            epoch_started = time.perf_counter()
            group_loss_stats = torch.zeros(2, dtype=torch.float64, device=device)
            rolling_losses: deque[float] = deque(maxlen=50)
            epoch_loss_sum = 0.0
            epoch_observations = 0
            epoch_min_loss = math.inf
            epoch_max_loss = -math.inf
            epoch_successful_steps = 0
            epoch_amp_skipped_steps = 0
            stage(
                f"starting epoch {epoch + 1}/{int(args.epochs)} "
                f"global_step={global_step} batches_per_rank={len(loader)}"
            )
            if sampler is not None:
                sampler.set_epoch(epoch)
            dataset.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                batch_number = batch_index + 1
                trace_batch = batch_number <= 100 or batch_number % 25 == 0
                batch_started = time.perf_counter()
                if trace_batch:
                    tensor_shapes = {
                        key: list(value.shape)
                        for key, value in batch.items()
                        if hasattr(value, "shape")
                    }
                    stage(
                        f"batch_start epoch={epoch + 1} "
                        f"batch={batch_number}/{len(loader)} "
                        f"optimizer_step={global_step} shapes={tensor_shapes}"
                    )
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
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if args.objective == "pointwise_bce":
                            labels = batch.pop("labels")
                            logits = train_model(**batch).logits.view(-1)
                            raw_loss = criterion(logits, labels)
                            observation_count = labels.numel()
                        elif args.objective == "pointwise_mil_bce":
                            labels = batch.pop("labels")
                            bag_ids = batch.pop("bag_ids")
                            chunk_scores = train_model(**batch).logits.view(-1)
                            logits = torch.stack([chunk_scores[bag_ids == index].max() for index in range(labels.numel())])
                            raw_loss = criterion(logits, labels)
                            observation_count = labels.numel()
                        else:
                            positive_bag_ids = batch.pop("positive_bag_ids", None)
                            negative_bag_ids = batch.pop("negative_bag_ids", None)
                            positive_batch = {
                                key.removeprefix("positive_"): value
                                for key, value in batch.items()
                                if key.startswith("positive_")
                            }
                            negative_batch = {
                                key.removeprefix("negative_"): value
                                for key, value in batch.items()
                                if key.startswith("negative_")
                            }
                            positive_scores = train_model(**positive_batch).logits.view(-1)
                            negative_scores = train_model(**negative_batch).logits.view(-1)
                            if args.objective == "pairwise_mil_logistic":
                                bag_count = max(
                                    int(positive_bag_ids.max().item()),
                                    int(negative_bag_ids.max().item()),
                                ) + 1
                                positive_scores = torch.stack([positive_scores[positive_bag_ids == index].max() for index in range(bag_count)])
                                negative_scores = torch.stack([negative_scores[negative_bag_ids == index].max() for index in range(bag_count)])
                            raw_loss = torch.nn.functional.softplus(
                                negative_scores - positive_scores
                            ).mean()
                            observation_count = positive_scores.numel()
                        loss = raw_loss / group_size
                    scaler.scale(loss).backward()
                    group_loss_stats[0] += raw_loss.detach().to(torch.float64) * observation_count
                    group_loss_stats[1] += observation_count

                if trace_batch:
                    stage(
                        f"batch_backward_complete epoch={epoch + 1} "
                        f"batch={batch_number}/{len(loader)} "
                        f"seconds={time.perf_counter() - batch_started:.1f} "
                        f"will_optimizer_step={should_step}"
                    )

                if not should_step:
                    continue
                reduced_loss_stats = group_loss_stats.clone()
                if distributed:
                    dist.all_reduce(reduced_loss_stats, op=dist.ReduceOp.SUM)
                group_loss_sum = float(reduced_loss_stats[0].item())
                group_observations = int(reduced_loss_stats[1].item())
                step_loss = group_loss_sum / group_observations
                group_loss_stats.zero_()
                epoch_loss_sum += group_loss_sum
                epoch_observations += group_observations
                epoch_min_loss = min(epoch_min_loss, step_loss)
                epoch_max_loss = max(epoch_max_loss, step_loss)
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                grad_norm = float(grad_norm_tensor.item())
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = scaler.get_scale() < scale_before_step
                if optimizer_step_skipped:
                    epoch_amp_skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        assert write_metric is not None
                        write_metric(
                            {
                                "record_type": "optimizer_step_skipped",
                                "epoch": epoch + 1,
                                "batch": batch_number,
                                "global_step": global_step,
                                "loss": step_loss,
                                "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                                "amp_scale": scaler.get_scale(),
                            }
                        )
                        print(
                            f"epoch={epoch + 1} optimizer_step_skipped "
                            f"batch={batch_index + 1}/{len(loader)} "
                            f"loss={step_loss:.6f} "
                            f"grad_norm={grad_norm:.3e} "
                            f"amp_scale={scaler.get_scale():.1f}",
                            flush=True,
                        )
                    continue
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_successful_steps += 1
                rolling_losses.append(step_loss)
                rolling_loss_50 = sum(rolling_losses) / len(rolling_losses)
                epoch_avg_loss = epoch_loss_sum / epoch_observations
                if rank == 0:
                    assert write_metric is not None
                    write_metric(
                        {
                            "record_type": "optimizer_step",
                            "epoch": epoch + 1,
                            "batch": batch_number,
                            "global_step": global_step,
                            "loss": step_loss,
                            "rolling_loss_50": rolling_loss_50,
                            "epoch_avg_loss": epoch_avg_loss,
                            "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                            "amp_scale": scaler.get_scale(),
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                if rank == 0 and (
                    global_step == 1 or global_step % args.logging_steps == 0
                ):
                    print(
                        f"epoch={epoch + 1} step={global_step}/{target_steps} "
                        f"loss={step_loss:.6f} "
                        f"rolling_loss_50={rolling_loss_50:.6f} "
                        f"epoch_avg_loss={epoch_avg_loss:.6f} "
                        f"grad_norm={grad_norm:.3e} "
                        f"amp_scale={scaler.get_scale():.1f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if global_step >= target_steps:
                    break

            epoch_seconds = time.perf_counter() - epoch_started
            epoch_avg_loss = epoch_loss_sum / epoch_observations
            if rank == 0:
                assert write_metric is not None
                epoch_summary = {
                    "record_type": "epoch_summary",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_loss": epoch_avg_loss,
                    "min_step_loss": epoch_min_loss,
                    "max_step_loss": epoch_max_loss,
                    "observations": epoch_observations,
                    "successful_optimizer_steps": epoch_successful_steps,
                    "amp_skipped_steps": epoch_amp_skipped_steps,
                    "seconds": epoch_seconds,
                    "observations_per_second": epoch_observations / epoch_seconds,
                }
                write_metric(epoch_summary)
                print(
                    f"epoch_complete={epoch + 1} "
                    f"avg_loss={epoch_avg_loss:.6f} "
                    f"min_step_loss={epoch_min_loss:.6f} "
                    f"max_step_loss={epoch_max_loss:.6f} "
                    f"optimizer_steps={epoch_successful_steps} "
                    f"amp_skipped_steps={epoch_amp_skipped_steps} "
                    f"observations={epoch_observations} "
                    f"seconds={epoch_seconds:.1f} "
                    f"observations_per_second={epoch_observations / epoch_seconds:.2f}",
                    flush=True,
                )

            checkpoint = args.output_dir / f"checkpoint-{global_step}"
            if rank == 0 and not args.skip_model_save:
                checkpoint = args.output_dir / f"checkpoint-{global_step}"
                stage(
                    f"saving epoch {epoch + 1}/{int(args.epochs)} "
                    f"checkpoint to {checkpoint}"
                )
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                (checkpoint / "checkpoint_manifest.json").write_text(
                    json.dumps(
                        {
                            "objective": args.objective,
                            "mil_passage_sampling": args.mil_passage_sampling,
                            "mil_max_passages_per_epoch": args.mil_max_passages_per_epoch,
                            "epoch_completed": epoch + 1,
                            "global_step": global_step,
                            "scheduler_total_epochs": scheduler_total_epochs,
                            "resumable": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                stage(
                    f"checkpoint ready epoch={epoch + 1} "
                    f"global_step={global_step} path={checkpoint}"
                )
            if distributed:
                dist.barrier()
            if rank == 0 and not args.skip_model_save:
                # Remove the previous resume state before writing the new one.
                # Keeping all epoch model snapshots plus two simultaneous Adam
                # states can exceed Kaggle's 20 GiB working-disk quota.
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
            if rank == 0 and not args.skip_model_save:
                torch.save(
                    {
                        "objective": args.objective,
                        "mil_passage_sampling": args.mil_passage_sampling,
                        "mil_max_passages_per_epoch": args.mil_max_passages_per_epoch,
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
                        "objective": args.objective,
                        "mil_passage_sampling": args.mil_passage_sampling,
                        "mil_max_passages_per_epoch": args.mil_max_passages_per_epoch,
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
                        "training_metrics": str(args.output_dir / "training_metrics.jsonl"),
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

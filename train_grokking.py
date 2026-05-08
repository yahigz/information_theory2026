from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from clearml import Task

RUNTIME_DIR = Path.cwd() / ".runtime"

np = None
torch = None
nn = None
F = None
plt = None
ClearMLDataset = None
DataLoader = None
TensorDataset = None


@dataclass
class SplitData:
    inputs: Any
    targets: Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def training_deps_loaded() -> bool:
    return torch is not None


def bootstrap_training_runtime() -> None:
    global np, torch, nn, F, plt, ClearMLDataset, DataLoader, TensorDataset
    if training_deps_loaded():
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
    os.environ.setdefault("CLEARML_CACHE_DIR", str(RUNTIME_DIR / "clearml_cache"))

    import matplotlib.pyplot as _plt
    import numpy as _np
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from clearml import Dataset as _ClearMLDataset
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import TensorDataset as _TensorDataset

    np = _np
    torch = _torch
    nn = _nn
    F = _F
    plt = _plt
    ClearMLDataset = _ClearMLDataset
    DataLoader = _DataLoader
    TensorDataset = _TensorDataset
    # On some platforms (macOS) the default multiprocessing sharing strategy
    # can leak semaphores. Prefer file_system strategy when possible.
    try:
        import torch.multiprocessing as _tmp

        _tmp.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        import multiprocessing as _mp

        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_modular_splits(
    prime: int,
    operation: str,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    train_size: int | None = None,
) -> dict[str, SplitData]:
    xs, ys = torch.meshgrid(torch.arange(prime), torch.arange(prime), indexing="ij")
    inputs = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    if operation == "add":
        targets = (inputs[:, 0] + inputs[:, 1]) % prime
    elif operation == "mul":
        targets = (inputs[:, 0] * inputs[:, 1]) % prime
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    total = inputs.shape[0]
    rng = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(total, generator=rng)

    if train_size is None:
        train_size = int(total * train_fraction)
    val_size = int(total * val_fraction)
    test_size = total - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split sizes must be positive.")

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    return {
        "train": SplitData(inputs[train_idx], targets[train_idx]),
        "val": SplitData(inputs[val_idx], targets[val_idx]),
        "test": SplitData(inputs[test_idx], targets[test_idx]),
        "all": SplitData(inputs, targets),
    }


def _to_tensor(name: str, value: Any) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Missing required array: {name}")
    return torch.as_tensor(value)


def build_splits_from_arrays(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_fraction: float,
    split_seed: int,
    train_fraction: float | None = None,
    train_size: int | None = None,
) -> dict[str, SplitData]:
    inputs = inputs.long()
    targets = targets.long()
    total = inputs.shape[0]
    if total != targets.shape[0]:
        raise ValueError("Inputs and targets must have the same number of rows.")

    rng = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(total, generator=rng)

    if train_size is None:
        if train_fraction is None:
            raise ValueError("Either train_size or train_fraction must be provided.")
        train_size = int(total * train_fraction)
    val_size = int(total * val_fraction)
    test_size = total - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split sizes must be positive.")

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]
    return {
        "train": SplitData(inputs[train_idx], targets[train_idx]),
        "val": SplitData(inputs[val_idx], targets[val_idx]),
        "test": SplitData(inputs[test_idx], targets[test_idx]),
        "all": SplitData(inputs, targets),
    }


def load_npz_splits(npz_path: Path, data_cfg: dict[str, Any]) -> dict[str, SplitData]:
    with np.load(npz_path, allow_pickle=False) as payload:
        keys = set(payload.files)
        explicit_split_keys = {
            "train_inputs",
            "train_targets",
            "val_inputs",
            "val_targets",
            "test_inputs",
            "test_targets",
        }
        if explicit_split_keys.issubset(keys):
            train_inputs = _to_tensor("train_inputs", payload["train_inputs"])
            train_targets = _to_tensor("train_targets", payload["train_targets"])
            val_inputs = _to_tensor("val_inputs", payload["val_inputs"])
            val_targets = _to_tensor("val_targets", payload["val_targets"])
            test_inputs = _to_tensor("test_inputs", payload["test_inputs"])
            test_targets = _to_tensor("test_targets", payload["test_targets"])
            return {
                "train": SplitData(train_inputs, train_targets),
                "val": SplitData(val_inputs, val_targets),
                "test": SplitData(test_inputs, test_targets),
                "all": SplitData(
                    torch.cat([train_inputs, val_inputs, test_inputs], dim=0),
                    torch.cat([train_targets, val_targets, test_targets], dim=0),
                ),
            }

        if {"inputs", "targets"}.issubset(keys):
            return build_splits_from_arrays(
                _to_tensor("inputs", payload["inputs"]),
                _to_tensor("targets", payload["targets"]),
                val_fraction=float(data_cfg["val_fraction"]),
                split_seed=int(data_cfg["split_seed"]),
                train_fraction=data_cfg.get("train_fraction"),
                train_size=data_cfg.get("train_size"),
            )

    raise ValueError(
        "NPZ must contain either explicit split arrays "
        "('train_inputs', 'train_targets', 'val_inputs', 'val_targets', 'test_inputs', 'test_targets') "
        "or generic arrays ('inputs', 'targets')."
    )


def resolve_data_from_clearml(data_cfg: dict[str, Any]) -> Path:
    source_cfg = data_cfg["source"]
    mode = source_cfg["mode"]
    if mode == "local_npz":
        return Path(source_cfg["path"]).expanduser().resolve()
    if mode == "clearml_dataset_npz":
        dataset_id = source_cfg["dataset_id"]
        local_root = Path(ClearMLDataset.get(dataset_id=dataset_id).get_local_copy())
        return (local_root / source_cfg["relative_path"]).resolve()
    if mode == "clearml_task_artifact_npz":
        task_id = source_cfg["task_id"]
        artifact_name = source_cfg["artifact_name"]
        task = Task.get_task(task_id=task_id)
        local_path = task.artifacts[artifact_name].get_local_copy()
        if local_path is None:
            raise ValueError(f"Could not download artifact '{artifact_name}' from task '{task_id}'.")
        return Path(local_path).resolve()
    raise ValueError(f"Unsupported data source mode: {mode}")


def load_splits(data_cfg: dict[str, Any]) -> dict[str, SplitData]:
    source_cfg = data_cfg.get("source", {"mode": "generated"})
    if source_cfg["mode"] == "generated":
        generated_cfg = {k: v for k, v in data_cfg.items() if k != "source"}
        return build_modular_splits(**generated_cfg)
    npz_path = resolve_data_from_clearml(data_cfg)
    return load_npz_splits(npz_path, data_cfg)


bootstrap_training_runtime()


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float, use_layer_norm: bool) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        self.ln1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout(attn_out)
        ff_in = self.ln2(x)
        x = x + self.dropout(self.ff(ff_in))
        return x


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        n_layers: int,
        dropout: float,
        use_layer_norm: bool,
        token_embedding_std: float,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_mult, dropout, use_layer_norm)
                for _ in range(n_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=token_embedding_std)
        nn.init.normal_(self.pos_embed, mean=0.0, std=token_embedding_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.readout(pooled)


def init_weights(module: nn.Module, init_scale: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=init_scale / math.sqrt(module.in_features))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_dataloaders(splits: dict[str, SplitData], batch_size: int, num_workers: int = 0) -> dict[str, DataLoader]:
    """Create DataLoader objects.

    Use `num_workers=0` by default to avoid multiprocessing semaphore leaks
    on some platforms. Set `pin_memory` when CUDA is available.
    """
    pin_memory = False
    try:
        pin_memory = bool(torch.cuda.is_available())
    except Exception:
        pin_memory = False

    return {
        name: DataLoader(
            TensorDataset(split.inputs.long(), split.targets.long()),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        for name, split in splits.items()
        if name in {"train", "val", "test"}
    }


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    logits_norm_acc = 0.0
    all_preds = []
    all_targets = []
    all_inputs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        preds = logits.argmax(dim=-1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_count += batch_size
        logits_norm_acc += logits.norm(dim=-1).mean().item() * batch_size

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_inputs.append(inputs.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    inputs = torch.cat(all_inputs)
    accuracy = total_correct / total_count
    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "logit_norm": logits_norm_acc / total_count,
        "preds": preds,
        "targets": targets,
        "inputs": inputs,
    }


def parameter_l2_norm(model: nn.Module) -> float:
    sq_sum = 0.0
    for param in model.parameters():
        sq_sum += param.detach().float().pow(2).sum().item()
    return math.sqrt(sq_sum)


def embedding_l2_norm(model: SmallTransformer) -> float:
    return model.token_embed.weight.detach().float().norm().item()


def compute_grad_norm(model: nn.Module) -> float:
    sq_sum = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        sq_sum += param.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0


def build_optimizer(model: nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = training_cfg["optimizer"].lower()
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg["weight_decay"])
    betas = tuple(training_cfg.get("betas", [0.9, 0.999]))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = training_cfg.get("scheduler", "none").lower()
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    if name == "none":
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def report_hparams(task: Task, cfg: dict[str, Any], split_summary: dict[str, Any], device: torch.device) -> None:
    payload = dict(cfg)
    payload["runtime"] = {"device": str(device)}
    payload["data_summary"] = split_summary
    task.connect(payload)


def report_text_block(logger: Any, title: str, text: str) -> None:
    logger.report_text(f"{title}\n{text}")


def report_scalar_strict(
    logger: Any,
    title: str,
    series: str,
    iteration: int,
    value: float,
) -> None:
    # Enforce numeric monotonic-style reporting expected by ClearML charts.
    v = float(value)
    if not math.isfinite(v):
        return
    logger.report_scalar(title=title, series=series, iteration=int(iteration), value=v)


def build_split_summary(cfg: dict[str, Any], splits: dict[str, SplitData]) -> dict[str, Any]:
    source_mode = cfg["data"].get("source", {}).get("mode", "generated")
    total = int(splits["all"].targets.numel())
    summary = {
        "source_mode": source_mode,
        "train_size": int(splits["train"].targets.numel()),
        "val_size": int(splits["val"].targets.numel()),
        "test_size": int(splits["test"].targets.numel()),
        "train_fraction_actual": float(splits["train"].targets.numel() / total),
        "val_fraction_actual": float(splits["val"].targets.numel() / total),
        "test_fraction_actual": float(splits["test"].targets.numel() / total),
    }
    if source_mode == "generated":
        summary["prime"] = int(cfg["data"]["prime"])
        summary["operation"] = str(cfg["data"]["operation"])
        summary["total_pairs"] = total
    else:
        summary["num_classes"] = int(splits["all"].targets.max().item() + 1)
        summary["sequence_length"] = int(splits["all"].inputs.shape[1])
        summary["total_examples"] = total
    return summary


def infer_model_vocab_and_classes(cfg: dict[str, Any], splits: dict[str, SplitData]) -> tuple[int, int]:
    if cfg["data"].get("source", {}).get("mode", "generated") == "generated":
        prime = int(cfg["data"]["prime"])
        return prime + 1, prime
    vocab_size = int(splits["all"].inputs.max().item() + 1)
    num_classes = int(splits["all"].targets.max().item() + 1)
    return vocab_size, num_classes


def build_runtime_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "device": str(device),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torch_package_path": str(Path(torch.__file__).resolve()),
        "torch_cuda_build": str(torch.version.cuda),
        "numpy_version": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        summary["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    return summary


def build_dependency_summary() -> dict[str, str]:
    packages = ["torch", "numpy", "matplotlib", "clearml", "PyYAML"]
    resolved: dict[str, str] = {}
    for package in packages:
        try:
            resolved[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def confusion_figure(prime: int, preds: torch.Tensor, targets: torch.Tensor) -> plt.Figure:
    matrix = torch.zeros((prime, prime), dtype=torch.int64)
    for t, p in zip(targets, preds):
        matrix[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def accuracy_scatter_figure(history: dict[str, list[float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(history["train_accuracy"], history["test_accuracy"], s=12, alpha=0.75)
    ax.set_title("Train vs Test Accuracy")
    ax.set_xlabel("Train accuracy")
    ax.set_ylabel("Test accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def training_curves_figure(history: dict[str, list[float]], metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if metric == "loss":
        ax.plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
        ax.plot(history["epoch"], history["val_loss"], label="val", linewidth=2)
        ax.plot(history["epoch"], history["test_loss"], label="test", linewidth=2)
        ax.set_title("Loss curves")
        ax.set_ylabel("loss")
    elif metric == "accuracy":
        ax.plot(history["epoch"], history["train_accuracy"], label="train", linewidth=2)
        ax.plot(history["epoch"], history["val_accuracy"], label="val", linewidth=2)
        ax.plot(history["epoch"], history["test_accuracy"], label="test", linewidth=2)
        ax.set_title("Accuracy curves")
        ax.set_ylabel("accuracy")
    else:
        raise ValueError(f"Unsupported metric for curve plot: {metric}")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def history_overview_figure(history: dict[str, list[float]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_acc = axes

    ax_loss.plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    ax_loss.plot(history["epoch"], history["val_loss"], label="val", linewidth=2)
    ax_loss.plot(history["epoch"], history["test_loss"], label="test", linewidth=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(history["epoch"], history["train_accuracy"], label="train", linewidth=2)
    ax_acc.plot(history["epoch"], history["val_accuracy"], label="val", linewidth=2)
    ax_acc.plot(history["epoch"], history["test_accuracy"], label="test", linewidth=2)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.suptitle("Training history overview")
    fig.tight_layout()
    return fig


def log_predictions_table(
    logger: Any,
    title: str,
    iteration: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    max_rows: int,
) -> None:
    rows = [["x", "y", "target", "prediction", "correct"]]
    limit = min(max_rows, len(targets))
    for i in range(limit):
        rows.append(
            [
                int(inputs[i, 0]),
                int(inputs[i, 1]),
                int(targets[i]),
                int(preds[i]),
                bool(int(targets[i]) == int(preds[i])),
            ]
        )
    logger.report_table(title=title, series="examples", iteration=iteration, table_plot=rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    task = Task.init(
        project_name=cfg["experiment"]["project_name"],
        task_name=cfg["experiment"]["task_name"],
        tags=cfg["experiment"].get("tags", []),
        auto_connect_frameworks=False,
    )
    logger = task.get_logger()
    remote_cfg = cfg.get("clearml", {})
    if remote_cfg.get("queue"):
        task.execute_remotely(queue_name=remote_cfg["queue"], exit_process=True)

    bootstrap_training_runtime()
    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(cfg["training"]["device"])
    splits = load_splits(cfg["data"])
    split_summary = build_split_summary(cfg, splits)
    num_workers = int(cfg["training"].get("num_workers", 0))
    loaders = make_dataloaders(splits, batch_size=int(cfg["training"]["batch_size"]), num_workers=num_workers)
    vocab_size, num_classes = infer_model_vocab_and_classes(cfg, splits)

    report_hparams(task, cfg, split_summary, device)
    report_text_block(logger, "Split summary", yaml.safe_dump(split_summary, sort_keys=False))
    report_text_block(logger, "Config", yaml.safe_dump(cfg, sort_keys=False))
    report_text_block(logger, "Runtime summary", yaml.safe_dump(build_runtime_summary(device), sort_keys=False))
    report_text_block(logger, "Dependency summary", yaml.safe_dump(build_dependency_summary(), sort_keys=False))

    model = SmallTransformer(vocab_size=vocab_size, num_classes=num_classes, **cfg["model"]).to(device)
    model.apply(lambda module: init_weights(module, float(cfg["training"]["init_scale"])))

    optimizer = build_optimizer(model, cfg["training"])
    total_steps = int(cfg["training"]["epochs"]) * len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg["training"], total_steps=total_steps)

    history: dict[str, list[float]] = {
        "train_accuracy": [],
        "test_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "epoch": [],
    }
    best_test_accuracy = -1.0
    global_step = 0

    # directory to save generated plots for upload
    plots_dir = Path(cfg.get("logging", {}).get("save_dir", "artifacts"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    PRINT_EVERY = int(cfg.get("logging", {}).get("print_interval", 50))

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        last_grad_norm = 0.0

        batch_idx = 0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=float(cfg["training"]["label_smoothing"]),
            )
            loss.backward()
            last_grad_norm = compute_grad_norm(model)
            clip_norm = float(cfg["training"]["grad_clip_norm"])
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # per-iteration logging (train loss every step)
            global_step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            report_scalar_strict(logger, "loss_step", "train", global_step, loss.item())
            report_scalar_strict(logger, "optimization_step", "lr", global_step, current_lr)

            if (global_step % PRINT_EVERY) == 0:
                print(f"Epoch {epoch} step {global_step} batch {batch_idx} loss={loss.item():.4f} lr={current_lr:.6e}", flush=True)

            batch_idx += 1
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=-1) == targets).sum().item()
            running_count += batch_size

        train_loss_epoch = running_loss / running_count
        train_acc_epoch = running_correct / running_count
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        report_scalar_strict(logger, "loss_epoch", "train", epoch, train_loss_epoch)
        report_scalar_strict(logger, "accuracy_epoch", "train", epoch, train_acc_epoch)
        report_scalar_strict(logger, "optimization_epoch", "lr", epoch, current_lr)
        report_scalar_strict(logger, "optimization_epoch", "grad_norm", epoch, last_grad_norm)
        report_scalar_strict(logger, "optimization_epoch", "parameter_norm", epoch, parameter_l2_norm(model))
        report_scalar_strict(logger, "optimization_epoch", "embedding_norm", epoch, embedding_l2_norm(model))
        report_scalar_strict(logger, "runtime_epoch", "seconds", epoch, epoch_time)

        if epoch % int(cfg["logging"]["train_eval_interval"]) == 0 or epoch == 1:
            train_eval = evaluate(
                model,
                loaders["train"],
                device,
                label_smoothing=float(cfg["training"]["label_smoothing"]),
            )
            report_scalar_strict(logger, "loss_epoch", "train_eval", epoch, train_eval["loss"])
            report_scalar_strict(logger, "accuracy_epoch", "train_eval", epoch, train_eval["accuracy"])
            report_scalar_strict(logger, "error_rate_epoch", "train_eval", epoch, train_eval["error_rate"])
            report_scalar_strict(logger, "logit_norm_epoch", "train_eval", epoch, train_eval["logit_norm"])

        if epoch % int(cfg["logging"]["full_eval_interval"]) == 0 or epoch == 1:
            val_metrics = evaluate(
                model,
                loaders["val"],
                device,
                label_smoothing=float(cfg["training"]["label_smoothing"]),
            )
            test_metrics = evaluate(
                model,
                loaders["test"],
                device,
                label_smoothing=float(cfg["training"]["label_smoothing"]),
            )

            for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
                report_scalar_strict(logger, "loss_epoch", split_name, epoch, metrics["loss"])
                report_scalar_strict(logger, "accuracy_epoch", split_name, epoch, metrics["accuracy"])
                report_scalar_strict(logger, "error_rate_epoch", split_name, epoch, metrics["error_rate"])
                report_scalar_strict(logger, "logit_norm_epoch", split_name, epoch, metrics["logit_norm"])

            gap = train_acc_epoch - test_metrics["accuracy"]
            report_scalar_strict(logger, "generalization_epoch", "train_minus_test_accuracy", epoch, gap)
            report_scalar_strict(
                logger,
                "generalization_epoch",
                "train_minus_test_loss",
                epoch,
                train_loss_epoch - test_metrics["loss"],
            )

            history["epoch"].append(epoch)
            history["train_accuracy"].append(train_acc_epoch)
            history["train_loss"].append(train_loss_epoch)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["test_loss"].append(test_metrics["loss"])
            history["test_accuracy"].append(test_metrics["accuracy"])

            if epoch % int(cfg["logging"]["plot_interval"]) == 0 or epoch == 1:
                fig = training_curves_figure(history, metric="loss")
                logger.report_matplotlib_figure(
                    title="training_curves",
                    series="loss",
                    iteration=epoch,
                    figure=fig,
                )
                # save and upload PNG for easy download from ClearML UI
                try:
                    p = plots_dir / f"training_curves_loss_epoch_{epoch:06d}.png"
                    fig.savefig(p, bbox_inches="tight")
                    task.upload_artifact(f"plots/training_curves_loss_epoch_{epoch}", artifact_object=str(p))
                except Exception:
                    pass
                plt.close(fig)

                fig = training_curves_figure(history, metric="accuracy")
                logger.report_matplotlib_figure(
                    title="training_curves",
                    series="accuracy",
                    iteration=epoch,
                    figure=fig,
                )
                try:
                    p = plots_dir / f"training_curves_accuracy_epoch_{epoch:06d}.png"
                    fig.savefig(p, bbox_inches="tight")
                    task.upload_artifact(f"plots/training_curves_accuracy_epoch_{epoch}", artifact_object=str(p))
                except Exception:
                    pass
                plt.close(fig)

                fig = confusion_figure(num_classes, test_metrics["preds"], test_metrics["targets"])
                logger.report_matplotlib_figure(
                    title="confusion_matrix",
                    series="test",
                    iteration=epoch,
                    figure=fig,
                )
                try:
                    p = plots_dir / f"confusion_matrix_epoch_{epoch:06d}.png"
                    fig.savefig(p, bbox_inches="tight")
                    task.upload_artifact(f"plots/confusion_matrix_epoch_{epoch}", artifact_object=str(p))
                except Exception:
                    pass
                plt.close(fig)

                if len(history["epoch"]) > 1:
                    fig = accuracy_scatter_figure(history)
                    logger.report_matplotlib_figure(
                        title="accuracy_scatter",
                        series="train_vs_test",
                        iteration=epoch,
                        figure=fig,
                    )
                    try:
                        p = plots_dir / f"accuracy_scatter_epoch_{epoch:06d}.png"
                        fig.savefig(p, bbox_inches="tight")
                        task.upload_artifact(f"plots/accuracy_scatter_epoch_{epoch}", artifact_object=str(p))
                    except Exception:
                        pass
                    plt.close(fig)

            log_predictions_table(
                logger,
                title="sample_predictions",
                iteration=epoch,
                inputs=test_metrics["inputs"],
                targets=test_metrics["targets"],
                preds=test_metrics["preds"],
                max_rows=int(cfg["logging"]["sample_predictions"]),
            )

            if test_metrics["accuracy"] > best_test_accuracy:
                best_test_accuracy = test_metrics["accuracy"]

        if epoch % int(cfg["logging"]["histogram_interval"]) == 0 or epoch == 1:
            for name, param in model.named_parameters():
                logger.report_histogram(
                    title="weights",
                    series=name,
                    iteration=epoch,
                    values=param.detach().float().cpu().view(-1).numpy(),
                    xaxis="value",
                    yaxis="count",
                )

        # checkpoint saving disabled by user request (only logging/plots are kept)

    final_test = evaluate(
        model,
        loaders["test"],
        device,
        label_smoothing=float(cfg["training"]["label_smoothing"]),
    )
    final_summary = {
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": final_test["accuracy"],
        "final_test_loss": final_test["loss"],
        "final_parameter_norm": parameter_l2_norm(model),
    }
    report_text_block(logger, "Final summary", yaml.safe_dump(final_summary, sort_keys=False))
    task.upload_artifact("final_summary", artifact_object=final_summary)

    if history["epoch"]:
        final_history_fig = history_overview_figure(history)
        try:
            logger.report_matplotlib_figure(
                title="training_curves",
                series="history_overview_final",
                iteration=history["epoch"][-1],
                figure=final_history_fig,
            )
        except Exception:
            pass
        try:
            final_history_path = plots_dir / "history_overview_final.png"
            final_history_fig.savefig(final_history_path, bbox_inches="tight")
            task.upload_artifact("plots/history_overview_final", artifact_object=str(final_history_path))
        except Exception:
            pass
        plt.close(final_history_fig)


if __name__ == "__main__":
    main()

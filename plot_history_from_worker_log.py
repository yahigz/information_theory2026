from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml


HISTORY_BLOCK_BEGIN = "=== HISTORY_BLOCK_BEGIN ==="
HISTORY_BLOCK_END = "=== HISTORY_BLOCK_END ==="


def extract_history_payloads(log_text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        re.escape(HISTORY_BLOCK_BEGIN) + r"\s*(.*?)\s*" + re.escape(HISTORY_BLOCK_END),
        re.DOTALL,
    )
    payloads: list[dict[str, Any]] = []
    for match in pattern.finditer(log_text):
        block_text = match.group(1).strip()
        if not block_text:
            continue
        payload = yaml.safe_load(block_text)
        if isinstance(payload, dict) and "history" in payload:
            payloads.append(payload)
    return payloads


def pick_latest_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError(f"No history blocks found between {HISTORY_BLOCK_BEGIN!r} and {HISTORY_BLOCK_END!r}")
    return max(payloads, key=lambda payload: len(payload.get("history", {}).get("epoch", [])))


def finite_series(history: dict[str, list[float]], key: str) -> tuple[list[int], list[float]]:
    epochs = history.get("epoch", [])
    values = history.get(key, [])
    points = [(int(epoch), float(value)) for epoch, value in zip(epochs, values) if value is not None]
    return [epoch for epoch, _ in points], [value for _, value in points]


def plot_loss(history: dict[str, list[float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")]:
        epochs, values = finite_series(history, key)
        if epochs:
            ax.plot(epochs, values, label=label, linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(history: dict[str, list[float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [("train_accuracy", "train"), ("val_accuracy", "val"), ("test_accuracy", "test")]:
        epochs, values = finite_series(history, key)
        if epochs:
            ax.plot(epochs, values, label=label, linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overview(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_acc = axes
    for key, label in [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")]:
        epochs, values = finite_series(history, key)
        if epochs:
            ax_loss.plot(epochs, values, label=label, linewidth=2)
    for key, label in [("train_accuracy", "train"), ("val_accuracy", "val"), ("test_accuracy", "test")]:
        epochs, values = finite_series(history, key)
        if epochs:
            ax_acc.plot(epochs, values, label=label, linewidth=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    fig.suptitle("Training history overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a downloaded ClearML worker log.")
    parser.add_argument("log_file", type=Path, help="Path to the downloaded worker log text file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated PNG plots")
    args = parser.parse_args()

    log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
    payloads = extract_history_payloads(log_text)
    payload = pick_latest_payload(payloads)
    history = payload["history"]

    output_dir = args.output_dir or args.log_file.with_suffix("").with_name(args.log_file.stem + "_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(history, output_dir / "history_loss.png")
    plot_accuracy(history, output_dir / "history_accuracy.png")
    plot_overview(history, output_dir / "history_overview.png")

    extracted_history = output_dir / "history_extracted.yaml"
    extracted_history.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
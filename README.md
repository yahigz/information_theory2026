# Grokking on Modular Arithmetic

Минимальный воспроизводимый эксперимент для эффекта grokking на задаче
`x + y mod p` со small transformer и полным логированием в `ClearML`.

## Что внутри

- датасет всех пар `(x, y)` для простого `p`
- разбиение на маленький train и отдельные val/test
- small transformer без `LayerNorm`
- длинное обучение с `weight decay`
- логирование метрик, норм, зазоров обобщения, таблиц и графиков в `ClearML`

## Быстрый старт

```bash
uv venv
source .venv/bin/activate
uv sync
clearml-init
uv run train_grokking.py --config configs/grokking_mod_prime_113.yaml
```

## Запуск через очередь ClearML

В конфиге можно указать:

```yaml
clearml:
  queue: "gpu"
```

Тогда локальный запуск только создаст задачу и отправит ее в очередь, а воркер подхватит обучение сам:

```bash
./run_experiment.sh configs/grokking_mod_prime_113.yaml
```

Если `queue: null`, обучение идет сразу в текущем процессе.

Важно:
- при `queue != null` локальный запуск не должен тянуть `torch/numpy/matplotlib`
- для локальной отправки задачи достаточно Python с `clearml` и `pyyaml`
- полноценные training-зависимости ставятся уже на воркере

## Как прокинуть свои данные из ClearML

Скрипт поддерживает три режима в `data.source.mode`:

- `generated`: текущая синтетика `x + y mod p`
- `clearml_dataset_npz`: взять `.npz` из `ClearML Dataset`
- `clearml_task_artifact_npz`: взять `.npz`-артефакт из другой задачи

Пример для `ClearML Dataset` есть в `configs/clearml_dataset_example.yaml`.

Ожидаемый формат `.npz`:

1. Либо уже готовые сплиты:
   `train_inputs`, `train_targets`, `val_inputs`, `val_targets`, `test_inputs`, `test_targets`
2. Либо общий массив:
   `inputs`, `targets`

Ожидаемые формы:

- `*_inputs`: `(N, 2)` или в общем случае `(N, L)` из целочисленных токенов
- `*_targets`: `(N,)` из целочисленных классов `0..C-1`

Если вы даете только `inputs` и `targets`, скрипт сам разобьет данные по `train_fraction`, `val_fraction`, `split_seed`.

Пример загрузки из артефакта другой задачи:

```yaml
data:
  source:
    mode: "clearml_task_artifact_npz"
    task_id: "YOUR_TASK_ID"
    artifact_name: "dataset_npz"
  train_fraction: 0.3
  train_size: null
  val_fraction: 0.1
  split_seed: 17
```

## Что логируется в ClearML

- вместо большого числа scalars/plots в ClearML скрипт печатает один YAML-блок `history` только в самом конце обучения
- блок обрамлен маркерами `=== HISTORY_BLOCK_BEGIN ===` / `=== HISTORY_BLOCK_END ===`
- внутри блока лежит весь накопленный `history` и финальный `summary`

## Как строить графики из worker log

Скачайте worker log из ClearML и запустите:

```bash
python plot_history_from_worker_log.py /path/to/worker.log
```

Скрипт создаст папку рядом с логом и сохранит:

- `history_loss.png`
- `history_accuracy.png`
- `history_overview.png`
- `history_extracted.yaml`

## Рекомендации для гроккинга

- если обобщение приходит слишком рано, уменьшите `train_fraction` до `0.2-0.25`
- если модель не выходит из режима запоминания, увеличьте `epochs` до `10000+`
- если обучение нестабильно, уменьшите `lr` до `3e-4`
- если хотите усилить эффект, попробуйте `weight_decay` в диапазоне `0.1-1.0`

## Замечание

Тема 17 из PDF не была автоматически извлечена локальными системными утилитами в этой среде.
Я собрал эксперимент под ваш текущий фокус: grokking, modular arithmetic, small transformer,
длинное обучение и инструментирование через `ClearML`.

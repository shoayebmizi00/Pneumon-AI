from __future__ import annotations

import json
from pathlib import Path

from training.train import save_training_curves
from training.utils import write_json


LOGS = ("training_run.log", "training_run_resume.log", "training_compare.log", "training_finetune.log")


def main() -> None:
    histories: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for log_name in LOGS:
        path = Path(log_name)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not {"model", "stage", "epoch"} <= row.keys():
                continue
            key = (row["model"], row["stage"], row["epoch"])
            if key in seen:
                continue
            seen.add(key)
            histories.setdefault(row["model"], []).append(row)
    for rows in histories.values():
        rows.sort(key=lambda row: (0 if row["stage"] == "frozen" else 1, row["epoch"]))
    write_json("artifacts/training_history.json", histories)
    runs = [{"model": model, "history": rows} for model, rows in histories.items()]
    save_training_curves(runs, Path("results"))
    print({model: len(rows) for model, rows in histories.items()})


if __name__ == "__main__":
    main()

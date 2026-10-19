"""Create a run directory from a config template and a row file.

usage: python -m attrib.make_run --config configs/opsd.json --rows data/built/train.json --run-dir runs/opsd_e4b \
           --model google/gemma-4-E4B-it --data-root whowhen-pro [--set lr_peak=5e-6 lr_end=5e-7] [--subset 552]

Writes <run-dir>/config.json (with total_updates filled in) and <run-dir>/train.json in the exact order the trainer
visits it. OPSD and GRPO walk the file in order, so the epochs are laid out here: epoch 1 = the given order,
epoch e > 1 = the same rows shuffled with seed e. SFT shuffles per epoch inside its trainer, so its file holds one copy.
--subset N keeps a fixed random subset (seed 20261018) before laying out epochs (compute-matched GRPO uses 552).
"""
import argparse
import json
import math
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True, help="Hugging Face id or local path of the student (OPSD teacher = same model)")
    ap.add_argument("--data-root", required=True, help="the whowhen-pro folder (the trainers score rollouts with the official scorer)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (JSON values)")
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    cfg["student"] = cfg["teacher"] = a.model
    cfg["data_root"] = str(Path(a.data_root).resolve())
    rows = json.load(open(a.rows))
    if a.subset:
        rows = random.Random(20261018).sample(rows, a.subset)
    epochs = cfg.get("epochs", 1)
    if cfg["method"] == "sft":
        laid = rows
        cfg["total_updates"] = math.ceil(epochs * len(rows) / cfg["batch"])
    else:
        laid = list(rows)
        for e in range(2, epochs + 1):
            r = list(rows)
            random.Random(e).shuffle(r)
            laid += r
        cfg["total_updates"] = math.ceil(len(laid) / cfg["batch"])
    if cfg.get("export_epochs"):
        cfg["export_steps"] = [round(cfg["total_updates"] * e / epochs) for e in cfg["export_epochs"] if e < epochs]
    cfg.setdefault("warmup", max(5, round(cfg.get("warmup_frac", 0.0) * cfg["total_updates"])))
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "train.json").write_text(json.dumps(laid))
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    print(run, cfg["method"], "rows", len(laid), "updates", cfg["total_updates"], "exports", cfg.get("export_steps"))


if __name__ == "__main__":
    main()

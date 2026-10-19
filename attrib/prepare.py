"""Build training/eval rows from the WHO&WHEN PRO text release.

usage: python -m attrib.prepare --data-root whowhen-pro --splits data/splits --out data/built

Each row keeps the official all-at-once prompt (as content parts), the gold label, and the privileged notes a
teacher can be given:
  gold_step       the true decisive step and agent
  gold_step_mode  the true step, agent and error mode   (the OPSD teacher note)
  omission        a generic note naming the omission/process modes (prompting diagnostic only)

Splits come from data/splits/*_ids.txt (exactly the lists we use). To rebuild them instead, pass --rebuild:
frameworks magentic-one, metagpt and dylan -> ood_test; 15% of the remaining traces (seed 20261018) -> id_test;
the rest -> train, dropping traces whose prompt is longer than --max-prompt-tokens.
"""
import argparse
import json
import random
from pathlib import Path

from attrib import common  # noqa: F401  (stubs litellm before the harness import)
from whowhen_eval.prompts import load_taxonomy
from whowhen_eval.run import build_prompt, iter_trace_refs, load_release

OOD = {"magentic-one", "metagpt", "dylan"}
OMISSION = ("Note on error modes: many failures are errors of omission or process rather than a wrong statement. Before choosing the error mode, "
            "ask whether the decisive failure is something the agent did NOT do, or a flawed process, and consider these modes explicitly: "
            "A.3 Premature termination (stopped before the task objectives were fully met), V.2 Inadequate or incorrect verification, "
            "V.1 Context/memory loss, C.2 Communication failure, C.3 Over-reliance on other agents, PL.1 Ineffective planning. "
            "Choose one of them when it describes the cause better than a reasoning label (R.*).")


def gold_coord(fw, gt):
    """The (step coordinate, agent) a fully correct answer gives, following whowhen_eval.score's per-framework rules."""
    if fw == "mathchat":
        step = str(2 * int(gt["round"]) + int(gt["position"]))
    elif fw == "metagpt":
        step = str(gt["stage"])
    elif fw in ("debate", "dylan"):
        step = str(gt["round"])
    elif gt.get("step") is not None:
        step = str(gt["step"])
    elif gt.get("round") is not None:
        step = f"{gt['round']}.{gt.get('position', 0)}"
    else:
        step = "unknown"
    agent = "agent" if fw in ("smolagents", "alfagent") else (gt.get("agent") or ", ".join(gt.get("agents") or []))
    return step, agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="the whowhen-pro data folder (taxonomy.yaml, data/text.jsonl)")
    ap.add_argument("--splits", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tokenizer", default=None, help="with --rebuild: tokenizer used for the prompt-length filter")
    ap.add_argument("--max-prompt-tokens", type=int, default=16000)
    a = ap.parse_args()
    tax = load_taxonomy(a.data_root)
    rows = []
    for ref in iter_trace_refs(a.data_root, benchmark="all", modality="text"):
        rel = load_release(a.data_root, ref)
        gt = rel["ground_truth"]
        parts = build_prompt(rel, ref.framework, tax)
        step, agent = gold_coord(ref.framework, gt)
        note = f"\n\nFor reference, the decisive error occurs at step {step} by agent {agent}."
        rows.append(dict(id=ref.trace_id, framework=ref.framework, benchmark=ref.benchmark, gt=gt, parts=parts, hints=dict(
            gold_step=note + " Use this step and agent in your answer and focus on identifying the error mode.",
            gold_step_mode=note + f" Its error mode is {gt.get('mode')}. Use this step, agent and error mode in your answer and explain the error.",
            omission="\n\n" + OMISSION)))
    by_id = {r["id"]: r for r in rows}
    a.out.mkdir(parents=True, exist_ok=True)
    if a.rebuild:
        rng = random.Random(20261018)
        split = {r["id"]: "ood_test" if r["framework"] in OOD else ("id_test" if rng.random() < 0.15 else "train") for r in rows}
        ids = {s: [i for i in by_id if split[i] == s] for s in ("train", "id_test", "ood_test")}
        if a.tokenizer:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(a.tokenizer)
            from attrib.common import prompt_ids
            ids["train"] = [i for i in ids["train"] if len(prompt_ids(proc, by_id[i])) <= a.max_prompt_tokens]
        rng.shuffle(ids["train"])
    else:
        ids = {s: (a.splits / f"{s}_ids.txt").read_text().split() for s in ("train", "id_test", "ood_test")}
    for s, lst in ids.items():
        missing = [i for i in lst if i not in by_id]
        assert not missing, f"{s}: {len(missing)} ids not in the release (e.g. {missing[:3]})"
        (a.out / f"{s}.json").write_text(json.dumps([by_id[i] for i in lst]))
        (a.out / f"{s}_ids.txt").write_text("\n".join(lst) + "\n")
        print(s, len(lst), "frameworks", sorted({by_id[i]["framework"] for i in lst}))


if __name__ == "__main__":
    main()

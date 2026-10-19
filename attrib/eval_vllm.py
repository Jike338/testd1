"""WHO&WHEN PRO all-at-once evaluation against an OpenAI-compatible endpoint (e.g. a local vLLM server), using the
official harness for the prompt, parsing and scoring. Output records match whowhen_eval.run, so
`python -m whowhen_eval.leaderboard --results <out>` also works. Resumable (skips trace ids already on disk).

usage: python -m attrib.eval_vllm --model <served name> --base http://127.0.0.1:8000/v1 --data-root whowhen-pro \
           --out results/<tag> --ids data/splits/eval_ids.txt [--think 0] [--hint none|omission|gold_step|gold_step_mode]

Defaults match our evaluation: thinking off, greedy (temperature 0), up to 2048 new tokens.
--hint appends one note to the prompt: gold_step_mode = the privileged-teacher view (used to build SFT-RS targets),
gold_step / omission = diagnostics.
"""
import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from attrib import common  # noqa: F401  (stubs litellm)
from attrib.prepare import OMISSION, gold_coord
from openai import AsyncOpenAI
from whowhen_eval.parse import parse_all_at_once
from whowhen_eval.prompts import load_taxonomy, user_msg
from whowhen_eval.run import build_prompt, iter_trace_refs, load_release
from whowhen_eval.score import score as score_prediction
from whowhen_eval.store import ResultsStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", default=None, help="file of trace ids to evaluate (one per line)")
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--hint", default="none", choices=["none", "omission", "gold_step", "gold_step_mode"])
    a = ap.parse_args()
    tax = load_taxonomy(a.data_root)
    client = AsyncOpenAI(base_url=a.base, api_key="local", timeout=1800, max_retries=3)
    sem = asyncio.Semaphore(a.concurrency)
    tag = f"{a.model}_think{a.think}" + ("" if a.hint == "none" else "_" + a.hint) + (f"_t{a.temperature}_s{a.seed}" if a.seed is not None else "")

    def hint_text(gt, fw):
        if a.hint == "omission":
            return OMISSION
        step, agent = gold_coord(fw, gt)
        note = f"For reference, the decisive error occurs at step {step} by agent {agent}."
        if a.hint == "gold_step_mode":
            return note + f" Its error mode is {gt.get('mode')}. Use this step, agent and error mode in your answer and explain the error."
        return note + " Use this step and agent in your answer and focus on identifying the error mode."

    async def one(ref, store):
        base = dict(protocol="all_at_once", model=tag, trace_id=ref.trace_id, framework=ref.framework, benchmark=ref.benchmark, modality=ref.split,
                    error_mode=None, trace_path=f"data/{ref.split}.jsonl", ground_truth=None, timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        rel = load_release(a.data_root, ref)
        gt = rel.get("ground_truth")
        base["ground_truth"] = gt
        base["error_mode"] = gt.get("mode") if isinstance(gt, dict) else None
        parts = build_prompt(rel, ref.framework, tax) + ([dict(type="text", text="\n\n" + hint_text(gt, ref.framework))] if a.hint != "none" else [])
        t0, raw, err, usage = time.monotonic(), None, None, {}
        try:
            extra = {"chat_template_kwargs": {"enable_thinking": bool(a.think)}}
            if a.seed is not None:
                extra["seed"] = a.seed
            async with sem:
                r = await client.chat.completions.create(model=a.model, messages=[user_msg(parts)], temperature=a.temperature,
                                                         max_tokens=a.max_tokens, extra_body=extra)
            raw = r.choices[0].message.content or ""
            usage = dict(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens, finish=r.choices[0].finish_reason)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        pred = None
        if raw is not None:
            p = parse_all_at_once(raw, tax.codes)
            pred = dict(agent_name=p.agent_name, step_coord=p.step_coord, error_mode=p.error_mode, reason=p.reason, parse_warnings=p.parse_warnings)
        rec = dict(base, raw_output=raw, prediction=pred, score=score_prediction(pred, gt, ref.framework), usage=usage,
                   duration_s=round(time.monotonic() - t0, 3), error=err)
        store.append(rec)
        return rec

    async def run():
        keep = set(open(a.ids).read().split()) if a.ids else None
        jobs = []
        by = {}
        for ref in iter_trace_refs(a.data_root, benchmark="all", modality="text"):
            if keep is None or ref.trace_id in keep:
                by.setdefault(ref.benchmark, []).append(ref)
        for bench, items in by.items():
            store = ResultsStore.for_cell(a.out, tag, bench)
            done = store.done_trace_ids()
            jobs += [one(r, store) for r in items if r.trace_id not in done]
        print(f"{len(jobs)} pending", file=sys.stderr, flush=True)
        n = fail = 0
        t0 = time.monotonic()
        for fut in asyncio.as_completed(jobs):
            rec = await fut
            n += 1
            fail += bool(rec.get("error"))
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} failed={fail} {n / (time.monotonic() - t0):.2f}/s", file=sys.stderr, flush=True)
        print(f"DONE total={n} failed={fail}", file=sys.stderr, flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()

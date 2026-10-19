"""GRPO baseline.

usage: python -m attrib.train_grpo <run_dir>

Each update: an in-process vLLM engine (holding the current weights) samples G answers per trace
(temperature 1, full distribution); reward = mean of agent/step/mode correctness under the official scorer
(0 if the answer cannot be parsed), i.e. the multi-granular reward of AgenTracer/AgentForesight extended with the
error mode; advantages are normalised within each group; groups whose G rewards are all equal carry no signal and are
skipped; one on-policy policy-gradient step per batch (no reference-KL term), token-mean loss.
Constant lr after warmup. Data-matched: all traces x G samples. Compute-matched: a 552-trace subset, 4 traces x 8
samples per update, so each update trains on 32 sampled answers like OPSD.
"""
import json
import sys
import time
import traceback

from attrib.common import (Optimizer, budget_groups, end_tokens, export, head, inputs, load_model, load_processor, load_run, make_vllm, prompt_ids,
                           save_json, score_text, vllm_wake_with, zero_grad)


def main(run_dir):
    root, cfg, rows = load_run(run_dir)
    llm = make_vllm(cfg["student"], cfg)
    import torch
    import torch.nn.functional as F
    from vllm import SamplingParams
    torch.manual_seed(cfg["seed"])
    proc = load_processor(cfg["student"])
    tok = proc.tokenizer
    gpu_opt = cfg.get("optimizer", "gpu") == "gpu"
    m = load_model(cfg["student"], train=True, fp32_trainable=gpu_opt)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = Optimizer(params, cfg["lr_peak"], mode="gpu" if gpu_opt else "cpu", threads=cfg.get("cpu_threads", 14))
    G = cfg["group"]
    sp = SamplingParams(n=G, temperature=cfg.get("temperature", 1.0), top_p=cfg.get("top_p", 1.0), top_k=cfg.get("top_k", 0) or -1,
                        max_tokens=cfg["max_new_tokens"], seed=cfg["seed"], logprobs=0)
    eos = end_tokens(m, tok)
    B, budget, rnd = cfg["batch"], cfg.get("token_budget", 16384), cfg.get("round_to", 512)
    start = time.time()
    for step, lo in enumerate(range(0, len(rows), B), 1):
        batch = rows[lo:lo + B]
        T = {}
        for r in batch:
            if "pids" not in r:
                r["pids"] = prompt_ids(proc, r)
        t = time.time()
        vllm_wake_with(llm, m)
        T["sync"] = time.time() - t
        t = time.time()
        res = llm.generate([{"prompt_token_ids": r["pids"]} for r in batch], sp, use_tqdm=False)
        T["rollout"] = time.time() - t
        llm.sleep(level=2)
        outs = []
        for r, o in zip(batch, res):
            for c in o.outputs:
                toks = list(c.token_ids) or [tok.eos_token_id]
                text = tok.decode(toks, skip_special_tokens=True)
                pred, sc, axes = score_text(r, text, cfg["data_root"])
                outs.append(dict(row=r, tokens=toks, score=sc, axes=axes, joint=all(axes.values()), unparsed=not pred.get("error_mode"),
                                 reward=sc if (pred.get("error_mode") or pred.get("agent_name")) else 0.0,
                                 vlp=[float(next(iter(d.values())).logprob) for d in (c.logprobs or [])],
                                 length_hit=len(toks) >= cfg["max_new_tokens"] and toks[-1] not in eos))
        samples, skipped = [], 0
        for i in range(len(batch)):
            grp = outs[i * G:(i + 1) * G]
            rw = torch.tensor([o["reward"] for o in grp])
            sd = float(rw.std())
            if sd < 1e-6:
                skipped += 1
                continue
            samples += list(zip(grp, ((rw - rw.mean()) / (sd + 1e-4)).tolist()))
        ntok = sum(len(o["tokens"]) for o, _ in samples)
        loss_sum, gaps, T["backward"], norm = 0.0, [], 0.0, 0.0
        lr = cfg["lr_peak"] * min(1.0, step / cfg["warmup"])
        if samples:
            while True:
                groups = budget_groups([len(o["row"]["pids"]) + len(o["tokens"]) for o, _ in samples], budget, rnd)
                zero_grad(params)
                m.train()
                torch.cuda.reset_peak_memory_stats()
                loss_sum, gaps, t = 0.0, [], time.time()
                try:
                    for g in groups:
                        grp = [samples[i] for i in g]
                        ts = [o["tokens"] for o, _ in grp]
                        x, ix = inputs(proc, [o["row"] for o, _ in grp], "student", ts, round_to=rnd)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            h = m.model(**x, use_cache=False).last_hidden_state
                        loss = 0.0
                        for j, (o, adv) in enumerate(grp):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                z = head(m, h[j, ix[j]:ix[j] + len(o["tokens"])]).float()
                            lps = -F.cross_entropy(z, torch.tensor(o["tokens"], device=z.device), reduction="none")
                            if len(o["vlp"]) == len(o["tokens"]):
                                gaps.append(float((lps.detach() - torch.tensor(o["vlp"], device=z.device)).abs().mean()))
                            loss = loss - adv * lps.sum()
                        loss = loss / ntok
                        loss.backward()
                        loss_sum += float(loss.detach())
                        del x, h, z, loss
                    break
                except torch.OutOfMemoryError:
                    if budget <= 2048:
                        raise
                    budget //= 2
                    zero_grad(params)
                    torch.cuda.empty_cache()
            T["backward"] = time.time() - t
            t = time.time()
            norm = opt.step(lr)
            torch.cuda.synchronize()
            T["optimizer"] = time.time() - t
        rw = [o["reward"] for o in outs]
        metric = dict(step=step, loss=loss_sum, lr=lr, gradient_norm=norm, reward=sum(rw) / len(rw), rollout_joint=sum(o["joint"] for o in outs) / len(outs),
                      rollout_axes={k: sum(o["axes"][k] for o in outs) / len(outs) for k in ("agent", "step", "mode")}, skipped_groups=skipped,
                      trained_samples=len(samples), generated_tokens=sum(len(o["tokens"]) for o in outs), trained_answer_tokens=ntok,
                      sync_logprob_gap=(sum(gaps) / len(gaps) if gaps else None), unparsed=sum(o["unparsed"] for o in outs),
                      length_hits=sum(o["length_hit"] for o in outs), token_budget=budget, seconds={k: round(v, 2) for k, v in T.items()},
                      elapsed=time.time() - start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        save_json(root / "updates" / f"update_{step:03}.json", metric)
        print("UPDATE", json.dumps({k: metric[k] for k in ("step", "reward", "rollout_joint", "skipped_groups", "sync_logprob_gap", "seconds")}), flush=True)
        if step in cfg.get("export_steps", []) or step == cfg["total_updates"]:
            export(m, proc, root / "checkpoints" / f"step{step:03}", step)
    save_json(root / "training_complete.json", dict(rows=len(rows), updates=cfg["total_updates"], seconds=time.time() - start))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        save_json(f"{sys.argv[1]}/failed.json", dict(error=traceback.format_exc(), time=time.time()))
        raise

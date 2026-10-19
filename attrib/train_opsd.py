"""On-policy self-distillation (OPSD) with a privileged teacher.

usage: python -m attrib.train_opsd <run_dir>

Student and teacher are the SAME starting model. Each update:
  1. the student samples one answer per trace to the plain prompt (in-process vLLM holding the current weights);
  2. a frozen copy of the starting model scores those same answer tokens with the privileged note appended
     (cfg teacher_cond, default 'gold_step_mode': true step + agent + error mode);
  3. the student minimises the full-vocabulary forward KL(teacher || student) on its own tokens
     (mean over tokens per answer, mean over the batch).
Rows are visited in file order (epochs laid out by attrib.make_run).

Placement: by default everything shares one GPU (fits a ~8B-parameter model on a 180 GB card). For an 80 GB
card set "teacher_device": "cuda:1" and "optimizer": "cpu" (bf16 weights + fp32 master and AdamW in host RAM).
Each update logs sync_logprob_gap = mean |log p_vLLM - log p_student| on the sampled tokens; it stays ~0.01 when
the engine holds the current weights.
"""
import json
import sys
import time
import traceback

from attrib.common import (Optimizer, budget_groups, cosine_lr, end_tokens, export, head, inputs, load_model, load_processor, load_run, make_vllm,
                           prompt_ids, repetitive, save_json, score_text, vllm_wake_with, zero_grad)


def main(run_dir):
    root, cfg, rows = load_run(run_dir)
    llm = make_vllm(cfg["student"], cfg)
    import torch
    import torch.nn.functional as F
    from vllm import SamplingParams
    torch.manual_seed(cfg["seed"])
    proc = load_processor(cfg["student"])
    tok = proc.tokenizer
    tdev = cfg.get("teacher_device", "cuda:0")
    teacher = load_model(cfg["teacher"], device=tdev)
    gpu_opt = cfg.get("optimizer", "gpu") == "gpu"
    m = load_model(cfg["student"], train=True, fp32_trainable=gpu_opt)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = Optimizer(params, cfg["lr_peak"], mode="gpu" if gpu_opt else "cpu", threads=cfg.get("cpu_threads", 14))
    sp = SamplingParams(n=1, temperature=cfg.get("temperature", 0.7), top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20),
                        max_tokens=cfg["max_new_tokens"], seed=cfg["seed"], logprobs=0)
    eos = end_tokens(m, tok)
    B, budget, rnd, cond = cfg["batch"], cfg.get("token_budget", 16384), cfg.get("round_to", 512), cfg.get("teacher_cond", "gold_step_mode")
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
            c = o.outputs[0]
            toks = list(c.token_ids) or [tok.eos_token_id]
            text = tok.decode(toks, skip_special_tokens=True)
            pred, sc, axes = score_text(r, text, cfg["data_root"])
            outs.append(dict(tokens=toks, text=text, score=sc, axes=axes, joint=all(axes.values()), unparsed=not pred.get("error_mode"),
                             vlp=[float(next(iter(d.values())).logprob) for d in (c.logprobs or [])], repetition=repetitive(text),
                             length_hit=len(toks) >= cfg["max_new_tokens"] and toks[-1] not in eos))
        reps = sum(o["repetition"] for o in outs)
        if reps > max(3, len(outs) * cfg.get("repetition_guard", 0.40)):
            raise RuntimeError("rollout repetition guard")
        while True:
            groups = budget_groups([len(r["pids"]) + len(o["tokens"]) + 160 for r, o in zip(batch, outs)], budget, rnd)  # +160: the note
            zero_grad(params)
            m.train()
            torch.cuda.reset_peak_memory_stats()
            kls, gaps, T["teacher"], T["student"] = [], [], 0.0, 0.0
            try:
                for g in groups:
                    rs = [batch[i] for i in g]
                    ts = [outs[i]["tokens"] for i in g]
                    t = time.time()
                    with torch.inference_mode():
                        x, ix = inputs(proc, rs, cond, ts, round_to=rnd, device=tdev)
                        h = teacher.model(**x, use_cache=False).last_hidden_state
                        lq = [F.log_softmax(head(teacher, h[j, ix[j]:ix[j] + len(u)]).float(), -1).to("cuda:0") for j, u in enumerate(ts)]
                        del x, h
                    T["teacher"] += time.time() - t
                    t = time.time()
                    x, ix = inputs(proc, rs, "student", ts, round_to=rnd)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = m.model(**x, use_cache=False).last_hidden_state
                    loss = 0.0
                    for j, u in enumerate(ts):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lp = F.log_softmax(head(m, h[j, ix[j]:ix[j] + len(u)]).float(), -1)
                        vl = outs[g[j]]["vlp"]
                        if len(vl) == len(u):
                            picked = lp.detach()[torch.arange(len(u), device=lp.device), torch.tensor(u, device=lp.device)]
                            gaps.append(float((picked - torch.tensor(vl, device=lp.device)).abs().mean()))
                        q = lq[j].clone()
                        raw = (q.exp() * (q - lp)).sum(-1).mean()
                        kls.append(float(raw.detach()))
                        loss = loss + raw / len(batch)
                    loss.backward()
                    del x, h, lp, lq, loss, q, raw
                    T["student"] += time.time() - t
                break
            except torch.OutOfMemoryError:
                if budget <= 2048:
                    raise
                budget //= 2
                zero_grad(params)
                torch.cuda.empty_cache()
        lr = cosine_lr(step, cfg)
        t = time.time()
        norm = opt.step(lr)
        torch.cuda.synchronize()
        T["optimizer"] = time.time() - t
        metric = dict(step=step, mean_kl=sum(kls) / len(kls), sync_logprob_gap=(sum(gaps) / len(gaps) if gaps else None), lr=lr, gradient_norm=norm,
                      rollout_joint=sum(o["joint"] for o in outs) / len(outs), rollout_score=sum(o["score"] for o in outs) / len(outs),
                      rollout_axes={k: sum(o["axes"][k] for o in outs) / len(outs) for k in ("agent", "step", "mode")},
                      unparsed=sum(o["unparsed"] for o in outs), repetitions=reps, length_hits=sum(o["length_hit"] for o in outs),
                      mean_response_tokens=sum(len(o["tokens"]) for o in outs) / len(outs), token_budget=budget, micro_batches=len(groups),
                      seconds={k: round(v, 2) for k, v in T.items()}, elapsed=time.time() - start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        save_json(root / "updates" / f"update_{step:03}.json", metric)
        print("UPDATE", json.dumps({k: metric[k] for k in ("step", "mean_kl", "sync_logprob_gap", "lr", "rollout_joint", "seconds")}), flush=True)
        if step in cfg.get("export_steps", []) or step == cfg["total_updates"]:
            export(m, proc, root / "checkpoints" / f"step{step:03}", step)
    save_json(root / "training_complete.json", dict(rows=len(rows), updates=cfg["total_updates"], seconds=time.time() - start))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        save_json(f"{sys.argv[1]}/failed.json", dict(error=traceback.format_exc(), time=time.time()))
        raise

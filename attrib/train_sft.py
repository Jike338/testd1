"""Supervised fine-tuning baseline (gold-label targets or rejection-sampled teacher targets).

usage: python -m attrib.train_sft <run_dir>        (one GPU; rows carry a 'target' answer text)

Cross-entropy on the answer tokens only (answer + end-of-turn token), token-mean over the batch of 32.
Rows are shuffled per epoch (seed + epoch). Logs per-update cost (seconds, tokens) to updates/update_NNN.json.
"""
import json
import random
import sys
import time
import traceback

from attrib.common import Optimizer, budget_groups, cosine_lr, export, head, inputs, load_model, load_processor, load_run, prompt_ids, save_json, zero_grad


def main(run_dir):
    import torch
    import torch.nn.functional as F
    root, cfg, base = load_run(run_dir)
    torch.manual_seed(cfg["seed"])
    proc = load_processor(cfg["student"])
    tok = proc.tokenizer
    gpu_opt = cfg.get("optimizer", "gpu") == "gpu"
    m = load_model(cfg["student"], train=True, fp32_trainable=gpu_opt)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = Optimizer(params, cfg["lr_peak"], mode="gpu" if gpu_opt else "cpu", threads=cfg.get("cpu_threads", 14))
    end = cfg["end_token_id"]  # the token that closes an assistant turn (Gemma 4: 106 '<turn|>'; Qwen 3.5: 248046 '<|im_end|>')
    for r in base:
        r["answer"] = tok(r["target"], add_special_tokens=False)["input_ids"] + [end]
        r["plen"] = len(prompt_ids(proc, r))
    order = []
    for e in range(cfg["epochs"]):
        x = list(base)
        random.Random(cfg["seed"] + e).shuffle(x)
        order += x
    B, budget, rnd = cfg["batch"], cfg.get("token_budget", 16384), cfg.get("round_to", 512)
    start = time.time()
    for step, lo in enumerate(range(0, len(order), B), 1):
        batch = order[lo:lo + B]
        ntok = sum(len(r["answer"]) for r in batch)
        tick = time.time()
        while True:
            groups = budget_groups([r["plen"] + len(r["answer"]) for r in batch], budget, rnd)
            zero_grad(params)
            m.train()
            torch.cuda.reset_peak_memory_stats()
            loss_sum, seq_tokens = 0.0, 0
            try:
                for g in groups:
                    rs = [batch[i] for i in g]
                    ans = [r["answer"] for r in rs]
                    x, ix = inputs(proc, rs, "student", ans, round_to=rnd)
                    seq_tokens += int(x["attention_mask"].sum())
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = m.model(**x, use_cache=False).last_hidden_state
                    loss = 0.0
                    for j, t in enumerate(ans):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            z = head(m, h[j, ix[j]:ix[j] + len(t)]).float()
                        loss = loss + F.cross_entropy(z, torch.tensor(t, device=z.device), reduction="sum")
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
        lr = cosine_lr(step, cfg)
        norm = opt.step(lr)
        torch.cuda.synchronize()
        metric = dict(step=step, epoch=lo / len(base), loss=loss_sum, lr=lr, gradient_norm=norm, answer_tokens=ntok, sequence_tokens=seq_tokens,
                      micro_batches=len(groups), token_budget=budget, seconds=time.time() - tick, elapsed=time.time() - start,
                      peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        save_json(root / "updates" / f"update_{step:03}.json", metric)
        print("UPDATE", json.dumps({k: metric[k] for k in ("step", "loss", "lr", "gradient_norm", "seconds", "peak_gib")}), flush=True)
        if step in cfg.get("export_steps", []) or step == cfg["total_updates"]:
            export(m, proc, root / "checkpoints" / f"step{step:03}", step)
    save_json(root / "training_complete.json", dict(rows=len(base), updates=cfg["total_updates"], seconds=time.time() - start))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        save_json(f"{sys.argv[1]}/failed.json", dict(error=traceback.format_exc(), time=time.time()))
        raise

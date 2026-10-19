"""Shared runtime for SFT, OPSD and GRPO on WHO&WHEN PRO failure attribution.

Every trainer reads `<run_dir>/config.json` and `<run_dir>/train.json` (rows built by `attrib/prepare.py`).
Prompts, parsing and scoring come from the official `whowhen_eval` harness, so training rewards and
evaluation use exactly the benchmark's rules.
"""
import collections
import json
import math
import re
import sys
import time
import types
from pathlib import Path

# The harness imports litellm for its API client; it is not needed here.
sys.modules.setdefault("litellm", types.ModuleType("litellm"))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load_run(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = json.loads((run_dir / "train.json").read_text())
    return run_dir, cfg, rows


# ---------------------------------------------------------------- models

def load_processor(path):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"
    return proc


def load_model(path, device="cuda:0", train=False, fp32_trainable=False, grad_ckpt=True):
    """Load an image-text-to-text checkpoint (Gemma 4, Qwen 3.5 ...). Only the text tower is trained.

    fp32_trainable: keep trainable weights in fp32 on the GPU (GPU AdamW). Otherwise weights stay bf16 and the
    optimizer keeps an fp32 master copy on the host (see `make_optimizer`).
    """
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    m.requires_grad_(False)
    if train:
        names = [n for n, _ in m.named_parameters()]
        if any("language_model" in n for n in names):
            keep = lambda n: "language_model" in n or n.startswith("lm_head")
        else:  # architectures without a 'language_model' sub-module: train everything except the vision/audio towers
            keep = lambda n: not any(t in n for t in ("vision", "visual", "audio", "multi_modal_projector", "embed_vision", "embed_audio"))
        for name, p in m.named_parameters():
            if keep(name):
                p.requires_grad_(True)
                if fp32_trainable:
                    p.data = p.data.float()
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert n_train > 0, "no trainable parameters matched; adjust the filter in attrib/common.load_model"
        print(f"trainable parameters: {n_train / 1e9:.2f}B", flush=True)
        if grad_ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.eval()
    return m


def head(model, h):
    """Output logits, with the model's final-logit soft-capping when it has one (Gemma: 30)."""
    import torch
    z = model.get_output_embeddings()(h)
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return torch.tanh(z / cap) * cap if cap else z


def end_tokens(model, tok):
    g = model.generation_config.eos_token_id
    return set([tok.eos_token_id] + (g if isinstance(g, list) else [g]))


class Optimizer:
    """AdamW either on the GPU (fp32 trainable weights, fused kernel) or with an fp32 master copy in host RAM
    (bf16 weights on the GPU; needed for a 4B model on an 80 GB card). Same update either way."""

    def __init__(self, params, lr, mode="gpu", threads=14):
        import torch
        self.params, self.mode = params, mode
        if mode == "cpu":
            torch.set_num_threads(threads)
            self.master = [p.detach().to("cpu", dtype=torch.float32, copy=True).pin_memory() for p in params]
            for c in self.master:
                c.grad = torch.zeros_like(c).pin_memory()
            self.opt = torch.optim.AdamW(self.master, lr=lr, weight_decay=0, foreach=True)
        else:
            self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=0, fused=True)

    def step(self, lr, clip=1.0):
        import torch
        for g in self.opt.param_groups:
            g["lr"] = lr
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, clip, error_if_nonfinite=True))
        if self.mode == "cpu":
            for c, p in zip(self.master, self.params):
                c.grad.copy_(p.grad, non_blocking=True)
            torch.cuda.synchronize()
            self.opt.step()
            for c, p in zip(self.master, self.params):
                p.data.copy_(c, non_blocking=True)
                p.grad = None
        else:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return norm


def zero_grad(params):
    for p in params:
        p.grad = None


def export(model, proc, dest, step):
    import torch
    state = {n: p.detach().to("cpu", dtype=torch.bfloat16) if p.is_floating_point() else p.cpu() for n, p in model.state_dict().items()}
    model.save_pretrained(dest, state_dict=state)
    proc.save_pretrained(dest)
    save_json(Path(dest) / "export.json", dict(step=step, time=time.time()))


# ---------------------------------------------------------------- prompts and batching

def prompt(proc, row, cond="student"):
    """Chat-formatted prompt. cond='student' is the official all-at-once prompt; any other cond appends that
    privileged note (row['hints'][cond]), e.g. 'gold_step_mode' for the OPSD teacher."""
    parts = row["parts"] + ([{"type": "text", "text": row["hints"][cond]}] if cond != "student" else [])
    return proc.apply_chat_template([{"role": "user", "content": parts}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt_ids(proc, row, cond="student"):
    return proc.tokenizer(prompt(proc, row, cond), add_special_tokens=False)["input_ids"]


def inputs(proc, rows, cond, answers, round_to=0, device="cuda:0"):
    """Left-padded prompt+answer batch. Returns tensors and, per row, the position whose hidden state predicts
    the first answer token. round_to pads the width up to a multiple (keeps the set of shapes small)."""
    import torch
    tok = proc.tokenizer
    enc = [prompt_ids(proc, r, cond) for r in rows]
    seq = [e + a for e, a in zip(enc, answers)]
    width = max(map(len, seq))
    if round_to:
        width = -(-width // round_to) * round_to
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seq):
        ids[i, -len(s):] = torch.tensor(s)
        mask[i, -len(s):] = 1
    offsets = [width - len(s) + len(e) - 1 for s, e in zip(seq, enc)]
    return dict(input_ids=ids.to(device), attention_mask=mask.to(device)), offsets


def budget_groups(lengths, budget, round_to=512):
    """Micro-batches over one batch: sort by length, fill each up to `budget` padded tokens. Only the chunking
    changes; the update still sums over the whole batch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups, cur = [], []
    for i in order:
        w = -(-lengths[i] // round_to) * round_to if round_to else lengths[i]
        if cur and (len(cur) + 1) * w > budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    return groups


def cosine_lr(step, cfg):
    peak, end, warm, total = cfg["lr_peak"], cfg["lr_end"], cfg["warmup"], cfg["total_updates"]
    if step <= warm:
        return peak * step / warm
    pr = (step - warm) / max(1, total - warm)
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * pr))


# ---------------------------------------------------------------- scoring

_TAX = None


def score_text(row, text, data_root):
    """Official parse + score. Returns (prediction, mean of agent/step/mode correctness, per-axis dict)."""
    global _TAX
    from whowhen_eval.parse import parse_all_at_once
    from whowhen_eval.prompts import load_taxonomy
    from whowhen_eval.score import score as score_prediction
    if _TAX is None:
        _TAX = load_taxonomy(Path(data_root))
    p = parse_all_at_once(text, _TAX.codes)
    pred = dict(agent_name=p.agent_name, step_coord=p.step_coord, error_mode=p.error_mode, reason=p.reason)
    s = score_prediction(pred, row["gt"], row["framework"])
    axes = {k: bool(s.get(k)) for k in ("agent", "step", "mode")}
    return pred, sum(axes.values()) / 3, axes


def repetitive(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    grams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    return max(collections.Counter(grams).values(), default=0) >= 3


# ---------------------------------------------------------------- in-process vLLM (OPSD, GRPO)

def make_vllm(path, cfg):
    """vLLM engine in this process (sleeps while the trainer runs), so new weights are copied from GPU memory."""
    import os
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=cfg.get("vllm_mem", 0.20), max_model_len=cfg.get("vllm_max_len", 20000),
              enable_sleep_mode=True, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, seed=cfg["seed"])
    llm.sleep(level=2)
    return llm


def vllm_wake_with(llm, model):
    """Wake the engine and load the trainer's current weights (bf16) into it."""
    import torch
    sd = model.state_dict()

    def push(m):
        m.load_weights((n, t.detach().to(torch.bfloat16)) for n, t in sd.items())
        return True

    torch.cuda.empty_cache()
    llm.wake_up(tags=["weights"])
    llm.apply_model(push)
    llm.wake_up(tags=["kv_cache"])

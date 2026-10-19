# Failure attribution on WHO&WHEN PRO: SFT, OPSD and GRPO

This repository trains and evaluates small open models as **failure attributors** for multi-agent LLM systems on
[WHO&WHEN PRO](https://arxiv.org/abs/2607.09996). It contains everything needed to reproduce our current setting:
- data preparation with our exact splits;
- three training methods;
- evaluation with the benchmark's official harness;
- the metric script.

The launch scripts are deliberately thin. Adapt them to your cluster or scheduler: all paths are arguments, and nothing
assumes a particular file system.

## The task

Each WHO&WHEN PRO trace is one failed run of an agent system. The attributor reads the whole trace once (the
benchmark's **all-at-once** protocol) and answers in a fixed format:

```
Agent Name: <the agent whose turn first introduces the decisive error>
Step Number: <its step coordinate>
Error Mode: <one code from the taxonomy>
Reason: <one or two sentences>
```

So every trace carries **three gold labels**:
- **Who:** the agent;
- **When:** the step;
- **What:** one of the 17 error modes in the taxonomy (14 of them occur in the text release).

The labels are exact by construction: the benchmark replays a successful run and injects one error at a chosen step.

**What we found.** Models label the visible *symptom*, not the *cause*:
- **Reasoning modes are over-predicted.** Gold labels are 57% reasoning modes (R.\*) and about 36% omission/process modes
  (premature termination 19%, inadequate verification 8%, over-reliance on another agent 6%, …). Zero-shot models
  predict a reasoning mode 82–96% of the time, and their exact-mode recall on omission modes is 0–4%.
- **Localization ≠ diagnosis.** Given the true step, models reach 93–99% on Who/When but gain only about 3 points on the
  error mode. Prompting and thinking mode don't fix it.
- **The same model can diagnose when shown the answer key.** Given the full answer (step + agent + mode), it explains
  the error well. That makes it usable as a privileged teacher (OPSD below).

## Data

1. **Get the benchmark harness and data:**
   ```bash
   git clone https://github.com/whowhenpro/whowhen_pro.git && pip install -e whowhen_pro
   ```
   Then download the release from Hugging Face ([`Leoxx/whowhen_pro`](https://huggingface.co/datasets/Leoxx/whowhen_pro)),
   following the harness README. You need a data folder (called `whowhen-pro/` below) containing `taxonomy.yaml` and
   `data/text.jsonl`.
2. **We use the text release only:** 6,257 traces from 8 frameworks. The image and video traces are not used.
3. **Splits** (exact id lists in `data/splits/`):

| Split | Traces | Frameworks | Use |
|---|---|---|---|
| `train` | 4,405 | smolagents, mathchat, alfagent, debate, macnet | training (file order = training order) |
| `id_test` | 784 | same 5 frameworks: a random 15% (seed 20261018) | in-domain evaluation |
| `ood_test` | 1,044 | magentic-one (811), dylan (152), metagpt (81), all held out | out-of-distribution evaluation |

   The train list excludes about 24 traces whose prompt is longer than 16K tokens. Counting tokens with a different
   tokenizer can shift that by a couple of traces; one of our clusters used 4,403. `prepare.py --rebuild` regenerates the
   splits from the rule.
4. **Build the rows:**
   ```bash
   python -m attrib.prepare --data-root whowhen-pro --splits data/splits --out data/built
   ```
   This writes `train.json`, `id_test.json` and `ood_test.json`. Each row holds:
   - the official prompt;
   - the gold label;
   - the privileged notes: `gold_step`, `gold_step_mode` (the OPSD teacher note) and `omission`.

## Metrics

`python -m attrib.compare --splits data/splits results/<run> ...` reports, for id_test and ood_test separately:

| Metric | Definition | Traces scored (id / ood) |
|---|---|---|
| **Who** | agent accuracy, averaged over frameworks; multi-agent systems only (the harness treats smolagents, alfagent and mathchat as single-agent) | 83 / 1,044 |
| **When** | step accuracy, averaged over frameworks | 784 / 1,044 |
| **What** | error-mode macro-F1 over the gold modes present (the benchmark's What) | 784 / 1,044 |
| **Joint** | agent, step and mode all correct on the same trace, averaged over frameworks | 784 / 1,044 |
| mode | pooled error-mode accuracy | 784 / 1,044 |
| predR | share of predictions that are reasoning modes (R.\*); the symptom bias | — |
| omRec / omPre | exact-mode recall / precision on omission and process modes (A.3, V.1, V.2, C.1, C.2, C.3, PL.1, PL.2) | — |

**Evaluation settings:** greedy decoding, thinking off, up to 2,048 new tokens, the official prompt, parser and scorer
(`attrib/eval_vllm.py` against a vLLM server). Results files are compatible with `python -m whowhen_eval.leaderboard`.

## Methods

Common to all three methods:
- **Students:** `google/gemma-4-E4B-it` and `Qwen/Qwen3.5-4B` (larger options under Hardware notes).
- **What is trained:** only the text tower (`language_model` + `lm_head`), with full fine-tuning.
- **Schedule:** batch 32, 2 epochs, one seed.
- **Precision:** bf16 compute with fp32 weights in the optimizer.

| Method | What it learns from | Key settings (`configs/`) |
|---|---|---|
| **SFT, gold labels** | the gold answer, with a reason sentence from the taxonomy definition | lr 1e-5 → 1e-6 cosine, 5% warmup, CE on answer tokens |
| **SFT, rejection-sampled (RS)** | the same model's answers **with** the `gold_step_mode` note, kept only when agent, step and mode are all correct (Gemma: 62% of traces, Qwen: 93%) | same as above |
| **OPSD** (ours) | the student samples its own answer; the **same starting model, frozen and given the `gold_step_mode` note**, scores those tokens; loss = full-vocabulary forward KL(teacher ‖ student) | lr 1e-5 → 1e-6 cosine, 5 warmup; sampling T 0.7, top-p 0.95, top-k 20, ≤256 tokens; lr sweep {1e-5, 5e-6} in progress |
| **GRPO, data-matched** | all train traces × 8 samples; reward = mean of agent/step/mode correctness (0 if unparsable); group-normalized advantages; zero-signal groups skipped; no KL term | lr 1e-6 constant after 5 warmup; T 1.0, full distribution |
| **GRPO, compute-matched** | a fixed 552-trace subset, 4 traces × 8 samples per update | same number of sampled answers and updates as OPSD |

At the same lr (2e-6), OPSD and RS-SFT tie. The gaps we have seen so far come from lr and schedule, which is why OPSD
gets the same lr as SFT and the sweep. Because OPSD needs no correct target, it uses every training trace; RS-SFT keeps
only the traces the teacher answered correctly.

### Running

```bash
# 1. SFT targets
python -m attrib.sft_targets gold --rows data/built/train.json --data-root whowhen-pro --out data/built/sft_gold.json
#    RS targets: evaluate the base model WITH the note on the train ids, then filter
bash scripts/serve_eval.sh google/gemma-4-E4B-it teacher_note_train whowhen-pro 8000 --hint gold_step_mode --ids data/splits/train_ids.txt
python -m attrib.sft_targets rs --rows data/built/train.json --teacher-results results/teacher_note_train --out data/built/sft_rs.json

# 2. Run directories (config + rows in training order)
M=google/gemma-4-E4B-it
python -m attrib.make_run --config configs/sft.json  --rows data/built/sft_gold.json --run-dir runs/sft_gold  --model $M --data-root whowhen-pro
python -m attrib.make_run --config configs/sft.json  --rows data/built/sft_rs.json   --run-dir runs/sft_rs    --model $M --data-root whowhen-pro
python -m attrib.make_run --config configs/opsd.json --rows data/built/train.json    --run-dir runs/opsd_1e5  --model $M --data-root whowhen-pro
python -m attrib.make_run --config configs/opsd.json --rows data/built/train.json    --run-dir runs/opsd_5e6  --model $M --data-root whowhen-pro --set lr_peak=5e-6 lr_end=5e-7
python -m attrib.make_run --config configs/grpo.json --rows data/built/train.json    --run-dir runs/grpo_data --model $M --data-root whowhen-pro
python -m attrib.make_run --config configs/grpo.json --rows data/built/train.json    --run-dir runs/grpo_comp --model $M --data-root whowhen-pro --subset 552 --set batch=4

# 3. Train + evaluate each export (one GPU per run; runs are independent, so launch them in parallel)
CUDA_VISIBLE_DEVICES=0 bash scripts/train_and_eval.sh opsd runs/opsd_1e5 whowhen-pro 8000
CUDA_VISIBLE_DEVICES=1 bash scripts/train_and_eval.sh opsd runs/opsd_5e6 whowhen-pro 8001

# 4. Zero-shot reference and comparison
bash scripts/serve_eval.sh google/gemma-4-E4B-it e4b_zero_shot whowhen-pro 8002
python -m attrib.compare --splits data/splits results/e4b_zero_shot results/opsd_1e5_step276 results/sft_rs_step172
```

Every trainer writes:
- `runs/<name>/updates/update_NNN.json`: loss/KL/reward, lr, gradient norm, per-phase seconds, token counts, peak memory;
- `checkpoints/stepNNN/`: bf16 weights at each epoch end;
- `training_complete.json`, or `failed.json` with the traceback.

### Hardware notes

- **Tested on:** NVIDIA B200 (180 GB) for Gemma E4B and A100 80 GB for Qwen3.5-4B.
- **One GPU per run.**
  - With 180 GB, a whole run fits on one GPU: student, frozen teacher and the vLLM engine (`vllm_mem` 0.20).
  - On 80 GB cards use `--set optimizer=cpu` (bf16 weights on GPU, fp32 master + AdamW in host RAM, about 70 GB for a 4B
    model). For OPSD, also set `teacher_device=cuda:1` and expose two GPUs.
- **Qwen3.5-4B in SFT:** pass `--set end_token_id=248046` (`<|im_end|>`). The Gemma default is 106.
- **Rollouts** come from an in-process vLLM engine that sleeps during training. The trainer copies its current weights
  into the engine each update. `sync_logprob_gap` in the update logs should stay around 0.01; much larger means the engine
  is not getting the new weights.
- **Memory:** micro-batches are length-sorted and filled up to `token_budget` (16K padded tokens), with widths rounded to
  512. If a batch runs out of memory, the budget halves automatically. Gradient checkpointing stays on (turning it off
  ran out of memory even on 180 GB).
- **Environment:** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `VLLM_USE_FLASHINFER_SAMPLER=0` (the
  FlashInfer sampling kernel failed to JIT-compile on our nodes). `scripts/train_and_eval.sh` sets both.
- **Larger students (not yet run by us):** the code is model-agnostic, so `--model` is the only required change.

| Student | Notes |
|---|---|
| `Qwen/Qwen3.5-9B` | same tokenizer as the 4B, so `end_token_id=248046` |
| `google/gemma-4-12B-it` | dense; `end_token_id=106`; a different model class, handled by the parameter filter in `common.load_model` |

  Gemma 4 also has a 26B-A4B mixture-of-experts model, which we haven't tried to train.

  At 9B and above, full fine-tuning with the optimizer on the GPU needs about 16 bytes/parameter, more than one 180 GB
  card holds alongside a teacher and vLLM. Use:
  - `--set optimizer=cpu`: bf16 weights on the GPU, fp32 master + AdamW in host RAM. Plan on about 16 bytes/parameter of
    host RAM: ~145 GB for 9B, ~190 GB for 12B.
  - for OPSD, also `teacher_device=cuda:1` (two GPUs per run);
  - possibly a smaller `token_budget` (e.g. 8192) if activations don't fit.

- **Rough wall-clock per 2-epoch run:**

| Run | B200, Gemma E4B | A100, Qwen 4B |
|---|---|---|
| SFT, gold labels | ~2.5 h | ~4 h |
| SFT, RS | ~1.7 h | ~3.7 h |
| OPSD | ~3.3 h | ~5.5 h (2 GPUs) |
| GRPO, compute-matched | ~1.5–2 h | — |
| GRPO, data-matched | ~9–10 h | about a day or more |

## Reference numbers (preliminary)

Held-out Joint, in-domain / OOD. Greedy, one seed.

| Model | Zero-shot | OPSD (lr 2e-6, 1 epoch) | SFT-RS (lr 1e-5, 2 ep) | SFT gold (lr 1e-5, 2 ep) |
|---|---|---|---|---|
| Gemma-4 E4B | 17.1 / 15.7 | 39.1 / 22.1 | 63.2 / 31.4 | 85.1 / 20.2 |
| Qwen3.5-4B | 23.4 / 12.1 | 67.8 / 32.7 | in progress | in progress |

For scale, zero-shot Joint for larger models is:
- Qwen3.5-27B: 28.2 / 26.0;
- Gemma-4 31B: 24.7 / 21.6;
- Gemini 3.1 Pro with thinking: 15.7 / 24.5.

Gold-label SFT nearly saturates the in-domain test, but its OOD step accuracy collapses (When 48). Out of domain, the
methods differ most.

## Layout

```
attrib/
  common.py       model loading, prompts, token-budget batching, optimizer, official scoring, in-process vLLM
  prepare.py      rows + splits from the release
  sft_targets.py  gold and rejection-sampled SFT targets
  make_run.py     run directory from a config template (epochs laid out, subset, overrides)
  train_sft.py    SFT
  train_opsd.py   on-policy self-distillation with the privileged note
  train_grpo.py   GRPO (data- or compute-matched)
  eval_vllm.py    official all-at-once evaluation against an OpenAI-compatible server
  compare.py      Who / When / What / Joint and the omission diagnostics
configs/          sft.json, opsd.json, grpo.json
data/splits/      train / id_test / ood_test trace ids
scripts/          serve_eval.sh, train_and_eval.sh (thin; adapt to your infrastructure)
```

## License

MIT.

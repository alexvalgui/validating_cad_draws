# CV4DC drawing–render verification pipeline (Qwen3-VL-2B)

Scripts mapping directly onto the four sections of the paper scope.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model: `Qwen/Qwen3-VL-2B-Instruct` (set in `config.py`; swap the string to run
the 8B comparison row — nothing else changes).

## Data expected

One manifest CSV per split source, listing each part once:

```
part_id,drawing_path,render_path
00021,drawings/00021.png,renders/00021.png
```

Keep three disjoint part sets: (a) ~29K training parts, (b) the 1,000-pair
test benchmark parts, (c) the ~300 Fusion 360 Gallery parts. Build the test
and Fusion 360 JSONLs with the same script (they're just pair files with
`label` 1/0) or hand-curate them once and freeze them.

## 1. Build training pairs with hard negatives (paper §3, data)

```bash
python 00_prepare_data.py --manifest train_parts.csv --root /data/zero2cad \
    --out-train data/train.jsonl --out-val data/val.jsonl
```

Hard negatives default to OpenCLIP visual nearest-neighbours among renders
(cheap "geometry-aware" proxy). If you have real geometric descriptors from
the CAD (recommended for the paper's claim), save them as an `.npy` matrix in
manifest row order and pass `--embeddings feats.npy` — the mining code is
identical.

## 2. Zero-shot baselines + calibration diagnosis (paper §1–2)

```bash
python 01_eval.py --data data/test.jsonl --out results/zs_2b.csv
python 01_eval.py --data data/test.jsonl --model Qwen/Qwen3-VL-8B-Instruct --out results/zs_8b.csv
```

The `*_report.json` gives you, per model:

- `accuracy_forced_choice` vs `roc_auc_p_yes` vs `accuracy_best_threshold` —
  the AUC/accuracy gap is the "signal present, badly decoded" evidence.
- `answered_yes_rate` and the two `mean_p_yes_*` values — the fixed Yes/No
  prior, for the diagnosis figure.
- `ece_15bin` — calibration.
- `neg_acc_hard` vs `neg_acc_random` — hard negatives really are harder.

The per-pair CSV (`p_yes` per example) is what you plot: score histograms by
class, ROC curves, reliability diagrams.

For the other four VLM baselines (non-Qwen), reuse the same JSONL and the same
prompt from `config.py`; only `load_model`/`build_inputs` need adapting per
model family.

## 3. LoRA fine-tune (paper §3, main result)

```bash
/usr/bin/time -v python 02_train_lora.py 2>&1 | tee runs/train.log
```

Defaults: r=16 LoRA on LM attention+MLP, vision tower frozen, bf16, loss on
answer tokens only, effective batch 16, 1 epoch. `train.log` + `nvidia-smi`
peak give you the cost-table numbers (GPU-hours, peak VRAM, adapter size).

Smoke test first: `python 02_train_lora.py --max-samples 200`.

If OOM on your card: lower `MAX_PIXELS` in `config.py` (memory scales with
vision tokens), or raise `GRAD_ACCUM` — never raise per-device batch above 1.

## 4. Fine-tuned eval + generalization (paper §3–4)

```bash
python 01_eval.py --data data/test.jsonl      --adapter runs/qwen3vl2b_lora --out results/ft_test.csv
python 01_eval.py --data data/fusion360.jsonl --adapter runs/qwen3vl2b_lora --out results/ft_f360.csv
python 01_eval.py --data data/fusion360.jsonl --out results/zs_f360.csv    # zero-shot comparison
```

Same script, same prompt, same metrics — so the before/after comparison in
the main table is apples-to-apples.

## 5. Release checkpoint

```bash
python 03_merge_adapter.py --adapter runs/qwen3vl2b_lora --out release/qwen3vl2b-dwgverify
```

Release the adapter (small) *and* the merged model, plus `config.py` (it is
the prompt registry) and the frozen test JSONLs.

## Gotchas worth knowing

- **Prompt identity**: zero-shot eval, training targets, and fine-tuned eval
  all read the prompt from `config.py`. Don't fork it, or the calibration
  story is confounded.
- **Split by part, not by pair** — `00_prepare_data.py` already does this;
  keep it true for the test/Fusion sets too.
- **Token ids**: `01_eval.py` prints the Yes/No token ids it aggregates at
  startup; sanity-check them once per model family.
- **Image order** is drawing-first, render-first-mentioned-in-prompt — keep
  consistent everywhere (an easy silent bug when adapting baselines).
- **Reproducibility statement**: record transformers/peft/torch versions,
  GPU model, seed (13), and wall-clock from `train.log`.

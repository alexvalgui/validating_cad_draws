#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import config_gemma 

config = config_gemma


def load_model(model_id, adapter=None):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.ATTN_IMPLEMENTATION,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"Loaded LoRA adapter from {adapter}")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    ip = processor.image_processor
    ip.do_pan_and_scan = config.DO_PAN_AND_SCAN
    ip.pan_and_scan_min_crop_size = config.PAN_AND_SCAN_MIN_CROP_SIZE
    ip.pan_and_scan_max_num_crops = config.PAN_AND_SCAN_MAX_NUM_CROPS
    ip.pan_and_scan_min_ratio_to_activate = config.PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE
    return model, processor


def answer_token_ids(tokenizer, variants):
    """First-token ids of every surface form of an answer word."""
    ids = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def build_inputs(processor, drawing_path, render_path):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": drawing_path},
            {"type": "image", "image": render_path},
            {"type": "text", "text": config.QUESTION},
        ]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


@torch.no_grad()
def p_yes_for_pair(model, processor, yes_ids, no_ids, ex):
    inputs = build_inputs(processor, ex["drawing"], ex["render"]).to(model.device)
    logits = model(**inputs).logits[0, -1]           # next-token distribution
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    lp_yes = torch.logsumexp(logprobs[yes_ids], dim=0).item()
    lp_no = torch.logsumexp(logprobs[no_ids], dim=0).item()
    m = max(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    return p_yes, lp_yes, lp_no


def roc_auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    rank_arr = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2 + 1
        for t in range(i, j):
            rank_arr[t] = avg
        i = j
    pos = [rank_arr[t] for t in range(n) if pairs[t][1] == 1]
    n_pos, n_neg = len(pos), n - len(pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(labels, scores, bins=15):
    total, err = len(labels), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(s, l) for s, l in zip(scores, labels)
               if lo <= s < hi or (b == bins - 1 and s == 1.0)]
        if not sel:
            continue
        conf = sum(s for s, _ in sel) / len(sel)
        acc = sum(l for _, l in sel) / len(sel)
        err += len(sel) / total * abs(conf - acc)
    return err


def best_threshold_accuracy(labels, scores):
    cand = sorted(set(scores)) + [1.01]
    best, best_t = 0.0, 0.5
    for t in cand:
        acc = sum((s >= t) == bool(l) for s, l in zip(scores, labels)) / len(labels)
        if acc > best:
            best, best_t = acc, t
    return best, best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True, help="Per-pair CSV output")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.data)]
    if args.limit:
        examples = examples[: args.limit]

    model, processor = load_model(args.model, args.adapter)
    tok = processor.tokenizer
    yes_ids = answer_token_ids(tok, config.YES_VARIANTS)
    no_ids = answer_token_ids(tok, config.NO_VARIANTS)
    print(f"Yes token ids: {yes_ids} | No token ids: {no_ids}")
    for v in config.YES_VARIANTS + config.NO_VARIANTS:
        print(f"  {v!r:8} -> {tok.encode(v, add_special_tokens=False)}")

    labels, scores, rows = [], [], []
    for k, ex in enumerate(examples):
        p_yes, lp_yes, lp_no = p_yes_for_pair(model, processor, yes_ids, no_ids, ex)
        labels.append(int(ex["label"]))
        scores.append(p_yes)
        rows.append((ex["drawing"], ex["render"], ex["label"],
                     ex.get("neg_type"), p_yes, lp_yes, lp_no))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(examples)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("drawing,render,label,neg_type,p_yes,lp_yes,lp_no\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    acc_forced = sum((s >= 0.5) == bool(l) for s, l in zip(scores, labels)) / len(labels)
    auc = roc_auc(labels, scores)
    acc_best, t = best_threshold_accuracy(labels, scores)
    yes_rate = sum(s >= 0.5 for s in scores) / len(scores)
    mean_pyes_pos = sum(s for s, l in zip(scores, labels) if l) / max(1, sum(labels))
    mean_pyes_neg = sum(s for s, l in zip(scores, labels) if not l) / max(1, len(labels) - sum(labels))

    report = {
        "model": args.model, "adapter": args.adapter, "n": len(labels),
        "accuracy_forced_choice": round(acc_forced, 4),
        "roc_auc_p_yes": round(auc, 4),
        "accuracy_best_threshold": round(acc_best, 4),
        "best_threshold": round(t, 4),
        "ece_15bin": round(ece(labels, scores), 4),
        "answered_yes_rate": round(yes_rate, 4),
        "mean_p_yes_when_positive": round(mean_pyes_pos, 4),
        "mean_p_yes_when_negative": round(mean_pyes_neg, 4),
    }
    for ntype in ("hard", "random"):
        sub = [(s, l) for (_, _, l, nt, s, _, _) in rows if nt == ntype]
        if sub:
            report[f"neg_acc_{ntype}"] = round(
                sum((s < 0.5) for s, _ in sub) / len(sub), 4)

    print(json.dumps(report, indent=2))
    with open(str(Path(args.out).with_suffix("")) + "_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import config_qwen as config


# --------------------------------------------------------------------------- #
def build_inputs(processor, drawing, render):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": drawing},
            {"type": "image", "image": render},
            {"type": "text", "text": config.QUESTION},
        ]},
    ]
    return processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


def first_token_id(tok, word):
    return tok.encode(word, add_special_tokens=False)[0]


def load_model(model_id, adapter):
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


class Attributor:
    """Hooks the vision tower so every stream reaching the LM carries grad."""

    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.visual = model.model.visual
        self.merge = getattr(self.visual, "spatial_merge_size", 2)
        self.feats = []
        self.visual.register_forward_hook(self._hook)
        tok = processor.tokenizer
        self.yes_id, self.no_id = first_token_id(tok, "Yes"), first_token_id(tok, "No")

    def _grab(self, t):
        t = t.detach().requires_grad_(True)
        self.feats.append(t)
        return t

    def _hook(self, mod, inp, out):
        self.feats = []
        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = self._grab(out.last_hidden_state)
            for name in ("deepstack_visual_embeds", "deepstack_features"):
                ds = getattr(out, name, None)
                if ds is not None:
                    setattr(out, name, [self._grab(t) for t in ds])
            return out
        feats = out[0] if isinstance(out, tuple) else out
        feats = self._grab(feats)
        return (feats,) + tuple(out[1:]) if isinstance(out, tuple) else feats

    def _one_pass(self, inputs):
        out = self.model(**inputs)
        logits = out.logits[0, -1]
        target = logits[self.yes_id] - logits[self.no_id]
        target.backward()
        attr = sum((f.detach().float() * f.grad.detach().float()).sum(-1)
                   for f in self.feats if f.grad is not None)
        return attr, target.item()

    def run(self, drawing, render, smooth=1, sigma=0.15):
        """Returns (drawing_map, render_map, logit_gap, grids)."""
        inputs = build_inputs(self.processor, drawing, render).to("cuda")
        pv = inputs["pixel_values"]
        std = pv.float().std().item()
        total, score = None, None
        for k in range(max(smooth, 1)):
            noisy = dict(inputs)
            if smooth > 1:
                noisy["pixel_values"] = (pv.float() + sigma * std * torch.randn_like(pv.float())).to(pv.dtype)
            self.model.zero_grad(set_to_none=True)
            attr, s = self._one_pass(noisy)
            total = attr if total is None else total + attr
            if k == 0:
                score = s                     # clean logit gap from the first pass
        attr = total / max(smooth, 1)

        grid = inputs["image_grid_thw"].cpu().numpy()
        maps, offset = [], 0
        for t, gh, gw in grid:
            mh, mw = gh // self.merge, gw // self.merge
            n = t * mh * mw
            maps.append(attr[offset:offset + n].view(t, mh, mw)[0].cpu().numpy())
            offset += n
        assert offset == attr.shape[0], "token count mismatch"
        return maps[0], maps[1], score


# --------------------------------------------------------------------------- #
def silhouette(path, shape):
    """Boolean part mask resized to the attribution grid (h, w)."""
    from scipy import ndimage
    im = Image.open(path)
    a = np.array(im.split()[-1]) if im.mode == "RGBA" else None
    if a is not None and (a < 128).any():          # real transparency (renders)
        m = a > 127
    else:                                          # line drawings: ink -> filled region
        g = np.array(im.convert("L"))
        ink = g < 200
        ink = ndimage.binary_closing(ink, iterations=6)
        m = ndimage.binary_fill_holes(ink)
        m = ndimage.binary_dilation(m, iterations=4)
    m = Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(m) > 127


def stats(dmap, rmap, drawing, render):
    """Per-pair summary: signed sums and in-silhouette fraction of |attr|."""
    out = {"draw_sum": float(dmap.sum()), "rend_sum": float(rmap.sum())}
    for name, m, path in (("draw", dmap, drawing), ("rend", rmap, render)):
        mask = silhouette(path, m.shape)
        a = np.abs(m)
        out[f"{name}_in_sil"] = float(a[mask].sum() / (a.sum() + 1e-9))
        out[f"{name}_sil_frac"] = float(mask.mean())   # area baseline for comparison
    return out


def overlay(dmap, rmap, drawing, render, score, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, path, m, title in zip(axes, [drawing, render], [dmap, rmap], ["drawing", "render"]):
        img = Image.open(path).convert("RGB")
        a = np.abs(m); a = a / (a.max() + 1e-9)
        a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
        ax.imshow(img); ax.imshow(a, cmap="jet", alpha=0.45)
        ax.set_title(f"{title}  (sum attr {m.sum():+.2f})"); ax.axis("off")
    fig.suptitle(f"logit(Yes) - logit(No) = {score:+.2f}")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drawing"); ap.add_argument("--render")
    ap.add_argument("--out", default="attr.png")
    ap.add_argument("--jsonl", help="batch mode: pairs file with drawing/render/label")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", default="attr_stats.csv")
    ap.add_argument("--save-dir", default=None, help="batch mode: also save overlay PNGs here")
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--smooth", type=int, default=25, help="SmoothGrad passes (1 = off)")
    ap.add_argument("--sigma", type=float, default=0.15, help="noise as fraction of pixel std")
    args = ap.parse_args()

    processor, model = load_model(args.model, args.adapter)
    A = Attributor(model, processor)

    if args.jsonl:
        items = [json.loads(l) for l in open(args.jsonl)]
        if args.limit:
            items = items[:args.limit]
        rows = []
        for i, ex in enumerate(items, 1):
            try:
                d, r, s = A.run(ex["drawing"], ex["render"], args.smooth, args.sigma)
            except Exception as e:
                print(f"skip {ex['drawing']}: {e}"); continue
            row = {"drawing": ex["drawing"], "render": ex["render"],
                   "label": ex.get("label"), "neg_type": ex.get("neg_type"),
                   "logit_gap": s, **stats(d, r, ex["drawing"], ex["render"])}
            rows.append(row)
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                tag = f"{i:04d}_lab{ex.get('label')}_{ex.get('neg_type') or 'pos'}"
                overlay(d, r, ex["drawing"], ex["render"], s, os.path.join(args.save_dir, tag + ".png"))
            if i % 20 == 0:
                print(f"{i}/{len(items)}")
        with open(args.stats, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        # quick summary
        import pandas as pd
        df = pd.DataFrame(rows)
        df["ptype"] = np.where(df.label == 1, "pos", "neg_" + df.neg_type.fillna("?").astype(str))
        cols = ["logit_gap", "draw_sum", "rend_sum", "draw_in_sil", "rend_in_sil", "draw_sil_frac", "rend_sil_frac"]
        print(df.groupby("ptype")[cols].mean().round(3).to_string())
        print(f"\nwrote {args.stats}")
    else:
        assert args.drawing and args.render, "give --drawing and --render, or --jsonl"
        d, r, s = A.run(args.drawing, args.render, args.smooth, args.sigma)
        print({k: round(v, 3) for k, v in stats(d, r, args.drawing, args.render).items()})
        overlay(d, r, args.drawing, args.render, s, args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
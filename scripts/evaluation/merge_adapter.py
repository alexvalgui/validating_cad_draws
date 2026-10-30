#!/usr/bin/env python
"""
Merge the LoRA adapter into the base model for release (single self-contained
checkpoint a TEC lab or supplier can load without PEFT).

Usage:
    python 03_merge_adapter.py --adapter runs/qwen3vl2b_lora --out release/qwen3vl2b-dwgverify
"""
import argparse

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.save_pretrained(args.out)
    print(f"Merged model written to {args.out}")


if __name__ == "__main__":
    main()

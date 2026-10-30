#!/usr/bin/env python

import argparse
import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import config_qwen

config = config_qwen


class PairDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None):
        items = [json.loads(l) for l in open(jsonl_path)]
        random.Random(config.SEED).shuffle(items)

        self.items = []
        skipped = 0
        for ex in items:
            if self._valid(ex["drawing"]) and self._valid(ex["render"]):
                self.items.append(ex)
            else:
                skipped += 1
        if skipped:
            print(f"[PairDataset] skipped {skipped}/{len(items)} examples "
                  f"with missing/corrupt images ({jsonl_path})")

        if max_samples:
            self.items = self.items[:max_samples]

    @staticmethod
    def _valid(path):
        if not os.path.isfile(path):
            return False
        try:
            with Image.open(path) as im:
                im.verify()  # cheap corruption check, doesn't decode pixels
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ex = self.items[i]
        answer = "Yes" if ex["label"] == 1 else "No"
        return {"drawing": ex["drawing"], "render": ex["render"], "answer": answer}


class Collator:
    """Tokenise (prompt + answer), mask loss to the answer tokens only.

    Falls back to a random neighboring sample if a file goes bad mid-run
    (e.g. deleted/corrupted after PairDataset's up-front validation).
    """

    def __init__(self, processor, dataset):
        self.processor = processor
        self.dataset = dataset

    def __call__(self, batch):
        assert len(batch) == 1, "per-device batch must be 1 (variable image sizes)"
        ex = batch[0]
        try:
            return self._build(ex)
        except Exception as e:
            print(f"[Collator] skipping bad sample ({e}); substituting")
            fallback = self.dataset[random.randrange(len(self.dataset))]
            return self._build(fallback)

    def _messages(self, ex, with_answer):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ex["drawing"]},
                {"type": "image", "image": ex["render"]},
                {"type": "text", "text": config.QUESTION},
            ]},
        ]
        if with_answer:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": ex["answer"]}]})
        return msgs

    def _build(self, ex):
        full = self.processor.apply_chat_template(
            self._messages(ex, with_answer=True),
            tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            self._messages(ex, with_answer=False),
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100  # mask prompt
        full["labels"] = labels
        return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--train", default=config.TRAIN_JSONL)
    ap.add_argument("--val", default=config.VAL_JSONL)
    ap.add_argument("--out", default=config.OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=config.EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--max-samples", type=int, default=config.MAX_TRAIN_SAMPLES)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    # Freeze everything visual; LoRA the language model.
    for name, p in model.named_parameters():
        if "visual" in name or "vision" in name:
            p.requires_grad = False

    lora = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    
    model.print_trainable_parameters()
    
    lora_mods = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_modules()
                        if n.endswith("lora_A")})
    layers = sorted({int(m.split(".layers.")[1].split(".")[0])
                     for m in lora_mods if ".layers." in m})
    proj = sorted({m.rsplit(".", 1)[-1] for m in lora_mods})
    print(f"[lora] {len(lora_mods)} modules adapted")

    train_ds = PairDataset(args.train, args.max_samples)
    val_ds = PairDataset(args.val, max_samples=500)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor, train_ds),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    
    with open(os.path.join(args.out, "loss_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    model.save_pretrained(args.out)  # adapter only (~tens of MB)
    processor.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}. Evaluate with:\n"
          f"  python 01_eval.py --data {config.TEST_JSONL} --adapter {args.out} "
          f"--out results/ft_test.csv")


if __name__ == "__main__":
    main()
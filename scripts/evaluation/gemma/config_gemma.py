"""Shared configuration for the CV4DC drawing-render verification pipeline.
Gemma 3 4B variant -- second model family for the recipe-generalisation row.
"""

MODEL_ID = "google/gemma-3-4b-it"
ATTN_IMPLEMENTATION = "sdpa"   # switch to "eager" if you see NaN losses during training


SYSTEM_PROMPT = (
    "You are an expert mechanical engineer performing quality inspection "
    "of 3D-printed upper-limb prosthetic components."
)

QUESTION = (
"The first image is a 2D engineering drawing. The second image is a render "
"of a 3D CAD model. Do the drawing and the model show the same part? "
"Answer with exactly one word: Yes or No."
)

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO"]

DO_PAN_AND_SCAN = True
PAN_AND_SCAN_MIN_CROP_SIZE = 256
PAN_AND_SCAN_MAX_NUM_CROPS = 4          # each extra crop = +256 tokens per image
PAN_AND_SCAN_MIN_RATIO_TO_ACTIVATE = 1.2  # square 1024px renders won't trigger crops;
                                          # lower this (e.g. 1.0) to force crops for small text

# ---------------------------------------------------------------- data ------
SEED = 13
TRAIN_JSONL = "/home/ec2-user/cad-2d-vlm-accv/scripts/finetune/data/omnicad/train_dim.jsonl"
VAL_JSONL = "/home/ec2-user/cad-2d-vlm-accv/scripts/finetune/data/omnicad/val_dim.jsonl"
TEST_JSONL = "data/test.jsonl"          # the 1,000-pair benchmark
FUSION360_JSONL = "data/fusion360.jsonl"  # ~300 held-out generalization pairs

# Negative sampling mix per positive pair (must sum to NEG_PER_POS)
NEG_PER_POS = 1          # keep classes balanced: 1 negative per positive
HARD_NEG_FRACTION = 0.7  # fraction of negatives that are geometry-aware hard negatives

# ---------------------------------------------------------------- LoRA ------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


LORA_TARGET_MODULES = [
    # Gemma 3 decoder
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    # SigLIP vision tower (q/k/v_proj above are shared with the decoder)
    "out_proj", "fc1", "fc2",
]

# ---------------------------------------------------------------- train -----
OUTPUT_DIR = "runs/gemma3_4b_lora"
LEARNING_RATE = 1e-4     # unchanged on purpose: same recipe, different backbone
EPOCHS = 1
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16          # effective batch 16
WARMUP_STEPS = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250
MAX_TRAIN_SAMPLES = None  # set an int for smoke tests
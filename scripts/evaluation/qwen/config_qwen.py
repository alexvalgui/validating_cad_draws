"""Shared configuration for the CV4DC drawing-render verification pipeline."""

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"   # swap to Qwen/Qwen3-VL-8B-Instruct for the comparison row

# ---------------------------------------------------------------- prompt ----
# Keep this EXACTLY identical between zero-shot eval, training, and fine-tuned
# eval, otherwise the calibration comparison is confounded.
SYSTEM_PROMPT = (
    "You are an expert mechanical engineer performing quality inspection "
    "of 3D-printed upper-limb prosthetic components."
)

QUESTION = (
"The first image is a 2D engineering drawing. The second image is a render "
"of a 3D CAD model. Do the drawing and the model show the same part? "
"Answer with exactly one word: Yes or No."
)

# Answer vocabulary. We aggregate log-probs over these surface forms because
# tokenizers may split "Yes"/" Yes"/"yes" differently.
YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO"]

# ---------------------------------------------------------------- images ----
# Cap vision tokens to keep memory predictable on a consumer card.
# Qwen's image processor works on pixel budgets (28x28 patches).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28   # raise to 1536*28*28 if drawings' small text matters and VRAM allows

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
    "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",  
    "qkv","proj","linear_fc1","linear_fc2",                                 
]
# ---------------------------------------------------------------- train -----
OUTPUT_DIR = "runs/qwen3vl2b_lora"
LEARNING_RATE = 1e-4
EPOCHS = 1
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16          # effective batch 16
WARMUP_STEPS = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250
MAX_TRAIN_SAMPLES = None  # set an int for smoke tests
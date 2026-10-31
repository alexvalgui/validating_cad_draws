<h1 align="center">
Teaching Small VLMs to Validate Technical Drawings Against CAD Models
</h1>

<div align="center">

Alexander Valverde<sup>1</sup>,
Danny Xie-Li<sup>2</sup>,
Zhichao Wang<sup>1</sup>,
Marcos V. Conde<sup>3</sup>

</div>

<p align="center">
  <sup>1</sup>Novineer, USA
  <br>
  <sup>2</sup>Instituto Tecnológico de Costa Rica, San José, Costa Rica
  <br>
  <sup>3</sup>Julius-Maximilians-Universität Würzburg, Würzburg, Germany
</p>

<p align="center">
  <a href="paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square" alt="Paper (PDF)"></a>
  <img src="https://img.shields.io/badge/ACCV-2026-blue?style=flat-square" alt="ACCV 2026">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B%20%7C%208B-6f42c1?style=flat-square" alt="Qwen3-VL">
  <img src="https://img.shields.io/badge/Gemma--3-4B-4285F4?style=flat-square" alt="Gemma-3">
</p>

<p align="center">
  <!-- Add links when available -->
  <!-- <a href="YOUR_DATASET_URL">Dataset</a> | <a href="YOUR_ARXIV_URL">arXiv</a> | <a href="YOUR_PROJECT_PAGE_URL">Project Page</a> -->
</p>

---

<p align="center">
  <img src="https://github.com/alexvalgui/adapt_eyes_no_words/blob/main/figures/teaser.png"
       alt="A 3D model is rendered as a hidden-line drawing and a shaded image, and a small open-weight VLM is asked whether they show the same part. The answer is read from the log-probabilities of Yes and No."
       style="width: 100%; height: auto; display: block; margin: 0 auto;">
</p>

## Abstract

In regulated manufacturing the 2D engineering drawing is the controlled document, and confirming that it still matches its 3D CAD model is a manual check repeated at every change order, quote, and legacy digitization. Hosted multimodal APIs are often ruled out by cost and by the confidentiality of proprietary or patient-linked geometry. We ask whether a compact open-weight vision-language model can perform this check locally. Framing drawing-to-model verification as a binary task, we evaluate Qwen3-VL-2B, Qwen3-VL-8B, and Gemma-3-4B zero-shot on drawing–render pairs from Omni-CAD, where the two smaller models answer Yes to nearly every pair. Fine-tuning with LoRA on a single GPU raises them to 92% accuracy and 0.976 AUC; adapting the vision encoder, not only the language model, is what makes them balanced and calibrated. Applied unchanged to a new benchmark of 199 upper-limb prosthesis meshes, the adapters reach 75% accuracy with errors confined to a few design families.

## Summary

- Qwen3-VL-2B with text + vision LoRA reaches 92.0% accuracy and 0.976 AUC on held-out Omni-CAD pairs. The same adapters, with no further training, reach 75.0% accuracy and 0.815 AUC on prosthetics.
- Training is one epoch of LoRA on a single NVIDIA L40S (48 GB). Adapters are a few tens of MB, so one base model can serve several adapters.
- We release a prosthetics benchmark: 199 upper-limb prosthesis meshes from GrabCAD and NIH3D, in 88 design families and eight categories, with aligned drawings and renders.
- Everything runs locally; geometry never leaves the machine.

## Task and Method

**Task.** Given a 2D engineering drawing and a render of a 3D CAD model, does the pair show the same part? The model is prompted once and must answer `Yes` or `No`.

```text
System: You are an expert mechanical engineer performing quality inspection.
User:   <image 1> <image 2> The first image is a 2D engineering drawing. The second
        image is a render of a 3D CAD model. Do the drawing and the model show the
        same part? Answer with exactly one word: Yes or No.
```

The answer is not sampled. We read the next-token log-probabilities of `Yes` and `No` (summed over tokenizations), which gives a score that can be thresholded or calibrated.

**Data generation.** Each 3D model is turned into aligned image pairs from six axis-aligned views (front, back, left, right, top, bottom):

- Drawings: orthographic projection with hidden-line removal using OpenCascade's exact HLR on B-reps (polygonal HLR for meshes, 30° dihedral threshold). Visible edges solid, hidden edges dashed, rasterized to 1024×1024 black-on-white PNGs. A dimensioned variant (overall width and height plus hole diameters) is also produced for ablation.
- Renders: Blender, 512×512, perspective camera (45° FOV), uniform matte grey material, fixed key/fill/rim lighting rotated with each view.

**Training data (Omni-CAD).** 10,000 models are selected via farthest-point sampling over geometric features, which favours rare shapes (freeform, thin-walled). The split is by part, never by view: 8,930 parts for training and 470 held-out parts (5,640 balanced pairs) for evaluation. Each positive receives one negative: with probability 0.7 a hard negative (sampled from the top-3 of the ten nearest OpenCLIP ViT-B/32 neighbours of the positive render), otherwise a random part. This yields 107,160 training pairs.

**LoRA scopes.**

| Scope | What is adapted |
|---|---|
| `text` | Language model only |
| `text + vision` | Language model and vision encoder |

## Results

### Omni-CAD held-out set (n = 5,640 balanced pairs)

`Acc` uses p(Yes) ≥ 0.5; `Acc*` uses the best threshold chosen on this set; `Yes%` is the fraction answered Yes.

| Model | Acc ↑ | AUC ↑ | Acc* ↑ | ECE ↓ | Yes% | Hard neg. ↑ | Random neg. ↑ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Zero-shot** | | | | | | | |
| Gemma-3-4B | 50.0 | 0.506 | 51.9 | 0.500 | 99.8 | 0.2 | 0.1 |
| Qwen3-VL-2B | 50.0 | 0.665 | 62.8 | 0.424 | 100.0 | 0.0 | 0.0 |
| Qwen3-VL-8B | 74.2 | 0.822 | 75.2 | 0.181 | 41.1 | 53.5 | 92.2 |
| **Fine-tuned (text)** | | | | | | | |
| Gemma-3-4B | 84.9 | 0.942 | 89.0 | 0.115 | 46.7 | 55.2 | 98.2 |
| Qwen3-VL-2B | 76.6 | 0.943 | 89.6 | 0.198 | 32.3 | 77.4 | 99.4 |
| **Fine-tuned (text + vision)** | | | | | | | |
| Gemma-3-4B | 90.3 | 0.973 | 92.8 | 0.090 | 47.2 | 73.3 | 99.1 |
| **Qwen3-VL-2B** | **92.0** | **0.976** | **93.1** | **0.070** | 51.0 | 64.6 | 99.1 |

Zero-shot, the two smaller models answer Yes to essentially every pair. Adapting the vision encoder brings Qwen3-VL-2B to a balanced 51% Yes rate and reduces the gap between forced-choice and best-threshold accuracy from 13 points to 1.

### Transfer to medical prosthetics (199 parts, 88 design families, n = 2,284 pairs)

Fine-tuned rows reuse the Omni-CAD adapters with no further training.

| Model | Acc ↑ | AUC ↑ | Acc* ↑ | ECE ↓ | Yes% |
|---|:-:|:-:|:-:|:-:|:-:|
| **Zero-shot** | | | | | |
| Gemma-3-4B | 50.0 | 0.504 | 51.2 | 0.500 | 99.7 |
| Qwen3-VL-2B | 50.0 | 0.628 | 59.6 | 0.439 | 100.0 |
| Qwen3-VL-8B | 63.3 | 0.691 | 63.4 | 0.290 | 67.2 |
| **Fine-tuned (text)** | | | | | |
| Gemma-3-4B | 68.6 | 0.763 | 70.2 | 0.127 | 53.9 |
| Qwen3-VL-2B | 72.8 | 0.793 | 73.1 | 0.099 | 58.8 |
| **Fine-tuned (text + vision)** | | | | | |
| Gemma-3-4B | 72.2 | 0.806 | 73.7 | 0.123 | 57.1 |
| **Qwen3-VL-2B** | **75.0** | **0.815** | **75.5** | 0.123 | 63.4 |

Accuracy by functional category (text + vision adapters, same-family negatives excluded):

| Category | Parts | Qwen3-VL-2B | Gemma-3-4B |
|---|:-:|:-:|:-:|
| Straps & small hardware | 13 | 90 | 81 |
| Enclosures & covers | 15 | 89 | 85 |
| Forearm inserts | 17 | 88 | 96 |
| Tensioners | 30 | 88 | 45 |
| Palms, gauntlets & arms | 30 | 86 | 80 |
| Fingers & phalanges | 39 | 85 | 74 |
| Pins | 31 | 80 | 70 |
| Locking mechanisms | 21 | 29 | 78 |

The two models fail on different families (Qwen on locking mechanisms, Gemma on tensioners). Both families are parametric sweeps of 20 to 30 scaled copies of one design, so a single wrong judgement repeats across the family.

### Additional observations

- Text-only adapters learn a decision rule over visual features that were never shaped by drawings. Adding the vision encoder gives the widest class separation and the best calibration for both backbones.
- Evaluated unchanged on dimensioned drawings, text + vision adapters improve (Gemma +2.9, Qwen +1.4 accuracy), while text-only adapters react in a model-specific way (Gemma +4.1, Qwen −6.2).
- Random negatives are solved at 98 to 99%. Remaining errors involve close relatives that differ in local detail such as holes, fillets, or bosses.
- Adding a prosthetics-specific system message changes accuracy by at most 0.2 points.

## Fine-tuning Details

| Parameter | Value |
|---|---|
| LoRA rank / α / dropout | 16 / 32 / 0.05 |
| Adapted modules (text) | `q, k, v, o, gate, up, down` of the language model |
| Adapted modules (text + vision) | the above, plus attention and MLP projections of the vision encoder |
| Precision | bf16 |
| Epochs | 1 (about 6,700 steps) |
| Learning rate | 1e-4, cosine decay, 3% warmup |
| Weight decay | 0 |
| Effective batch size | 16 (1 × 16 accumulation) |
| Visual token budget | 256 to 1024 patches of 28 × 28 px |
| Negatives per positive | 1 (hard with prob. 0.7, else random) |
| Hard-negative mining | OpenCLIP ViT-B/32 on renders, top-10 cosine, sample from top-3 |
| Seed | 13 |
| Hardware | 1 × NVIDIA L40S (48 GB) |

Qwen3-VL-8B is evaluated zero-shot only; training it would roughly double the cost and the 2B adapters already outperform it.

## Released Resources

- Rendered images and hidden-line drawings of our Omni-CAD subset
- The medical prosthetics benchmark (199 meshes, aligned drawings and renders)
- Code
- Trained LoRA adapter weights

The underlying Omni-CAD data is released by its authors under the MIT license. Qwen3-VL is Apache 2.0; Gemma-3 is under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Limitations

- Thresholds marked `τ*` / `Acc*` are chosen on the evaluation set and are in-sample quantities.
- The checkpoint is selected on a 500-pair subset of the held-out set; there is no separate test split.
- Held-out hard negatives are weaker than in training (23% realised hard fraction), so the Hard column is an upper bound for real "close relative" errors.
- One seed and one epoch per adapter.

## Citation

```bibtex
@inproceedings{valverde2026adapt,
  title     = {Adapt the Eyes, Not Just the Words: Teaching Small {VLMs} to Validate Technical Drawings Against {CAD} Models},
  author    = {Valverde, Alexander and Xie-Li, Danny and Wang, Zhichao and Conde, Marcos V.},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV) Workshops},
  year      = {2026}
  }
```

## Acknowledgements

Built on [Omni-CAD / CAD-MLLM](https://arxiv.org/abs/2411.04954), [Qwen3-VL](https://arxiv.org/abs/2511.21631), [Gemma 3](https://arxiv.org/abs/2503.19786), [LoRA](https://arxiv.org/abs/2106.09685), OpenCascade, and Blender. Prosthesis meshes come from GrabCAD and the [NIH 3D Print Exchange](https://3dprint.nih.gov/).
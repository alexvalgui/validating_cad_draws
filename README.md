<br>

<h1 align="center">
Adapt the Eyes, Not Just the Words: Teaching Small VLMs to Validate Technical Drawings Against CAD Models
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
  <!-- Add links when available -->
  <!-- 🤗 <a href="YOUR_DATASET_URL">Dataset</a> | -->
  <!-- <a href="YOUR_HUGGINGFACE_PAPER_URL">Hugging Face Paper</a> | -->
  <!-- <img src="docs/static/images/arxiv_logo.png" alt="arXiv" height="14">
  <a href="YOUR_ARXIV_URL">arXiv</a> | -->
  <!-- 🌐 <a href="YOUR_PROJECT_PAGE_URL">Project Page</a> -->
</p>

---

<p align="center">
  <img src="https://github.com/alexvalgui/adapt_eyes_no_words/blob/main/figures/teaser.png"
       alt="Overview. A 3D model is rendered as a hidden-line drawing and a shaded image, and a small open-weight VLM is asked whether they show the same part. The answer is read from the log-probabilities of \emph{Yes} and \emph{No}. Zero-shot models answer Yes to almost every pair. LoRA on the language model, or on both the language model and the vision encoder (ours), turns them into calibrated verifiers."
       style="width: 100%; height: auto; display: block; margin: 0 auto;">
</p>


## 📝 Abstract
In regulated manufacturing the 2D engineering drawing is the controlled document, and confirming that it still matches its 3D CAD model is a manual check repeated at every change order, quote, and legacy digitization. Hosted multimodal APIs are often ruled out by cost and by the confidentiality of proprietary or patient-linked geometry. We ask whether a compact open-weight vision-language model can perform this check locally. Framing drawing-to-model verification as a binary task, we evaluate Qwen3-VL-2B, Qwen3-VL-8B, and Gemma-3-4B zero-shot on drawing--render pairs from Omni-CAD, where the two smaller models answer Yes to nearly every pair. Our fine-tuning with LoRA on a single GPU raises them to 92% accuracy and 0.976 AUC; adapting the vision encoder, not only the language model, is what makes them balanced and calibrated. Applied unchanged to a new benchmark of 199 upper-limb prosthesis meshes, the adapters reach 75% accuracy with errors confined to a few design families. 


## Citation


# Session 1 Summary: Design, Implementation & Data Pipeline
**Date**: November 14, 2026  
**Project**: *Katzen oder Brötchen* (Coursework Image Classifier)  
**Notebook**: [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb)  
**Design Document**: [`DESIGN.md`](../DESIGN.md)  

---

## 1. Architectural Decisions

1. **Framework & Methodology**:
   - Built on **FastAI** and **PyTorch**, synthesizing Jeremy Howard's Kaggle tutorial (*"Is it a bird?"*) and *Fastbook* Chapters 1 & 2.
   - **Pretrained Backbone**: ResNet-34 fine-tuned using the 1cycle policy and discriminative learning rates (`learn.fine_tune(3)`).
   - **Presizing Augmentation**: Two-stage transformation with CPU `Resize(460)` and GPU `aug_transforms(size=224, min_scale=0.75)` in a single interpolation pass.

2. **Non-Binary Multi-Class Output**:
   - Rather than returning a binary `is_cat: True/False` flag, the model uses `CategoryBlock` and computes normalized **Softmax probability distributions** (e.g. `84.2% Cat, 15.8% Bun`).
   - Inference visualizes a side-by-side plot of the input image and a horizontal probability bar chart.

3. **Pedagogical Structure**:
   - Every code block is preceded by a structured 3-part Markdown cell:
     1. *Practical Goal*
     2. *Core ML Concept*
     3. *Python & FastAI Implementation Details*
   - Code includes clean inline comments.

4. **Git & Dataset Best Practice**:
   - Created [`.gitignore`](../.gitignore) to exclude large binary assets (`data/`, `models/`, `*.pkl`, `.venv/`) from git to prevent repository bloat and GitHub 100 MB push limits.
   - Preserved lightweight external test folder [`testing_images/`](../testing_images/) with `katzebroetchen.jpg` (~9 KB) tracked for instant testing.

---

## 2. Challenges & Errors Encountered (and Resolutions)

| # | Challenge / Error | Root Cause | Resolution |
|---|---|---|---|
| 1 | `ipywidgets` dependency conflict | Legacy `fastbook 0.0.29` strictly requires `ipywidgets<8`, while initial config had `>=8.0.0`. | Pinned `ipywidgets>=7.7.0,<8.0.0` in [`requirements.txt`](../requirements.txt). |
| 2 | `NameError: name '__version__' is not defined` | `from fastai.vision.all import *` does not export `__version__` to global namespace. | Explicitly imported `import fastai` and accessed `fastai.__version__`. |
| 3 | DuckDuckGo `202 Ratelimit` | DuckDuckGo anti-bot heuristics throttled automated scraping requests. | Migrated to direct **Bing asynchronous image search** with browser headers (zero API keys required, immune to 202 blocks) and DDGS fallback. |
| 4 | Python 3.14 `ResourceWarning` on 403 Forbidden | A few third-party image hosts returned 403; Python 3.14 strictly audits unclosed `tempfile` objects on garbage collection. | Confirmed downloads completed successfully (85 cats, 98 buns) and added `warnings.filterwarnings("ignore", category=ResourceWarning)` to Cell 1. |
| 5 | AI/meme "cat loaf" blend images | Query `'cat loaf photo'` returned surreal hybrid cat-bread images. | Preserved existing images for educational hard-case analysis; added `'real cat photo'` and `'domestic cat photo'` queries with target count of 120 images. |
| 6 | Redundant downloads on re-run | Re-running scraping cells caused repeated network calls and risk of duplicate images. | Added **Smart Caching** (skips download if folder has $\ge$ target images, toggleable via `force_redownload`) and **SHA-256 byte deduplication**. |

---

## 3. Iterations Completed in This Session

- **Iteration 1 (Architecture & Design)**: Resolved the complete design tree via interactive grilling; documented in [`DESIGN.md`](../DESIGN.md).
- **Iteration 2 (Initial Generation)**: Created [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb), [`requirements.txt`](../requirements.txt), and updated [`README.md`](../README.md).
- **Iteration 3 (Multi-Class Refinement)**: Configured categorical Softmax probabilities and horizontal bar plot display.
- **Iteration 4 (Resilient Scraper & Deduplication)**: Integrated Bing search, smart caching thresholds, and cryptographic SHA-256 hashing.
- **Iteration 5 (Real Cat Expansion & Git Cleanup)**: Supplemented dataset with authentic cat queries up to 120 images, integrated [`testing_images/katzebroetchen.jpg`](../testing_images/katzebroetchen.jpg) into Cell 12, and configured [`.gitignore`](../.gitignore).

---

## 4. Context & Next Steps for Session 2

1. **Execute Dataset Validation & Training**:
   - Run **Cell 4** to verify images with `verify_images()` and prune corrupt files.
   - Run **Cells 5 & 6** to build `DataBlock` and preview augmented batch.
   - Run **Cells 7 & 8** to fine-tune ResNet-34 for 4 epochs (1 frozen + 3 unfrozen).
2. **Evaluate Diagnostics**:
   - Run **Cells 9 & 10** to inspect Confusion Matrix and Top Losses (evaluating how the model handles ambiguous cat-loaf edge cases).
3. **Inference Validation**:
   - Run **Cells 11 & 12** to export `models/cat_or_bun_model.pkl` and evaluate `testing_images/katzebroetchen.jpg`.
4. **Future Iteration Roadmap**:
   - Build an interactive Gradio web application for drag-and-drop user uploads (Hugging Face Spaces deployment).

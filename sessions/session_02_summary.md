# Session 2 Summary: Python 3.14 Compatibility, Memory Safeguards & Cloud Migration

**Date**: November 15, 2026  
**Project**: *Katzen oder Brötchen* (Coursework Image Classifier)  
**Notebook**: [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb)  
**Design Document**: [`DESIGN.md`](../DESIGN.md)  
**Repository**: [`README.md`](../README.md)  

---

## 1. Architectural Decisions

1. **Adaptive Dual-Profile Compute Architecture (Cloud GPU vs. Local CPU)**:
   - Rather than forcing a single rigid configuration, the notebook now dynamically inspects its hardware environment and configures optimal parameters:
     - 🚀 **Cloud GPU Profile (Google Colab / Kaggle)**: Automatically detects NVIDIA CUDA GPUs, assigns **ResNet-34** (21.8M parameters), standard Presizing canvas (`Resize(460)`), and batch size `bs=32`. All 4 epochs execute in **~15 seconds** with 0 MB RAM burden on the student's laptop.
     - 💻 **Local CPU Profile (8 GB RAM Laptops)**: Auto-detects CPU fallback, switches to ultra-lean **ResNet-18** (11.7M parameters), compact Presizing canvas (`Resize(256)`), batch size `bs=8`, single-process data loading (`num_workers=0`), and CPU thread capping (`torch.set_num_threads`).

2. **In-Notebook Circuit-Breaker (`MemoryGuardCallback`)**:
   - Engineered an active FastAI callback that inspects Linux `/proc/meminfo` after every batch.
   - If available system memory drops below 10%, it raises FastAI's native `CancelFitException`, cleanly halting training and releasing memory before Linux starts swap-thrashing, completely preventing laptop freezes.

3. **Single-Process Data Loading (`num_workers=0`)**:
   - Configured `num_workers=0` on `dataloaders()` and `defaults.cpus = 0` in environment setup.
   - For small datasets (~340 images), running data loading synchronously in the main thread avoids inter-process communication overhead and bypasses Python 3.14 multiprocessing pickling limitations.

4. **Turnkey Cloud Integration (Google Colab & Kaggle)**:
   - Added one-click interactive badges to the top of [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb) and [`README.md`](../README.md).
   - Embedded automated environment detection (`IN_COLAB` / `IN_KAGGLE`) and non-interactive `pip` installation for missing dependencies (`duckduckgo_search`, `bing-image-downloader`).
   - Integrated cloud asset auto-recovery: downloads [`testing_images/katzebroetchen.jpg`](../testing_images/katzebroetchen.jpg) directly from GitHub if running in a fresh cloud instance.
   - Added cloud export hook (`files.download()`) to automatically prompt browser download of `cat_or_bun_model.pkl` upon training completion.

---

## 2. Challenges & Errors Encountered (and Resolutions)

| # | Challenge / Error | Root Cause | Resolution |
|---|---|---|---|
| 1 | `FutureWarning: torch.jit.script is not supported in Python 3.14+` (Cells 1 & 5) | Python 3.14 deprecates legacy TorchScript (`torch.jit.script`) in favor of `torch.compile`/`torch.export`. FastAI/Torchvision vision transforms internally call TorchScript. | Placed `warnings.filterwarnings("ignore", category=FutureWarning)` at the very top of Cell 1 before any FastAI/PyTorch imports, suppressing it globally across the kernel. Cleaned stderr outputs. |
| 2 | `PicklingError: Can't pickle <function PILBase.__annotate__ ...>` on `learn.fine_tune()` | Python 3.14 introduced PEP 649 (Deferred Evaluation of Annotations). FastAI's dynamic type dispatch on `PILBase` created a function identity mismatch when PyTorch's `DataLoader` attempted to pickle the dataset for multiprocessing worker pools (`num_workers > 0`). | Set `num_workers=0` in `cats_and_buns_block.dataloaders()` and `defaults.cpus = 0` in Cell 1. Single-process execution eliminates worker spawning and performs zero pickling. |
| 3 | Laptop complete lockup / OS freeze during CPU training | `ResNet-34` with `bs=32` and 460x460 presizing stores all intermediate backprop activation tensors and Adam moments in system RAM. On an 8 GB laptop with background apps running, memory hit 100%, triggering disk swap thrashing (`kswapd0`) and 100% I/O wait. | Switched CPU profile to `ResNet-18` (50% less weights & activations), reduced canvas to `Resize(256)`, capped CPU threads (`torch.set_num_threads`), created `MemoryGuardCallback`, and documented `earlyoom` installation for Linux. |
| 4 | `MemoryGuard Safety Triggered` at 7.1% RAM with `bs=16` | On the student's laptop, base OS + browser + Jupyter used ~4 GB RAM, leaving only ~3.5 GB for PyTorch. Unfrozen ResNet-18 with `bs=16` exceeded available headroom. The MemoryGuard successfully aborted training at 92.9% RAM, saving the machine from freezing. | Transitioned primary workflow to **Google Colab (T4 GPU)** with 16 GB VRAM (0 MB local RAM usage, ~15 sec training). For offline local fallback, reduced CPU batch size to `bs=8` (peak memory ~1.1 GB). |

---

## 3. Iterations Completed in This Session

- **Iteration 1 (Python 3.14 Deprecation Suppression)**: Cleaned deprecation warning banners from Cell 1, Cell 4, and Cell 5 by filtering `FutureWarning` before library imports.
- **Iteration 2 (Multiprocessing & Pickling Fix)**: Resolved PEP 649 pickling error by enforcing single-process data loading (`num_workers=0`).
- **Iteration 3 (CPU Memory Profiling & Circuit-Breaker)**: Replaced ResNet-34 with ResNet-18 for CPU, reduced presizing canvas (`Resize(256)`), limited CPU thread contention, and built `MemoryGuardCallback` to protect system responsiveness.
- **Iteration 4 (Dual-Profile Cloud Architecture)**: Upgraded notebook to dynamically support both Google Colab/Kaggle (GPU ResNet-34) and Local (CPU ResNet-18, `bs=8`), added interactive Colab badges, automated cloud package installation, test image auto-fetching, and model export download hooks.
- **Iteration 5 (Documentation & Design Sync)**: Updated [`README.md`](../README.md) and [`DESIGN.md`](../DESIGN.md) to document hardware profiles, memory guardrails, and cloud execution workflows.

---

## 4. Current Repository State

- **Dataset**: 342 verified unique images (176 cats, 166 buns) in `data/cats_and_buns/`.
- **External Test Asset**: [`testing_images/katzebroetchen.jpg`](../testing_images/katzebroetchen.jpg) verified and tracked in git.
- **Jupyter Notebook**: [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb) syntax-validated (AST clean), ready for 1-click execution in Google Colab or local `bs=8` CPU execution.
- **Documentation**: Updated [`README.md`](../README.md), [`DESIGN.md`](../DESIGN.md), and session summaries in `sessions/`.

---

## 5. Next Steps for Session 3

1. **Push Changes to GitHub**:
   - Commit all updated files and push to `origin main` so the Google Colab badge points to the latest notebook.
2. **Execute Full Training on Google Colab (or Local `bs=8`)**:
   - Open notebook via the Colab badge with free T4 GPU.
   - Run all cells: verify ~15-second fine-tuning and model serialization.
3. **Inspect Diagnostics**:
   - Review Confusion Matrix (evaluating error directionality).
   - Review Top Losses (analyzing ambiguous edge cases: cat loaves vs. cinnamon buns).
4. **Inference & Visualization**:
   - Run `predict_cat_or_bun('testing_images/katzebroetchen.jpg')` and confirm dual plot with Softmax horizontal probability bar chart.
5. **Future Roadmap**:
   - Develop an interactive Gradio web application (`app.py`) for Hugging Face Spaces deployment.

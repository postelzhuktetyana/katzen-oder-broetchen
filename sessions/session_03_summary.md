# Session 3 Summary: Colab Model Training, Standalone Tester Notebook & Environment Parity

**Date**: November 15, 2026  
**Project**: *Katzen oder Brötchen* (Coursework Image Classifier)  
**Notebooks**: [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb), [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb)  
**Dependencies**: [`requirements.txt`](../requirements.txt)

---

## 1. Key Outcomes & Architectural Decisions

1. **Cloud Model Training Completed (Google Colab)**:
   - Trained and exported `models/cat_or_bun_model.pkl` (`83.43 MB`, `Sequential` architecture, vocabulary `['bun', 'cat']`) on Google Colab (`python=3.13.16`, `fastai==2.8.12`, `torch==2.14.1+cu130`, `plum-dispatch==2.9.0`, `numpy==2.4.6`).

2. **Standalone Inference & Testing Notebook ([`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb))**:
   - Created a dedicated, decoupled inference notebook to test `models/cat_or_bun_model.pkl` locally against arbitrary filesystem or browser-uploaded images.
   - Implemented multiple testing modes:
     - Quick sanity check on [`testing_images/katzebroetchen.jpg`](../testing_images/katzebroetchen.jpg).
     - Interactive filesystem dropdown picker + custom path input.
     - Browser drag-and-drop uploader (`ipywidgets.FileUpload`, compatible with both `ipywidgets` 7.x and 8.x).
     - Batch directory evaluation gallery grid.

3. **Environment Parity via Mamba (`Python 3.13`) Instead of Monkey-Patches**:
   - Rejected fragile runtime pickle monkey-patches in favor of matching the local runtime to Google Colab's exact Python and package versions using `mamba` (`katzen` environment, registered via `ipykernel` so existing JupyterLab did not need reinstallation).
   - Cleaned all Python 3.14 workarounds (`_patch_plum_slots()`, `defaults.cpus = 0`, blanket warning suppressions) out of [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) and [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb), keeping only a targeted filter for PyTorch's internal `torch.jit.script` `FutureWarning` and CPU-memory safeguards (`NUM_WORKERS = 0` on CPU, `MemoryGuardCallback`).
   - Updated [`requirements.txt`](../requirements.txt) with Colab-aligned pins (`fastai>=2.8.12`, `numpy>=2.0.0`, `plum-dispatch==2.9.0`).

---

## 2. Challenges & Errors Encountered (and Resolutions)

| # | Challenge / Error | Root Cause | Resolution |
|---|---|---|---|
| 1 | `AttributeError: 'Resolver' object has no attribute '__dict__'` when loading `.pkl` in Python 3.14 (`andre_env`) | Pickle deserialization mismatch between Colab's Python 3.13 / `plum-dispatch` export and local Python 3.14 runtime; monkey-patch attempts proved brittle and added unnecessary complexity. | Removed monkey-patches completely and switched to matching Colab's environment via `mamba`. |
| 2 | `UnboundLocalError: local variable 'res' referenced before assignment` in `fastai/learner.py:466` on Python 3.10 (`katzen` v1) | Inspecting `models/cat_or_bun_model.pkl` revealed `pathlib._local` (introduced in Python 3.13) and `numpy._core` (NumPy 2.x). Python 3.10 lacked `pathlib._local`, and FastAI's `except ImportError` block swallowed non-`fastcore` import errors, leaving `res` undefined. | Recreated the `katzen` `mamba` environment with **Python 3.13** (`python=3.13`) and aligned [`requirements.txt`](../requirements.txt) to Colab's versions. Model now loads cleanly with plain `load_learner()`. |
| 3 | `FutureWarning: torch.jit.script is deprecated` on import | Emitted internally by PyTorch 2.x when FastAI/Torchvision loads on Python 3.13 as well as 3.14. | Added a single targeted `warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.jit\.script.*")` in Cell 2 of both notebooks. |
| 4 | `UserWarning: load_learner uses Python's insecure pickle module...` | Standard built-in FastAI security reminder when loading `.pkl` artifacts. | Safe for self-trained local model (`models/cat_or_bun_model.pkl`); left visible as normal FastAI output. |

---

## 3. Current Repository & Verification State

- **Model Loading Verified**: [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) successfully loads `models/cat_or_bun_model.pkl` (`83.43 MB`, `Architecture: Sequential`, `Vocabulary: ['bun', 'cat']`) under the `Python 3.13 (katzen)` kernel and runs end-to-end.
- **Notebooks Cleaned**: Both [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) and [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb) are free of Python 3.14 monkey-patches.

---

## 4. IMPORTANT: Critical Issue Observed — Next Steps for Session 4

> [!IMPORTANT]
> **Classification Behavior Issue (Not Yet Investigated — Starting Point for Session 4)**
>
> Testing in [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) indicates the classifier is likely **working incorrectly** for independent / out-of-distribution inputs:
> 1. **Not independent classification**: It is not independently scoring "cat" vs. "bun" presence (`Cat + Bun` percentages are currently coupled and always sum to **100%**).
> 2. **Unknown / unrelated images collapse to 100% Cat**: When testing an unknown image (something that is neither a cat nor a bun), the model outputs **100% Cat**.
>
> *No investigation was performed in Session 3 per user instruction. Session 4 will begin directly by diagnosing and addressing this behavior.*

### Planned Focus for Session 4
1. **Investigate the Classification & Output Formulation**:
   - Diagnose why unknown images predict ~100% Cat and evaluate the difference between closed-world single-label Softmax (`Cat + Bun = 100%`) vs. multi-label / independent classification (e.g., Sigmoid / `MultiCategoryBlock`, adding an `other`/`neither` class, or confidence/OOD handling) and/or dataset/model bias.
2. **Git Commit & Sync**:
   - Review pending changes (`cat_or_bun_classifier.ipynb`, `cat_or_bun_tester.ipynb`, `requirements.txt`, `README.md`, `DESIGN.md`, `sessions/`) and commit once the classification fix is verified.

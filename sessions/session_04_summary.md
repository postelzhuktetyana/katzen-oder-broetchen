# Session 4 Summary: Independent Multi-Label Classification (Sigmoid), Negative Class & Clean-Retrain Loop

**Date**: November 21, 2026  
**Project**: *Katzen oder Brötchen* (Coursework Image Classifier)  
**Notebooks**: [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb), [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb)  
**Documentation**: [`DESIGN.md`](../DESIGN.md), [`README.md`](../README.md)

---

## 1. Key Outcomes & Architectural Decisions

1. **Root-Cause Diagnosis of Closed-World Softmax Behavior**:
   - Diagnosed why the Session 3 model coupled `Cat + Bun = 100%` and collapsed unrelated images to `100% Cat`: `CategoryBlock` uses `CrossEntropyLossFlat` + **Softmax**, which is mutually exclusive and only compares logit differences $z_{\text{cat}} - z_{\text{bun}}$.
   - Clarified the transfer-learning nuance: while ResNet's pretrained ImageNet backbone retains visual feature detectors for 1,000 classes, `vision_learner()` **replaces** the 1,000-class head with a randomly initialized 2-output head (`bun`, `cat`). Without negative (`[]`) examples, the new head never learns to suppress both outputs on out-of-distribution inputs.

2. **Migration to Independent Multi-Label Classification (`MultiCategoryBlock` + Sigmoid)**:
   - Replaced `CategoryBlock` with `MultiCategoryBlock` (`BCEWithLogitsLossFlat` + independent Sigmoid per output) and a named, pickle-safe `get_multi_labels()` mapper (`cat/` → `['cat']`, `bun/` → `['bun']`, `other/` → `[]`).
   - Added `'other'` negative queries in Cell 6 (`'dog photo'`, `'plush toy photo'`, `'car photo'`, `'croissant bread photo'`), intentionally excluding `'cat loaf meme'` and `'drawn cats'` per user direction so `katzebroetchen.jpg` remains an unbiased edge case.
   - Switched training metric from `error_rate` to `partial(accuracy_multi, thresh=0.5)`, where `thresh=0.5` represents a `50%` independent probability cutoff per category, yielding four outcomes: `CAT`, `BUN`, `BOTH (Cat & Bun)`, or `NEITHER (Other / Unknown)`.

3. **Multi-Label Diagnostics & Iterative Clean-Retrain Workflow (*Train → Clean → Retrain*)**:
   - Replaced single-label `ClassificationInterpretation` (incompatible with `MultiCategoryBlock`) with per-class 2×2 confusion matrices (TP/FP/FN/TN, Precision, Recall) using `learn.get_preds()`.
   - Updated Top Losses inspection to use `Interpretation.from_learner(learn)` printing exact file paths, true labels, independent probabilities, and loss values.
   - Added an optional **Data Cleaning & Clean Retrain** section featuring both FastAI's interactive `ImageClassifierCleaner(learn)` widget and an explicit `files_to_remove` list that prunes noisy scraped images and retrains cleanly.

4. **Updated Inference Visualizations (`cat_or_bun_classifier.ipynb` & `cat_or_bun_tester.ipynb`)**:
   - Updated `predict_cat_or_bun`, `predict_and_display`, and `test_directory_gallery` to display independent Sigmoid percentages, a `50%` dashed decision-threshold line, and `CAT` / `BUN` / `BOTH` / `NEITHER` verdicts (while keeping `cat_or_bun_tester.ipynb` backward-compatible with legacy Softmax `.pkl` files).

5. **Presentation & Report Case Study (Closed-World Softmax vs. Independent Sigmoid)**:
   - *Symptom*: `Cat + Bun` always summed to `100%`, and unrelated images (dogs, cars, etc.) collapsed to `~100% Cat`.
   - *Root Cause*: `CategoryBlock` (`CrossEntropyLossFlat` + Softmax) enforces a mutually exclusive 2-class choice and amplifies small logit differences $z_{\text{cat}} - z_{\text{bun}}$; meanwhile `vision_learner()` replaces ResNet's 1,000-class ImageNet head with a new 2-output head that had never seen negative (`[0, 0]`) examples.
   - *Rejected Alternatives*:
     1. *Post-hoc confidence thresholding on Softmax*: Still leaves `Cat` and `Bun` coupled (`Cat + Bun = 100%`) and Softmax remains overconfident on OOD inputs.
     2. *3-class Softmax (`cat`, `bun`, `other`)*: Handles unknown inputs, but `Cat`, `Bun`, and `Other` still sum to `100%` (not independent; cannot express `BOTH`).
     3. *`MultiCategoryBlock` without `other/` negatives*: Outputs are Sigmoid, but the head never sees `[0, 0]` during training so OOD inputs still trigger high confidence.
   - *Final Fix*: `MultiCategoryBlock` (`BCEWithLogitsLossFlat` + independent Sigmoid, `thresh=0.5`) **plus** `other/` negatives (`[]`), saved as `models/cat_or_bun_multilabel_model.pkl` while keeping `models/cat_or_bun_model.pkl` intact for side-by-side comparison in the report and Session 5 tester updates.

---

## 2. Challenges & Errors Encountered (and Resolutions)

| # | Challenge / Error | Root Cause | Resolution |
|---|---|---|---|
| 1 | Closed-world coupling (`Cat + Bun = 100%`) & OOD collapse (`100% Cat` on unknown images) | `CategoryBlock` applies Softmax over 2 logits; unfamiliar textures with slightly higher `cat` logit than `bun` logit are amplified to ~100% Cat. | Switched to `MultiCategoryBlock` (independent Sigmoid via `BCEWithLogitsLossFlat`) + `other/` negative images labeled `[]`. |
| 2 | Risk of `other/` polluting ambiguous cat-loaf behavior | Labeling `"cat loaf meme"` as `other` (`[]`) would teach the model that cat-bread hybrids are *neither* cat nor bun, hurting `katzebroetchen.jpg` and curled-up cats. | Excluded `"cat loaf meme"` and `"drawn cats"` from `other/`; used `dogs`, `plush toys`, `cars`, and `croissant/bread` as clean negatives. |
| 3 | `ClassificationInterpretation` & `error_rate` incompatibility with `MultiCategoryBlock` | FastAI's `error_rate` and `ClassificationInterpretation.plot_confusion_matrix()` assume single-label `CategoryBlock` targets. | Switched metric to `partial(accuracy_multi, thresh=0.5)`, built per-class 2×2 confusion matrices via `learn.get_preds()`, and used `Interpretation.from_learner(learn)` for top losses. |
| 4 | Lambda `get_y` pickling risk on `learn.export()` | Anonymous `lambda` functions in `DataBlock(get_y=...)` can fail during cross-environment pickle export/load. | Defined a named top-level function `get_multi_labels(img_path)`. |
| 5 | **Admin / Versioning**: Overwriting `models/cat_or_bun_model.pkl` would erase the Softmax baseline | Reusing the same export filename prevents side-by-side comparison of Softmax vs. Sigmoid in the report & presentation. | Changed new export path to `models/cat_or_bun_multilabel_model.pkl`, keeping `models/cat_or_bun_model.pkl` intact for Session 5 dual-model testing. |
| 6 | **Admin / Process**: `AGENTS.md` only mentioned technical bugs/model issues, not administrative/workflow issues | Without an explicit rule, workflow/scoping/versioning decisions could be omitted from session logs. | Updated [`AGENTS.md`](../AGENTS.md) Section 1.2 to explicitly require logging technical, modeling, environment, and administrative/workflow issues. |

---

## 3. Current Repository & Verification State

- **Files Modified**:
  - [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb) (28 cells, all code cells AST-validated).
  - [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) (15 cells, all code cells AST-validated).
  - [`DESIGN.md`](../DESIGN.md), [`README.md`](../README.md), [`AGENTS.md`](../AGENTS.md), and [`sessions/session_04_summary.md`](session_04_summary.md).
- **Not Yet Executed**:
  - Notebook retraining (`other/` image download, `MultiCategoryBlock` fine-tuning, optional cleaning, and exporting `models/cat_or_bun_multilabel_model.pkl`) has not yet been run on Google Colab / local `katzen` kernel.
- **Model Preservation for Report Comparison**:
  - `models/cat_or_bun_model.pkl` (Softmax baseline) is preserved; the new model exports to `models/cat_or_bun_multilabel_model.pkl`.

---

## 4. Next Steps / Open Issues

1. **Retrain & Export New Model (Google Colab T4 GPU recommended)**:
   - Run [`cat_or_bun_classifier.ipynb`](../cat_or_bun_classifier.ipynb) to download `data/cats_and_buns/other/` (~120 images), train the `MultiCategoryBlock` model, inspect Top Losses (optionally clean noisy images and retrain), and export `models/cat_or_bun_multilabel_model.pkl` (keeping `models/cat_or_bun_model.pkl` untouched).
2. **Session 5 — Dual-Model Comparison in [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb)**:
   - Update [`cat_or_bun_tester.ipynb`](../cat_or_bun_tester.ipynb) to load both `models/cat_or_bun_model.pkl` (Softmax baseline) and `models/cat_or_bun_multilabel_model.pkl` (independent Sigmoid) and display side-by-side predictions for the report & presentation.
3. **Git Commit & Sync**:
   - Commit all updated notebooks, docs, and session summaries to `origin main`.

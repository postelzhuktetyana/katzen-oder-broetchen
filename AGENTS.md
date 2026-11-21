# Project Rules: Katzen oder Brötchen

This is an **educational coursework project** (FastAI / PyTorch image classifier) with three interconnected deliverables:
1. **Program**: Working, well-documented code and notebooks ([`cat_or_bun_classifier.ipynb`](cat_or_bun_classifier.ipynb), [`cat_or_bun_tester.ipynb`](cat_or_bun_tester.ipynb)).
2. **Report**: Technical coursework report grounded in [`DESIGN.md`](DESIGN.md) and session logs.
3. **Presentation**: Presentation analyzing not only the final solution, but specifically **the challenges, errors, trade-offs, and iterations encountered during development**.

---

## 1. Session Tracking Protocol (`sessions/`)

We maintain a chronological log of every development session in `sessions/session_NN_summary.md` (zero-padded, e.g., `session_01_summary.md`, `session_02_summary.md`, `session_03_summary.md`, ...).

- **Start of Every Session**:
  1. Check `sessions/` and read the latest `session_NN_summary.md` before proposing or making changes.
  2. Resume from the documented **Current State** and **Next Steps / Open Issues** unless the user directs otherwise.
- **During & End of Every Session**:
  1. Create (or update) the current session's `sessions/session_NN_summary.md` (incrementing `NN` for a new session).
  2. Follow the established structure:
     - **Header**: Title, Date, Project, Notebooks/Files touched.
     - **1. Key Outcomes & Architectural Decisions**: What was built, changed, or decided (and *why*).
     - **2. Challenges & Errors Encountered (and Resolutions)**: A markdown table with columns `| # | Challenge / Error | Root Cause | Resolution |`. **Never omit or gloss over technical, modeling, environment, or administrative/workflow issues** (bugs, failed experiments, environment mismatches, model failure modes, dataset scoping, artifact versioning, or cloud/local sync friction) — these are primary material for the final report and presentation.
     - **3. Current Repository & Verification State**: Exact files changed, what was actually run/verified, and with which environment/kernel.
     - **4. Next Steps / Open Issues**: Clear handoff for the next session.

---

## 2. Educational & Documentation Standards

- **Pedagogical Clarity**:
  - In notebooks, precede code blocks with structured Markdown explanations (*Practical Goal*, *Core ML Concept*, *Implementation Details*) and keep inline comments clear and educational.
  - Preserve interesting edge cases and ambiguous/hard examples (e.g., `testing_images/katzebroetchen.jpg`) for error analysis and presentation slides.
- **Documentation Sync**:
  - Keep [`DESIGN.md`](DESIGN.md), [`README.md`](README.md), and [`requirements.txt`](requirements.txt) aligned whenever architecture, dependencies, or execution workflows change.
  - Note that [`README.md`](README.md) is written in Ukrainian (with English technical terms); preserve its language unless asked to change it.
- **Report & Presentation Readiness**:
  - When solving problems, explicitly note the *symptom*, *root cause*, *rejected alternatives* (and why they were rejected), and *final fix* so they can be directly referenced in the report and presentation.

---

## 3. Technical & Repository Conventions

- **Environment**: Prefer clean environment parity (e.g., matching Colab's Python 3.13 / FastAI / PyTorch versions via `mamba` `katzen` kernel) over brittle runtime monkey-patches.
- **Hardware & Memory Safety**: Keep dual-profile support (Cloud GPU vs. Local CPU) and memory safeguards (`MemoryGuardCallback`, `num_workers=0` on CPU) intact.
- **Git Hygiene**: Never commit large binary artifacts (`data/`, `models/*.pkl`, `.venv/`). Keep [`.gitignore`](.gitignore) respected and only track lightweight test assets (e.g., `testing_images/`).
- **Truthful Reporting**: Only report tests, notebook runs, or metrics as verified if they were actually executed and observed in the current session.

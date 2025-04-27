## 2024-07-29: Phase 2 - Execution & Debugging (New Instance)

- Cloned repo, set up `venv`, installed requirements.
- Debugged multiple issues culminating in a persistent hang within `trainer.train()` when using the custom `EntangledModel` wrapper.
- **Solution:** Implemented a manual training loop using `accelerate`, bypassing `trainer.train()`. Debugged subsequent errors (`NameError`, `TypeError`, `RuntimeError`) within the manual loop and forward pass.
- Manual loop successfully completed 10 steps with full model logic (incl. KL loss calc) and saved components.
- **Status:** Phase 2 training script is technically functional. Next steps involve longer training runs on appropriate datasets and implementing evaluation.

## 2024-07-29: Phase 2 - Training Run 1

- Implemented manual training loop using `accelerate` to bypass `Trainer` hang.
- Successfully ran manual loop for 500 steps on `guanaco-1k` placeholder dataset.
- Confirmed LM loss decreased and KL loss was stable, indicating functional learning.

## Plan for Phase 2 - Full Training Run

1.  **Dataset:** Switch from `guanaco-1k` to a larger instruction dataset (e.g., `HuggingFaceH4/databricks-dolly-15k` proposed).
2.  **Duration:** Increase training steps for meaningful learning (e.g., `max_steps = 10000` proposed for Dolly dataset).
3.  **Hardware:** Attempt multi-GPU training (`device_map="auto"`) with the manual loop, reverting to single GPU if issues arise.
4.  **Goal:** Train the variational head and projection matrix, collect loss curves, and prepare for Phase 2 evaluation.

## 2024-07-29: Phase 2 - Code Committed

- Committed the changes for the manual Accelerate training loop implementation (`ab22b5a`).
- The commit reflects the successful 500-step run on `guanaco-1k` and the functional state of the training script. 
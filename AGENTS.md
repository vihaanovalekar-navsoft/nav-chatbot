# Evaluation Runs

- Save a per-case checkpoint after each completed answer and judge response, before the next case starts.
- Resume from the checkpoint after any failure. Do not rerun completed cases.
- Do not start a paid model evaluation until checkpointing and resume are implemented and tested.
- Check the provider's remaining quota before starting. Stop on quota exhaustion and resume after reset.
- A log is not a checkpoint. It cannot rebuild answers, judge reasons, sources, or token counts.

# Validation record

15 September 2026:

- Both Python entry points pass syntax checks.
- The app imports without requiring a local model directory; empty input returns a consistent result.
- The published Hugging Face model loaded and accepted a short synthetic sentence, but returned incorrect spans. The dataset and checkpoint label mappings differ.
- The training label map and subtoken continuation logic are corrected in source. Retraining and a held-out evaluation remain required before using an updated checkpoint.

No updated F1 score is claimed. The existing public model weights and hosted demo have not been updated.

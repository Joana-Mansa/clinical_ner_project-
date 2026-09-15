# Validation record

15 September 2026:

- Both Python files parse after the malformed F1 statement was repaired.
- The app imports without requiring a local model directory; empty input returns a consistent result.
- The published Hugging Face model loaded and accepted a short synthetic sentence, but returned incorrect spans. This prompted inspection of the dataset/checkpoint label mappings, which differ.
- The training label map and subtoken continuation logic are corrected in source. Retraining and a held-out evaluation remain required before considering the checkpoint repaired.

No updated F1 score is claimed. The GitHub source and local interface can be repaired independently of the existing public model weights and hosted demo.

# Inference and training

## Inference

`app.py` loads `JoanaOA/clinical-ner-biobert-bc5cdr` lazily. `CLINICAL_NER_MODEL` may instead name a local Transformers checkpoint directory containing the model, config and tokenizer files. This fixes the previous startup dependency on an absent local training output.

The app aggregates token spans, groups them by CHEMICAL/DISEASE, filters by confidence and displays highlighted text, a table and JSON. Empty input produces an instruction rather than a model call. Offsets refer to the supplied input string. Long-text handling depends on the model/tokenizer window; the current app does not implement document chunking.

The server binds to loopback by default and does not create a public Gradio tunnel. The hosted Hugging Face demo is a separate deployment; a GitHub push does not automatically update it.

## Fine-tuning

The percent-cell tutorial is a Python script, not a pre-executed notebook. It runs data exploration and training from top to bottom, uses `tner/bc5cdr`, aligns labels to tokenized inputs, evaluates on the validation split during training, saves the best model, then evaluates the test split. The syntax error in the final F1 print statement has been fixed and TrainingArguments uses `eval_strategy` for the supported Transformers versions.

Outputs include checkpoint directories and example extraction files. Training is not needed to run the public checkpoint in the app. No automatic model publishing is enabled.

## Interpreting labels

CHEMICAL includes chemical entities rather than only prescribed medicines; DISEASE follows BC5CDR’s annotation schema. Entity recognition does not infer a diagnosis or a relationship between a chemical and disease. Text from a new domain needs a separate evaluation before performance claims are made.

Original references: [BC5CDR paper](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414), [BioBERT paper](https://arxiv.org/abs/1901.08746).

## Dataset-label correction

The authoritative [dataset label file](https://huggingface.co/datasets/tner/bc5cdr/blob/main/dataset/label.json) uses O=0, B-Chemical=1, B-Disease=2, I-Disease=3, I-Chemical=4. The earlier tutorial/checkpoint used a different order for IDs 2–4 and hard-coded continuation IDs based on that order.

The corrected tutorial keeps the dataset ID order and derives continuation tags by name. Existing weights need retraining and fresh held-out evaluation; changing only the checkpoint metadata is insufficient. The local app selects a newly saved `clinical_ner_model/final` when available, or the public checkpoint with a visible status notice. This maintenance change does not modify the separate Hugging Face deployment.

# 🧬 Biomedical Named Entity Recognition

BioBERT fine-tuned on BC5CDR to identify **CHEMICAL** and **DISEASE** spans in biomedical text. By Joana Owusu-Appiah.

🤗 [Published model](https://huggingface.co/JoanaOA/clinical-ner-biobert-bc5cdr) · ▶️ [Hosted demo](https://huggingface.co/spaces/JoanaOA/clinical-ner-demo)

## Model status

The published checkpoint’s label IDs differ from `tner/bc5cdr`’s source mapping, and an inference check produced incorrect spans. The tutorial now preserves the dataset mapping and aligns continuation tokens correctly. **The published checkpoint has not been retrained or replaced.** A metadata-only rename cannot undo training targets; use a newly trained checkpoint and evaluate it before quoting performance.

## Run the local app

```bash
git clone https://github.com/Joana-Mansa/clinical_ner_project-.git
cd clinical_ner_project-
python -m venv .venv
source .venv/bin/activate
python -m pip install --use-pep517 -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`. The model downloads from Joana’s Hugging Face repository on first inference and is cached. Training is not required. To use your own local checkpoint:

```bash
CLINICAL_NER_MODEL=./clinical_ner_model/final python app.py
```

## Repository map

| File | Purpose |
|---|---|
| `app.py` | Local Gradio inference, confidence filtering and entity display |
| `clinical_ner_tutorial.py` | Data exploration, fine-tuning, held-out evaluation and example extraction |
| `clinical_extractions.json` | Previously generated example extraction artifact |
| `requirements.txt` | App and training dependencies |

## Training and evaluation

`python clinical_ner_tutorial.py` executes the full tutorial and trains a model. It downloads BC5CDR through Hugging Face, uses train/validation/test splits, and writes `clinical_ner_model/final`. A GPU is recommended; the runtime depends on hardware and data. Publishing a model is separate from this script.

📖 [Workflow, labels and limitations](docs/workflow.md) · [Validation record](docs/validation.md)

BC5CDR contains biomedical literature annotations. Good extraction on these examples does not establish performance on hospital notes, medication dosing, relations or diagnosis. Confidence values are model scores, not calibrated clinical probabilities.

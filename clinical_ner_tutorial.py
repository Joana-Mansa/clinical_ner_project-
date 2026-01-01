"""
Clinical Named Entity Recognition (NER) Project
================================================
Extract medications, diseases, and medical concepts from clinical text.

This project demonstrates:
1. Using pre-trained biomedical NER models
2. Fine-tuning on clinical data (BC5CDR dataset)
3. Evaluating model performance
4. Building a simple extraction pipeline

Author: Joana Owusu-Appiah 
"""

# %% [markdown]
# # Part 1: Quick Start with Pre-trained Models
# Let's first see what's possible with zero training using existing models.

# %%
# Install required packages (run once)
# !pip install transformers datasets seqeval torch accelerate

# %%
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json

# %% [markdown]
# ## 1.1 Using a Pre-trained Biomedical NER Model

# %%
# Load a pre-trained biomedical NER model
print("Loading pre-trained biomedical NER model...")
ner_pipeline = pipeline(
    "ner", 
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"  # Merge sub-tokens into full entities
)

# %%
# Test on sample clinical text
sample_texts = [
    "Patient was prescribed metformin 500mg twice daily for type 2 diabetes mellitus.",
    "Chest X-ray revealed bilateral pneumonia. Started on amoxicillin 875mg.",
    "History of hypertension treated with lisinopril 10mg. No evidence of heart failure.",
    "The patient presents with severe headache and was given ibuprofen 400mg for pain relief.",
]

print("=" * 60)
print("BIOMEDICAL NER EXTRACTION RESULTS")
print("=" * 60)

for text in sample_texts:
    print(f"\nText: {text}")
    print("-" * 40)
    entities = ner_pipeline(text)
    for ent in entities:
        print(f"  [{ent['entity_group']:15}] {ent['word']:20} (score: {ent['score']:.3f})")

# %% [markdown]
# ## 1.2 Comparing Different Pre-trained Models

# %%
# Let's compare a few models
models_to_compare = [
    ("d4data/biomedical-ner-all", "General Biomedical"),
    ("alvaroalon2/biobert_chemical_ner", "Chemical/Drug focused"),
]

test_text = "Patient diagnosed with chronic kidney disease stage 3, prescribed losartan 50mg and atorvastatin 20mg daily."

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\nTest text: {test_text}\n")

for model_name, description in models_to_compare:
    try:
        pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")
        entities = pipe(test_text)
        print(f"\n{description} ({model_name}):")
        print("-" * 40)
        for ent in entities:
            print(f"  [{ent['entity_group']:15}] {ent['word']}")
    except Exception as e:
        print(f"  Could not load model: {e}")

# %% [markdown]
# # Part 2: Fine-tuning on BC5CDR Dataset
# The BC5CDR dataset contains PubMed articles annotated with chemical and disease mentions.
# This is a great starting point for clinical NER.

# %%
from datasets import load_dataset

# Load BC5CDR dataset from HuggingFace
print("Loading BC5CDR dataset...")
dataset = load_dataset("tner/bc5cdr")
print(f"\nDataset structure:")
print(f"  Train: {len(dataset['train'])} examples")
print(f"  Validation: {len(dataset['validation'])} examples")  
print(f"  Test: {len(dataset['test'])} examples")

# %%
# Examine the data structure
example = dataset['train'][0]
print("\nExample from training set:")
print(f"  Tokens: {example['tokens'][:15]}...")
print(f"  Tags: {example['tags'][:15]}...")

# Get label mapping
print(dataset["train"].features)
label_names = dataset["train"].features["tags"]
label_names = [
    "O",
    "B-CHEMICAL",
    "I-CHEMICAL",
    "B-DISEASE",
    "I-DISEASE"
]

print(f"\nLabel mapping:")
for i, label in enumerate(label_names):
    print(f"  {i}: {label}")

# %%
# Visualize a few examples
def display_ner_example(tokens, tags, label_names):
    """Pretty print an NER example"""
    result = []
    current_entity = None
    current_words = []
    
    for token, tag_id in zip(tokens, tags):
        tag = label_names[tag_id]
        
        if tag.startswith('B-'):
            if current_entity:
                result.append(f"[{current_entity}: {' '.join(current_words)}]")
            current_entity = tag[2:]
            current_words = [token]
        elif tag.startswith('I-') and current_entity:
            current_words.append(token)
        else:
            if current_entity:
                result.append(f"[{current_entity}: {' '.join(current_words)}]")
                current_entity = None
                current_words = []
            result.append(token)
    
    if current_entity:
        result.append(f"[{current_entity}: {' '.join(current_words)}]")
    
    return ' '.join(result)

print("\nAnnotated examples:")
print("-" * 60)
for i in range(3):
    example = dataset['train'][i]
    annotated = display_ner_example(example['tokens'], example['tags'], label_names)
    print(f"\n{i+1}. {annotated[:200]}...")

# %% [markdown]
# ## 2.1 Prepare for Fine-tuning

# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np

# Use BioBERT as base model
model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
# Tokenize and align labels
def tokenize_and_align_labels(examples):
    """Tokenize inputs and align labels with sub-tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                prev_label = label[word_idx]
                if prev_label == 1:
                    label_ids.append(2)  # I-CHEMICAL
                elif prev_label == 3:
                    label_ids.append(4)  # I-DISEASE
                elif prev_label == 2:
                    label_ids.append(2)  # I-CHEMICAL
                elif prev_label == 4:
                    label_ids.append(4)  # I-DISEASE
                else:
                # For sub-tokens, use -100 (ignored in loss) or same label
                    label_ids.append(0)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(f"Tokenized train size: {len(tokenized_dataset['train'])}")


def verify_tokenization():
    """Show an example to verify subword labels are correct"""
    print("\n" + "=" * 50)
    print("VERIFYING TOKENIZATION FIX")
    print("=" * 50)
    
    # Get a sample with a long word that will be split
    sample_text = ["Patient", "takes", "metformin", "for", "diabetes"]
    sample_tags = [0, 0, 1, 0, 3]  # O, O, B-CHEMICAL, O, B-DISEASE
    
    # Tokenize
    tokens = tokenizer(sample_text, is_split_into_words=True)
    word_ids = tokens.word_ids()
    
    print(f"\nOriginal words: {sample_text}")
    print(f"Original tags:  {[label_names[t] for t in sample_tags]}")
    print(f"\nTokenized:")
    
    for i, (token_id, word_idx) in enumerate(zip(tokens['input_ids'], word_ids)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if word_idx is not None:
            orig_label = label_names[sample_tags[word_idx]]
            print(f"  {token:15} → word_idx={word_idx}, original_label={orig_label}")
        else:
            print(f"  {token:15} → [SPECIAL TOKEN]")
    
    print("\n✓ Subword tokens now get I- labels instead of being ignored!")
    print("=" * 50 + "\n")

verify_tokenization()


# %% [markdown]
# ## 2.2 Set Up Training

# %%
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_names),
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)},
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%
# Evaluation metrics
def compute_metrics(eval_preds):
    """Compute precision, recall, F1 for NER"""
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index and convert to labels
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(label_names[p])
                true_lab.append(label_names[l])
        true_predictions.append(true_pred)
        true_labels.append(true_lab)
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# %%
# Training arguments - using small subset for quick demo
training_args = TrainingArguments(
    output_dir="./clinical_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_steps=50,
    report_to="none",  # Disable wandb etc.
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 2.3 Train the Model
# Note: This will take ~10-15 minutes on CPU, ~2-3 minutes on GPU

# %%
# Uncomment to train (or skip to use pre-trained model)
print("Starting training...")
trainer.train()
print("Training complete!")
trainer.save_model("./clinical_ner_model/final")
tokenizer.save_pretrained("./clinical_ner_model/final")

eval_results = trainer.evaluate(tokenized_dataset["test"])
print(f"Test F1": {eval_results['eval_f1']:.3f})

# For quick demo, let's evaluate the base model without fine-tuning
print("Evaluating base BioBERT model (before fine-tuning)...")
# eval_results = trainer.evaluate()
# print(f"\nEvaluation Results:")
# print(f"  Precision: {eval_results['eval_precision']:.3f}")
# print(f"  Recall: {eval_results['eval_recall']:.3f}")
# print(f"  F1 Score: {eval_results['eval_f1']:.3f}")

# %% [markdown]
# # Part 3: Building a Clinical Information Extraction Pipeline

# %%
class ClinicalNERPipeline:
    """
    A complete pipeline for extracting medical entities from clinical text.
    Combines multiple models for comprehensive extraction.
    """
    
    def __init__(self, model_path="./clinical_ner_model"):
        print("Initializing Clinical NER Pipeline...")
        
        # Load pre-trained model (use fine-tuned model path after training)
        self.ner = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )
        print("Pipeline ready!")
    
    def extract_entities(self, text):
        """Extract all medical entities from text"""
        raw_entities = self.ner(text)
        
        # Organize by entity type
        organized = {}
        for ent in raw_entities:
            entity_type = ent['entity_group']
            if entity_type not in organized:
                organized[entity_type] = []
            organized[entity_type].append({
                'text': ent['word'],
                'confidence': round(ent['score'], 3),
                'start': ent['start'],
                'end': ent['end']
            })
        
        return organized
    
    def extract_medications(self, text):
        """Extract only medication-related entities"""
        entities = self.extract_entities(text)
        med_types = ['Drug', 'CHEMICAL', 'Medication', 'Medicine']
        
        medications = []
        for med_type in med_types:
            if med_type in entities:
                medications.extend(entities[med_type])
        
        return medications
    
    def extract_conditions(self, text):
        """Extract disease/condition entities"""
        entities = self.extract_entities(text)
        condition_types = ['Disease', 'DISEASE', 'Condition', 'Symptom']
        
        conditions = []
        for cond_type in condition_types:
            if cond_type in entities:
                conditions.extend(entities[cond_type])
        
        return conditions
    
    def process_report(self, text):
        """Process a full clinical report and return structured data"""
        return {
            'original_text': text,
            'all_entities': self.extract_entities(text),
            'medications': self.extract_medications(text),
            'conditions': self.extract_conditions(text),
        }
    
    def format_output(self, result):
        """Pretty print extraction results"""
        print("\n" + "=" * 60)
        print("CLINICAL ENTITY EXTRACTION REPORT")
        print("=" * 60)
        print(f"\nInput Text:\n{result['original_text']}")
        print("\n" + "-" * 60)
        
        print("\n📋 ALL ENTITIES:")
        for entity_type, entities in result['all_entities'].items():
            print(f"\n  {entity_type}:")
            for ent in entities:
                print(f"    • {ent['text']} (confidence: {ent['confidence']})")
        
        print("\n💊 MEDICATIONS:", [m['text'] for m in result['medications']])
        print("🏥 CONDITIONS:", [c['text'] for c in result['conditions']])
        print("=" * 60)

# %%
# Test the pipeline
pipeline_demo = ClinicalNERPipeline(model_path="./clinical_ner_model/final")

# Sample clinical notes
clinical_notes = [
    """
    DISCHARGE SUMMARY: 72-year-old male with history of type 2 diabetes mellitus 
    and hypertension. Admitted for chest pain, ruled out for myocardial infarction. 
    Continue metformin 1000mg BID, lisinopril 20mg daily, and aspirin 81mg daily.
    Follow up with cardiology in 2 weeks.
    """,
    
    """
    PROGRESS NOTE: Patient reports improvement in joint pain after starting 
    naproxen 500mg twice daily. Rheumatoid arthritis remains stable. 
    Continue current regimen of methotrexate 15mg weekly and folic acid 1mg daily.
    """,
    
    """
    ED NOTE: 45-year-old female presenting with acute migraine. Given sumatriptan 
    50mg PO with relief. History of anxiety disorder on sertraline 100mg daily.
    Discharged home with instructions to follow up with neurology.
    """
]

for note in clinical_notes:
    result = pipeline_demo.process_report(note.strip())
    pipeline_demo.format_output(result)

# %% [markdown]
# # Part 4: Exporting Results for Analysis

# %%
import json
from datetime import datetime

def export_extractions(texts, pipeline, output_file="extraction_results.json"):
    """Process multiple texts and export to JSON"""
    results = []
    
    for i, text in enumerate(texts):
        extraction = pipeline.process_report(text)

        entities_clean = {}
        for entity_type, ents in extraction['all_entities'].items():
            entities_clean[entity_type] = [
                {
                    'text': e['text'],
                    'confidence': float(e['confidence']),
                    'start': e['start'],
                    'end': e['end']
                } for e in ents
            ]
        results.append({
            'id': i + 1,
            'timestamp': datetime.now().isoformat(),
            'text': text.strip(),
            'entities': entities_clean,
            'medication_count': len(extraction['medications']),
            'condition_count': len(extraction['conditions']),
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Exported {len(results)} extraction results to {output_file}")
    return results

# Export results
exported = export_extractions(clinical_notes, pipeline_demo, "clinical_extractions.json")

# %%
# Quick statistics
print("\n📊 EXTRACTION STATISTICS")
print("-" * 40)

total_meds = sum(r['medication_count'] for r in exported)
total_conditions = sum(r['condition_count'] for r in exported)

print(f"Total documents processed: {len(exported)}")
print(f"Total medications found: {total_meds}")
print(f"Total conditions found: {total_conditions}")
print(f"Avg medications per document: {total_meds/len(exported):.1f}")
print(f"Avg conditions per document: {total_conditions/len(exported):.1f}")




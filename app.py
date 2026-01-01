"""
Clinical NER Web Interface
==========================
A simple Gradio app to extract medical entities from clinical text.

Run with: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from transformers import pipeline
import json

# ============================================================
# Load the fine-tuned model
# ============================================================

print("Loading Clinical NER model...")
ner_pipeline = pipeline(
    "ner",
    model="./clinical_ner_model/final",
    tokenizer="./clinical_ner_model/final",
    aggregation_strategy="simple"
)
print("Model loaded!")


# ============================================================
# Core extraction functions
# ============================================================

def extract_entities(text):
    """Extract entities and organize by type"""
    if not text.strip():
        return {}, "", ""
    
    raw_entities = ner_pipeline(text)
    
    # Organize by entity type
    organized = {"CHEMICAL": [], "DISEASE": []}
    
    for ent in raw_entities:
        entity_type = ent['entity_group']
        if entity_type in organized:
            organized[entity_type].append({
                'text': ent['word'],
                'confidence': float(ent['score']),
                'start': int(ent['start']),
                'end': int(ent['end'])
            })
    
    return organized, raw_entities


def highlight_entities(text, entities):
    """Create highlighted text with entity annotations"""
    if not entities:
        return text
    
    # Sort entities by start position (reverse) to replace from end
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted = text
    for ent in sorted_entities:
        start, end = ent['start'], ent['end']
        entity_type = ent['entity_group']
        
        # Color coding
        if entity_type == "CHEMICAL":
            color = "#90EE90"  # Light green
            label = "💊"
        elif entity_type == "DISEASE":
            color = "#FFB6C1"  # Light pink
            label = "🏥"
        else:
            color = "#D3D3D3"  # Light gray
            label = "📋"
        
        # Wrap entity in highlighted span
        original_text = highlighted[start:end]
        highlighted = (
            highlighted[:start] + 
            f"**{label} {original_text}**" + 
            highlighted[end:]
        )
    
    return highlighted


def format_results_table(organized):
    """Format entities as a markdown table"""
    rows = []
    
    for entity_type, entities in organized.items():
        emoji = "💊" if entity_type == "CHEMICAL" else "🏥"
        for ent in entities:
            rows.append([
                emoji,
                entity_type,
                ent['text'],
                f"{ent['confidence']:.1%}"
            ])
    
    if not rows:
        return "No entities found."
    
    # Create markdown table
    table = "| | Type | Entity | Confidence |\n"
    table += "|---|------|--------|------------|\n"
    for row in rows:
        table += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n"
    
    return table


def process_clinical_text(text, confidence_threshold):
    """Main processing function for Gradio"""
    if not text.strip():
        return "Please enter some clinical text.", "", "{}"
    
    # Extract entities
    organized, raw_entities = extract_entities(text)
    
    # Filter by confidence
    filtered_organized = {}
    filtered_raw = []
    
    for entity_type, entities in organized.items():
        filtered = [e for e in entities if e['confidence'] >= confidence_threshold]
        if filtered:
            filtered_organized[entity_type] = filtered
    
    for ent in raw_entities:
        if float(ent['score']) >= confidence_threshold:
            filtered_raw.append({
                "word": ent['word'],
                "entity_group": ent['entity_group'],
                "score": float(ent['score']),
                "start": int(ent['start']), 
                "end": int(ent['end'])
            })
    
    # Generate outputs
    highlighted = highlight_entities(text, filtered_raw)
    table = format_results_table(filtered_organized)
    
    # JSON output
    json_output = json.dumps(filtered_organized, indent=2, default=float)
    
    # Summary stats
    num_chemicals = len(filtered_organized.get("CHEMICAL", []))
    num_diseases = len(filtered_organized.get("DISEASE", []))
    
    summary = f"**Found:** {num_chemicals} medications/chemicals, {num_diseases} diseases/conditions"
    
    return highlighted, summary + "\n\n" + table, json_output


# ============================================================
# Sample clinical texts for demo
# ============================================================

EXAMPLES = [
    ["""DISCHARGE SUMMARY: 72-year-old male with history of type 2 diabetes mellitus 
and hypertension. Admitted for chest pain, ruled out for myocardial infarction. 
Continue metformin 1000mg BID, lisinopril 20mg daily, and aspirin 81mg daily.
Follow up with cardiology in 2 weeks.""", 0.5],
    
    ["""PROGRESS NOTE: Patient reports improvement in joint pain after starting 
naproxen 500mg twice daily. Rheumatoid arthritis remains stable. 
Continue current regimen of methotrexate 15mg weekly and folic acid 1mg daily.""", 0.5],
    
    ["""ED NOTE: 45-year-old female presenting with acute migraine. Given sumatriptan 
50mg PO with relief. History of anxiety disorder on sertraline 100mg daily.
Discharged home with instructions to follow up with neurology.""", 0.5],
    
    ["""CONSULT NOTE: Patient with chronic kidney disease stage 3 and heart failure 
with reduced ejection fraction. Currently on furosemide 40mg daily, carvedilol 
12.5mg BID, and lisinopril 10mg daily. Recommend adding spironolactone 25mg daily.""", 0.5],
]


# ============================================================
# Build Gradio Interface
# ============================================================

with gr.Blocks(
    title="Clinical NER Extractor",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🏥 Clinical Named Entity Recognition
    
    Extract **medications** (💊 CHEMICAL) and **conditions** (🏥 DISEASE) from clinical text.
    
    *Powered by BioBERT fine-tuned on BC5CDR dataset*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Clinical Text",
                placeholder="Enter clinical notes, discharge summaries, or medical reports...",
                lines=8
            )
            
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
                info="Only show entities above this confidence level"
            )
            
            submit_btn = gr.Button("🔍 Extract Entities", variant="primary")
        
        with gr.Column(scale=2):
            highlighted_output = gr.Markdown(label="Highlighted Text")
    
    with gr.Row():
        with gr.Column():
            results_table = gr.Markdown(label="Extracted Entities")
        
        with gr.Column():
            json_output = gr.Code(label="JSON Output", language="json")
    
    # Examples section
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, confidence_slider],
        outputs=[highlighted_output, results_table, json_output],
        fn=process_clinical_text,
        cache_examples=False
    )
    
    # Connect the button
    submit_btn.click(
        fn=process_clinical_text,
        inputs=[input_text, confidence_slider],
        outputs=[highlighted_output, results_table, json_output]
    )
    
    # Also trigger on Enter key
    input_text.submit(
        fn=process_clinical_text,
        inputs=[input_text, confidence_slider],
        outputs=[highlighted_output, results_table, json_output]
    )
    
    gr.Markdown("""
    ---
    **About this project:**
    - Model: BioBERT fine-tuned on BC5CDR (BioCreative V CDR) dataset
    - Entities: CHEMICAL (medications, drugs) and DISEASE (conditions, symptoms)
    - Author: Joana Owusu-Appiah
    
    *For research and educational purposes only. Not for clinical decision-making.*
    """)


# ============================================================
# Launch the app
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Clinical NER Web Interface...")
    print("="*50 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True for public link
        show_error=True
    )
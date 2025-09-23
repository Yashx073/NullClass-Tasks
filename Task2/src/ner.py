# src/ner.py

import spacy
import os # Added os for checking model path if needed, though not strictly required here

# --- MODEL LOADING ---

# The specialized model 'en_core_sci_lg' is preferred but requires a separate install/download.
try:
    # Attempt to load the medical-specific model
    NER_MODEL = spacy.load("en_core_sci_lg")
    print("NER Model: Loaded specialized 'en_core_sci_lg'.")
except OSError:
    # Fallback to the general small model if the specialized one is missing
    try:
        NER_MODEL = spacy.load("en_core_web_sm")
        print("NER Model: Falling back to general 'en_core_web_sm'.")
    except OSError:
        # Handle case where even the general model isn't installed
        print("FATAL ERROR: Neither 'en_core_sci_lg' nor 'en_core_web_sm' is installed. Run: python -m spacy download en_core_web_sm")
        raise


# --- ENTITY EXTRACTION FUNCTION ---

def get_medical_entities(text):
    """Identifies basic entities in the text."""
    doc = NER_MODEL(text)
    entities = {}
    
    # Simple entity extraction (customization logic)
    for ent in doc.ents:
        # Filter out irrelevant general entities that don't indicate a medical concept
        # NOTE: If using the general 'en_core_web_sm' model, most entities will fall here.
        if ent.label_ in ['GPE', 'ORG', 'NORP', 'PRODUCT', 'DATE', 'CARDINAL']:
             continue 

        # Add entity to its category (label)
        if ent.label_ not in entities:
            entities[ent.label_] = []
        
        if ent.text.strip().lower() not in [e.lower() for e in entities[ent.label_]]:
            entities[ent.label_].append(ent.text.strip())

    # Fallback: If specialized filtering yields nothing (common with general models), 
    # extract all detected named entities as potential keywords/concepts.
    if not entities:
        # Collect all non-empty entities as 'Concepts'
        all_entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
        if all_entities:
             # Using 'Concepts' as a generic label for anything spaCy found
             entities['Concepts'] = list(set(all_entities)) 
             
    return entities


# --- TEST BLOCK ---
if __name__ == '__main__':
    # Added necessary pandas import for testing/standalone use
    import pandas as pd 
    
    test_question = "What are the symptoms and treatments for COVID-19 and the history of the disease?"
    entities = get_medical_entities(test_question)
    
    print("\n" + "="*50)
    print(f"Question: {test_question}")
    print("Extracted Entities:")
    
    # Print in a clean, readable format
    if entities:
        for label, items in entities.items():
            print(f"- {label}: {', '.join(items)}")
    else:
        print("- None detected.")
    print("="*50)
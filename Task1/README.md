# Extractive Summarization


This repository contains an extractive summarization pipeline (TF-IDF + sentence-ranking) with:


- `src/summarizer.py` — summarization code
- `notebook/summarizer_training.ipynb` — training, evaluation (confusion matrix, precision, recall, F1, ROUGE)
- `src/gui.py` — optional Streamlit GUI
- `models/` — saved model files (pickles/joblib or links to Google Drive for large files)
- `data/` — sample texts and dataset


## Setup


1. Create a virtual environment and activate it.


```bash
python -m venv venv
source venv/bin/activate # mac / linux
venv\Scripts\activate # windows
pip install -r requirements.txt
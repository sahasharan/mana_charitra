# Mana Charitra — Telugu History Chatbot

A Streamlit application exploring Telugu history and cultural knowledge through semantic retrieval and Gemini-generated conversational responses.

> **Project status:** Student learning / prototype project. Verify the current deployment and API configuration before sharing a live demo.

## What it does

- Accepts questions about Telugu history and culture in a chat interface.
- Encodes text with the `all-MiniLM-L6-v2` Sentence Transformers model.
- Uses a FAISS inner-product index to retrieve relevant text chunks.
- Passes retrieved context to Google's Gemini API to generate a Telugu response.
- Supports PDF and UTF-8 text uploads for adding material to the knowledge base.
- Stores document records and user-submitted stories in SQLite.

## Architecture

```text
Question / uploaded document
            |
            v
  Text extraction & chunking
            |
            v
 Sentence Transformers embeddings
            |
            v
        FAISS search
            |
            v
   Retrieved context + query
            |
            v
       Gemini response
            |
            v
       Streamlit chat UI
```

## Tech stack

- Python, Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS CPU
- Google Gemini API (`google-generativeai` client)
- SQLite, PyPDF2, pandas, NumPy

## Run locally

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Set your Gemini API key as an environment variable.

   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY = "YOUR_API_KEY"

   # macOS / Linux
   export GEMINI_API_KEY="YOUR_API_KEY"
   ```

4. Start Streamlit.

   ```bash
   streamlit run app.py
   ```

## Data and configuration

The application creates local database and vector-index files under `data/`. These runtime artifacts are ignored by Git; they should be regenerated locally or through a documented data-ingestion process rather than committed.

Never commit API keys, `.env` files, or Streamlit secrets. For hosted deployment, add `GEMINI_API_KEY` through the host's secret-management settings.

## Current limitations

- Answer quality depends on the documents in the knowledge base and the retrieval threshold.
- The application should be evaluated for factual accuracy before use as a historical reference.
- Uploaded content is not automatically verified for accuracy or provenance.
- This repository is a learning project, not a production historical authority.

## Future improvements

- Add automated tests for chunking, indexing, and retrieval.
- Attach source references to generated answers.
- Add retrieval-quality evaluation and configurable thresholds.
- Improve persistence and re-indexing for large knowledge bases.
- Add a reproducible deployment guide and screenshots.

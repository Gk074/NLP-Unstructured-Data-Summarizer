# Faithful Meeting Minutes Generator (MoM)

Research-driven NLP pipeline for structured, evidence-grounded meeting minutes generation from long multi-speaker transcripts.

## Key Features
- Embedding-based topic segmentation
- Decision and action item extraction
- Evidence span attribution
- Structured JSON-constrained output (Pydantic)
- Markdown MoM rendering
- Designed to mitigate hallucination and long-context bias

## Architecture
Transcript → Parsing → Topic Segmentation → Salience Scoring → 
Decision/Action Extraction → Structured MoM JSON → Markdown Output

## Setup

### 1. Create Python 3.11 environment
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
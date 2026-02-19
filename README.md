# G44-RAG — University Regulations RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about Sharif University of Technology's academic regulations, using **only** the official bylaws and guideline documents.

## Architecture

```
User Question
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  Embedding  │───▶│  FAISS Index │───▶│  Top-k Chunks       │
│  (E5-base)  │    │  (cosine)    │    │  (with metadata)    │
└─────────────┘    └──────────────┘    └────────┬────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │  Qwen2.5-7B-Instruct   │
                                   │  (4-bit quantized)      │
                                   │  + Persian system prompt │
                                   └────────────┬───────────┘
                                                │
                                                ▼
                                        Grounded Answer
                                        with Citations
```

## Project Structure

```
G44-RAG/
├── data/
│   ├── sharif_rules_chunks.json   # 252 chunks from 26 regulations
│   ├── faiss_index.bin            # FAISS vector index (generated)
│   └── chunk_mapping.json         # Chunk metadata mapping (generated)
├── src/
│   ├── build_kb.py                # Scrape & chunk regulations
│   ├── embed_chunks.py            # Compute embeddings + build FAISS index
│   ├── retriever.py               # Query embedding + vector search
│   ├── generator.py               # LLM wrapper (Qwen2.5-7B)
│   ├── rag_pipeline.py            # Full RAG pipeline
│   ├── app.py                     # Gradio web UI
│   └── evaluate.py                # Evaluation on question set
├── notebooks/
│   └── rag_pipeline.ipynb         # Complete Colab notebook (recommended)
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Google Colab (Recommended — needs GPU)

1. Upload `notebooks/rag_pipeline.ipynb` to Google Colab
2. Set runtime to **GPU** (T4 or better)
3. Upload `data/sharif_rules_chunks.json` when prompted
4. Run all cells sequentially

### Option 2: Local (embedding + retrieval only, LLM needs GPU)

```bash
pip install -r requirements.txt

# Step 1: Build FAISS index
python -m src.embed_chunks

# Step 2: Test retrieval
python -m src.retriever

# Step 3: Run full pipeline (requires GPU for LLM)
python -m src.rag_pipeline

# Step 4: Launch Gradio UI
python -m src.app
```

## Components

### Knowledge Base (20 pts)
- **26 regulations** scraped from ac.sharif.edu/rules/
- **252 chunks** with metadata: rule_title, section_title, parent_section, rule_date, rule_url
- Chunking based on article/clause structure (headers, ماده, بند)

### Embedding & Retrieval
- **Model:** `intfloat/multilingual-e5-base` (768-dim, Persian-compatible)
- **Index:** FAISS IndexFlatIP with L2-normalized vectors (cosine similarity)
- **Retrieval:** Top-k chunks (default k=5) with E5 query/passage prefixes

### Language Model
- **Model:** Qwen2.5-7B-Instruct
- **Quantization:** 4-bit NF4 via bitsandbytes
- **Prompt:** Persian system prompt enforcing source-grounded answers with citations

### User Interface
- Gradio ChatInterface with Persian UI
- Shows answer + source citations (regulation name, section, relevance score)

## Evaluation

The pipeline can be evaluated on a set of questions. Results are saved as both JSON and CSV with: question, answer, retrieved sources, and scores.

## Team
Group 44

# Legal Document Q&A System (RAG-based)

A Retrieval-Augmented Generation (RAG) system that answers legal questions
using Indian legal judgments.

---

## 🚀 Features
- Legal document loading & cleaning
- Text chunking
- Semantic embeddings (Sentence Transformers)
- Vector search using FAISS
- Answer generation using **Gemini API**

---

## 🧠 Architecture
1. Documents are split into chunks  
2. Chunks are embedded into vectors  
3. FAISS index stores embeddings  
4. User query is embedded  
5. Top-k relevant chunks are retrieved  
6. Gemini generates a grounded answer  

---

## 📂 Project Structure

```text
legal-doc-ai/
│
├── app/
│   ├── data_loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── index.py
│   └── llm.py
│
├── scripts/
│   ├── build_corpus.py
│   └── query.py
│
├── data/               # generated locally (ignored in git)
├── venv/               # virtual environment (ignored)
├── .gitignore
├── README.md
└── requirements.txt
```
---

## SETUP
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
---

## Set Gemini API key (Windows):
```bash
setx GEMINI_API_KEY "your_api_key_here"
```
---
## ▶️ Run
Build the vector index:
```bash
python -m scripts.build_corpus
```
Start the Q&A system:
```bash
python -m scripts.query
```
---
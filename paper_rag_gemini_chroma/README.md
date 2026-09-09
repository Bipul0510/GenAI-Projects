# Simple Gemini + ChromaDB RAG

## What is this project?

A beginner-friendly RAG application for the three supplied AI research papers.

The whole project in one sentence:

**Find relevant text from the papers, then give that text to Gemini so Gemini can answer the question.**

## Simple architecture

PDF papers
-> split into small pieces
-> Gemini embeddings
-> ChromaDB
-> user question
-> find 3 relevant pieces
-> Gemini
-> answer + sources

## Files

- `ingest.py` = reads papers and creates the ChromaDB database
- `app.py` = asks questions and displays answers
- `requirements.txt` = libraries
- `.env.example` = API-key template
- `papers/` = the three supplied papers
- `chroma_db/` = local vector database

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add:

```text
GEMINI_API_KEY=your_key_here
```

Do not upload `.env` to GitHub.

## Build the database

```powershell
python ingest.py
```

## Run the application

```powershell
streamlit run app.py
```

## Test questions

- What are the two sub-layers in a Transformer encoder?
- Why is positional encoding necessary?
- What is multi-head attention?
- What is RAG?
- What is few-shot learning in GPT-3?

## How to explain the project in an interview

> "I built a simple RAG application for three AI research papers. First I read the PDFs and split them into small pieces. I converted each piece into an embedding and stored the embeddings in ChromaDB. When a user asks a question, I convert the question into an embedding and use ChromaDB to find the three most relevant pieces. I send those pieces to Gemini as context, and Gemini generates the answer."

## Three words to remember

**Embedding:** turns text into numbers that represent meaning.

**ChromaDB:** stores those numbers and finds similar text.

**RAG:** Retrieve useful information, add it to the prompt, then Generate an answer.

## Why not simply ask Gemini?

Without RAG, Gemini answers from its model knowledge.

With RAG, the application first finds information from our supplied papers and gives that information to Gemini.

## Important

This is intentionally a learning project. It uses the core RAG idea from the supplied RAG paper, but it is not a reproduction of the original research implementation.

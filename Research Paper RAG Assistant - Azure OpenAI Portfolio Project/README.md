# Research Paper RAG Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application that answers questions about three foundational AI papers:

- **Attention Is All You Need** - the Transformer architecture.
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** - the RAG pattern.
- **Language Models are Few-Shot Learners** - GPT-3 and in-context learning.

The app uses **Azure OpenAI** for embeddings and answer generation, **FAISS** for local semantic search, and **Streamlit** for the web interface. Every answer is grounded in retrieved excerpts and shows the original paper and page number.

## Why this is a strong portfolio project

It demonstrates an end-to-end GenAI workflow, not just an API call:

```text
PDF research papers
       ↓
Page-aware text extraction and overlapping chunks
       ↓
Azure OpenAI embeddings
       ↓
FAISS vector index (persisted locally)
       ↓
Top-3 relevant passages
       ↓
Azure OpenAI grounded answer with citations
```

## Features

- Answers questions over the supplied papers instead of the public web.
- Stores `paper_name`, filename, page number, and chunk ID alongside every chunk.
- Builds the embedding index once and reuses it on later app launches.
- Offers a **Refresh index** button when PDFs change.
- Displays retrieved excerpts so users can inspect the evidence.
- Uses a constrained prompt to reduce hallucinations and state when the papers do not contain an answer.
- Keeps all application logic in one beginner-friendly file: `app.py`.

## Project structure

```text
research-paper-rag/
├── app.py                 # Entire application
├── requirements.txt       # Python packages
├── .env.example           # Safe configuration template
├── data/                  # Source research papers
└── storage/               # Created at runtime; FAISS index and metadata
```

`storage/` is generated locally after the first successful indexing run. Do not commit your `.env` file or Azure key.

## Prerequisites

- Python 3.10 or later
- An Azure OpenAI resource
- One deployed chat model and one deployed text-embedding model in that resource

Use the **deployment names** you created in Azure, not necessarily the base model names. Confirm that both deployments are available in your Azure region before running the app.

## Azure OpenAI setup

1. In your Azure OpenAI / Azure AI Foundry resource, deploy a chat model and a text embedding model.
2. Copy the resource endpoint and an API key from the Azure portal.
3. Copy `.env.example` to `.env`.
4. Replace every placeholder in `.env` with your own values:

```env
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=your-secret-key
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name
```

If your Azure resource supports a different API version, use that supported version instead. The application intentionally reads deployment names from environment variables, so it is not tied to one model or region.

## Run locally on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env in a text editor and add your Azure values.
streamlit run app.py
```

The browser should open at `http://localhost:8501`. On the first question, the app reads the PDFs and creates embeddings. Later runs reuse `storage/papers.faiss`, which is faster and avoids unnecessary embedding calls.

## Suggested demo questions

| Question | Expected primary paper |
| --- | --- |
| What are the main components of a RAG model, and how do they interact? | Retrieval-Augmented Generation |
| What are the two sub-layers in each encoder layer of the Transformer model? | Attention Is All You Need |
| Explain positional encoding in Transformers and why it is necessary. | Attention Is All You Need |
| What is multi-head attention and why is it beneficial? | Attention Is All You Need |
| What is few-shot learning, and how does GPT-3 implement it during inference? | Language Models are Few-Shot Learners |

Also test an unrelated question such as **"What is the capital of France?"**. A well-grounded answer should say that it could not find this in the supplied papers rather than guessing.

## Manual test checklist

- [ ] Each suggested question returns a relevant answer.
- [ ] Each answer shows at least one matching paper title and page number.
- [ ] The expandable context contains excerpts supporting the response.
- [ ] An unrelated question receives the grounded fallback message.
- [ ] Restarting the app reuses the existing index.
- [ ] Clicking **Refresh index** successfully rebuilds it.
- [ ] Missing Azure settings show an instructional message and never reveal a secret.

## Screenshots for your portfolio

After configuring Azure, add these screenshots to this README or your GitHub repository:

1. The home screen with a sample question selected.
2. A complete answer with its page-level citations.
3. The expanded retrieved-excerpts section.
4. The sidebar showing the ready index status.

## Limitations and honest next steps

This intentionally small project searches three static PDFs. Retrieval quality depends on PDF text extraction and chunk size, and answers may be limited by the retrieved top-3 excerpts. It is a learning project, not a production knowledge system.

Useful next steps:

1. Add hybrid retrieval (keyword + vector search).
2. Add an automated evaluation set and retrieval Recall@k.
3. Add chat history and answer feedback.
4. Replace local FAISS with Azure AI Search for a scalable cloud deployment.
5. Add document upload, authentication, and observability before production use.

## Resume and interview talking points

- Built a grounded RAG QA application over academic PDFs using Azure OpenAI, Streamlit, and FAISS.
- Implemented page-aware PDF chunking, embedding-based semantic retrieval, and persisted vector indexing.
- Passed retrieved evidence and page metadata to the LLM to produce citation-backed answers.
- Designed the prompt and UI to make evidence visible and reduce unsupported answers.
- Balanced a portfolio-quality UX with a deliberately small, explainable architecture suitable for a first GenAI project.

## Cost and security notes

The initial indexing run sends document chunks to your Azure embedding deployment, and every question uses one embedding request plus one chat request. Avoid unnecessary refreshes while learning. Keep `.env` private, rotate exposed keys immediately, and use Azure access controls appropriate for your account.

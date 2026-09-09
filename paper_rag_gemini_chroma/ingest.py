from pathlib import Path
import os
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Add GEMINI_API_KEY to your .env file.")

client = genai.Client(api_key=API_KEY)
db = chromadb.PersistentClient(path="chroma_db")
EMBEDDING_MODEL = "gemini-embedding-001"

def embed(text):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def make_chunks(text, size=1200):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size)]

try:
    db.delete_collection("papers")
except Exception:
    pass

collection = db.create_collection("papers")
total = 0

for pdf in Path("papers").glob("*.pdf"):
    print("Reading:", pdf.name)
    reader = PdfReader(str(pdf))

    for page, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""

        for number, chunk in enumerate(make_chunks(text)):
            if not chunk.strip():
                continue

            collection.add(
                ids=[f"{pdf.stem}-{page}-{number}"],
                documents=[chunk],
                embeddings=[embed(chunk)],
                metadatas=[{"paper": pdf.name, "page": page}],
            )
            total += 1

print(f"Done! {total} chunks stored in ChromaDB.")

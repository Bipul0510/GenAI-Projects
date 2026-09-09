import os
import chromadb
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Create a .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("papers")

try:
    collection = db.get_collection("papers")
except Exception:
    st.error(
        "ChromaDB is not ready yet. "
        "Please run 'python ingest.py' first."
    )
    st.stop()

def embed_question(question):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def answer_question(question):
    results = collection.query(
        query_embeddings=[embed_question(question)],
        n_results=3,
    )

    context = ""
    sources = []

    for i, text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        context += f"\nSOURCE {i + 1}:\n{text}\n"
        sources.append(f"{i + 1}. {meta['paper']} — page {meta['page']}")

    prompt = f"""
Answer the question using ONLY the CONTEXT below.
If the answer is not in the context, say:
"I could not find the answer in the papers."
Keep the answer simple and clear.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text, sources

st.title("📚 AI Research Paper Assistant")
st.write("Ask questions about the three supplied AI research papers.")

question = st.text_input(
    "Your question:",
    placeholder="Example: What is multi-head attention?"
)

if st.button("Ask") and question.strip():
    with st.spinner("Finding information..."):
        try:
            answer, sources = answer_question(question)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for source in sources:
                st.write(source)
        except Exception as e:
            st.error(f"Error: {e}")

# Beginner Interview Notes

### What is RAG?
Retrieval-Augmented Generation. We retrieve useful information before asking the LLM to generate an answer.

### Why embeddings?
They turn text into numerical vectors so similar meanings can be compared.

### Why ChromaDB?
It is a simple local vector database for storing embeddings and finding similar text.

### What does Gemini do?
Gemini creates embeddings and generates the final answer.

### What happens when I ask a question?
Question -> embedding -> ChromaDB -> top 3 chunks -> Gemini -> answer.

### What is hallucination?
When an LLM gives information that is not supported by the available evidence.

### How did I reduce hallucination?
The prompt tells Gemini to use only the retrieved context and say when the answer is not found.

### What would I improve later?
Better chunking, reranking, evaluation, and better citations.

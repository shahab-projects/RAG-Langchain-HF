from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ingestion, retrieval, evaluation, embedding, generator

app = FastAPI()

# Define request schema
class Query(BaseModel):
    question: str

# Preload data + FAISS index (so it's ready at startup)
doc_path = "data/sample_docs"
chunks = ingestion.load_and_chunk(doc_path)
emb = embedding.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Converting chunks of documents to embeddings
vectorstore = retrieval.build_faiss_index(chunks, emb)                                      # Building FAISS vectorstore index
llm = generator.load_generator()

@app.post("/rag")
async def rag_answer(query: Query):
    try:
        print('query.question', query.question)
        docs = retrieval.retrieve_similar_docs(vectorstore, query.question) # Retrieve documents
        doc_texts = [d.page_content for d in docs] # Extract the text from each document
        enriched_q = query.question + " " + " ".join(doc_texts) # Build enriched query
        answer = generator.generate_answer(llm, enriched_q) # Generate answer
        return {"answer": answer}

    except Exception as e:
        # For debugging, return error as JSON
        return {"error": str(e)}
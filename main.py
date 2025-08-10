from rag_pipeline import ingestion, embedding, retrieval, generator

def run_rag(query, doc_path):
    chunks = ingestion.load_and_chunk(doc_path)             # 1. Load and chunk
    embedder = embedding.HuggingFaceEmbeddings(chunks)      # 2. Embeddings + FAISS index
    index = retrieval.build_faiss_index(chunks, embedder)   # 3. Retrieve context
    context = retrieval.retrieve_similar_docs(index, query)
    llm = generator.load_generator()
    return generator.generate_answer(llm, query, context)
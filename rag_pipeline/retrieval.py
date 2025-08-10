from langchain.vectorstores import FAISS

def build_faiss_index(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def retrieve_similar_docs(index, query, k=3):
    return index.similarity_search(query, k=k)

from langchain_huggingface import HuggingFaceEmbeddings

def embed_documents(doc_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model.embed_documents([doc.page_content for doc in doc_chunks])

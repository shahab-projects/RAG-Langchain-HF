import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(data_path, chunk_size=1000, chunk_overlap=200):
    """
    Loads documents from a directory and chunks them into smaller pieces.
    
    Args:
        data_path (str): Path to the directory containing documents.
        chunk_size (int): Number of characters per chunk.
        chunk_overlap (int): Number of characters overlapping between chunks.
    
    Returns:
        list: List of Document objects
    """
    # Load all files in the directory
    loaders = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(data_path, filename)))
        elif filename.endswith(".txt"):
            loaders.append(TextLoader(os.path.join(data_path, filename)))

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs

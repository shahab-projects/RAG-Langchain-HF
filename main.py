from rag_pipeline import ingestion, embedding, retrieval, generator, evaluation

def main():
    # 1. Load QA dataset
    questions, answers = evaluation.load_qa_dataset("data/qa_dataset.csv") # We need it to evaluate questions and their ground-truth answers

    # 2. Load sample docs & build vectorstore
    doc_path = 'data/sample_docs'
    chunks = ingestion.load_and_chunk(doc_path)             

    print(f"Number of chunks: {len(chunks)}") 
    if len(chunks) == 0:                                                                       # Check that the document loader actually returned content
        raise ValueError("No document chunks found! Check your document loading/splitting.")

    emb = embedding.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Converting chunks of documents to embeddings
    vectorstore = retrieval.build_faiss_index(chunks, emb)                                     # Building FAISS vectorstore index

    # 3. Load LLM
    llm = generator.load_generator()
    
    # # Test single query generation to see output
    # sample_query = "What is 5G?"
    # sample_context = vectorstore.similarity_search(sample_query, k=3)
    # answer = generator.generate_answer(llm, sample_query, sample_context)
    # print("\nSample Query:", sample_query)
    # print("Sample generated answer:", answer)

    # 4. Evaluate baseline (LLM only)
    baseline_scores = evaluation.evaluate_model(questions, answers, llm)

    # 5. Evaluate RAG
    rag_scores = evaluation.evaluate_model(questions, answers, llm, vectorstore)

    # 6. Print results
    print("\n--- Evaluation Results ---")
    print("Baseline Example Score:", baseline_scores[0])
    print("RAG Example Score:", rag_scores[0])
    
    # 7. Showing sample answers from both models

    # Pick one question from your dataset
    sample_question = questions[0]
    # print('sample_question', sample_question)
    reference_answer = answers[0]

    # Baseline answer
    baseline_answer = generator.generate_answer(llm, sample_question)

    # RAG answer
    retrieved_docs = retrieval.retrieve_similar_docs(vectorstore, sample_question)
    rag_answer = generator.generate_answer(llm, sample_question, retrieved_docs)

    print("\n=== Example QA ===")
    print(f"Question: {sample_question}")
    print(f"Reference Answer: {reference_answer}")
    print(f"Baseline Answer: {baseline_answer[:200]} ...")
    print(f"RAG Answer: {rag_answer[:200]} ...")
    print("=================\n")    

if __name__ == "__main__":
    main()
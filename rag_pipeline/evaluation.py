import evaluate
import pandas as pd

def evaluate_model(questions, answers, llm, vectorstore=None, k=3):
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    results = []
    for q, gt in zip(questions, answers):
        if vectorstore:  # RAG mode
            context_docs = vectorstore.similarity_search(q, k=k)
            context_str = "\n".join([d.page_content for d in context_docs])
            prompt = f"Answer based on context:\n{context_str}\n\nQuestion: {q}"
            # print('enriched question', prompt)
        else:  # No-RAG mode
            prompt = q
            # print('original question', prompt)
        pred = llm.invoke(prompt)
        # print('answer', pred)
        rouge_score = rouge.compute(predictions=[pred], references=[gt])
        bert_score = bertscore.compute(predictions=[pred], references=[gt], lang="en")
        results.append({"rouge": rouge_score, "bertscore": bert_score})
    return results

def load_qa_dataset(path):
    df = pd.read_csv(path)
    return df['question'].tolist(), df['answer'].tolist()

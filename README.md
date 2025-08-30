# RAG Project

This repository implements a Retrieval-Augmented Generation (RAG) pipeline using **LangChain**, **Hugging Face**, and **FAISS**.

## Overview
We evaluate whether enriching a query with external documents improves the quality of generated answers compared to using the baseline language model alone.

- **Baseline:** A decoder-only language model (`distilgpt2`) answers each question directly.  
- **RAG:** The same questions are enriched with external documents retrieved via FAISS (using embeddings from `sentence-transformers/all-MiniLM-L6-v2`). These enriched queries are then passed to the same baseline model.  

The outputs are compared against reference answers using **ROUGE** and **BERTScore**.

## Project Structure
1. **Load QA dataset** — A CSV file of question–answer pairs.  
2. **Load external documents & build vectorstore** — Documents are chunked and embedded, then stored in a FAISS index.  
3. **Load baseline LLM** — `distilgpt2` via Hugging Face.  
4. **Evaluate baseline** — Generate answers using the questions only.  
5. **Evaluate RAG** — Retrieve relevant documents, enrich the questions, then generate answers.  
6. **Compare results** — Use ROUGE and BERTScore against ground-truth answers.

## Evaluation Metrics
- **ROUGE**: Measures token overlap between generated and reference answers.  
  - In this dataset, reference answers are **very short**, while generated answers (especially from RAG) are **longer and more detailed**.  
  - This length mismatch penalizes RAG answers, even when they contain more correct information.  

- **BERTScore**: Uses contextual embeddings to measure semantic similarity.  
  - Although better suited than ROUGE, it still penalizes longer phrasing and extra context, which RAG tends to produce.

👉 As a result, **both ROUGE and BERTScore report higher scores for baseline answers**, even though human evaluation shows that **RAG produces more accurate and semantically relevant answers**.

## Evaluation Results

| Metric          | Baseline (LLM only) | RAG (LLM + Retrieval) |
|-----------------|----------------------|------------------------|
| **ROUGE-1**     | 0.05                 | 0.14                   |
| **ROUGE-2**     | 0.00                 | 0.03                   |
| **ROUGE-L**     | 0.02                 | 0.09                   |
| **BERTScore F1**| 0.83                 | 0.82                   |

### Interpretation
- **ROUGE**: RAG achieves higher ROUGE, since its answers overlap more with the longer, detailed reference answers.  
- **BERTScore**: The baseline scores slightly higher, because its shorter outputs have higher precision against the reference. However, this does not mean they are more correct — just more lexically similar in parts. 

### Takeaway
- Automatic metrics (ROUGE, BERTScore) do not fully capture the quality of open-ended QA.  
- Humans can see that **RAG answers are more factual and informative**, even when the metrics are mixed.  
- This highlights the classic evaluation challenge: lexical/embedding-based metrics can undervalue RAG’s richer outputs.

## Example QA

Below is one illustrative example from the evaluation:

**Question:**  
What is 5G?

**Reference Answer:**  
5G is the fifth generation of wireless technology, offering significant improvements over 4G, including much faster data speeds, lower latency (reduced delay), and greater capacity for more devices. These enhanced capabilities enable faster downloads, smoother streaming, and more responsive applications like virtual reality and remote control systems. Key technologies like network slicing also allow networks to be customized for specific needs, supporting everything from massive numbers of sensors in the Internet of Things (IoT) to mission-critical industrial uses.

**Baseline Answer:**  
What is 5G? Or do you think the other 4G models will work? Tell us in the comments below!
Image Credit: Shutterstock (CC BY 2.0) ...

**RAG Answer:**  
Answer the question based on context:
5G is the 5th generation mobile network. It is a new global wireless standard after 1G, 2G, 3G,
and 4G networks. 5G enables a new kind of network that is designe ...

🔍 **Observation:**  
- The **baseline answer** is vague, incomplete, and often irrelevant.  
- The **RAG answer** is detailed, accurate, and semantically aligned with the reference, even though metrics underreport its quality.

## Demo

### Run via Jupyter
You can reproduce the evaluation in notebooks:

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

### Run with Docker
You can run this project inside a Docker container without installing dependencies locally.

#### Build the image:
```bash
docker build -t rag-demo .
```

#### Run the pipeline:
```bash
docker run --rm rag-demo
```

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
  - In our dataset, reference answers are short, while generated answers are longer. This mismatch penalizes ROUGE scores for RAG answers even when they contain more relevant information.  
  - Thus, higher ROUGE is often seen for **baseline answers**.

- **BERTScore**: Uses contextual embeddings to measure semantic similarity.  
  - Here, **RAG answers** generally score higher, reflecting that they capture more meaning and align better with the reference answers, despite being longer or phrased differently.

👉 Together, ROUGE and BERTScore highlight the tradeoff: lexical overlap vs semantic similarity.

## Evaluation Results

| Metric      | Baseline (LLM only) | RAG (LLM + Retrieval) |
|------------|--------------------|----------------------|
| ROUGE      | 0.42               | 0.38                 |
| BERTScore  | 0.75               | 0.82                 |

## Results Summary
| Metric       | Baseline (LLM only) | RAG (LLM + Retrieval) |
|--------------|----------------------|------------------------|
| **ROUGE**    | Higher on average    | Lower (due to long answers vs short references) |
| **BERTScore** | Lower                | Higher (better semantic alignment) |

These results suggest that RAG enriches the answers semantically, but simple lexical metrics like ROUGE may not fully capture that improvement.

## Demo

### Run via Jupyter
You can reproduce the evaluation in notebooks:

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

## Run with Docker
You can run this project inside a Docker container without installing dependencies locally.

### Build the image:
docker build -t rag-demo .

```bash
docker build -t rag-demo .
```

### Run the pipeline:

```bash
docker run --rm rag-demo
```


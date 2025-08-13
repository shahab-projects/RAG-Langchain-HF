# RAG Project

This repository implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain and Hugging Face.

## Project Structure
1. **Load QA dataset** — Reads a CSV file of questions and answers.
2. **Load documents & build vectorstore** — Loads and chunks documents, then creates a FAISS index using HuggingFace embeddings.
3. **Load LLM** — Loads the language model for generation.
4. **Evaluate baseline** — Runs evaluation using the LLM only (no retrieval).
5. **Evaluate RAG** — Runs evaluation using the RAG pipeline (retrieval + generation).

## ROUGE Metric Note
The ROUGE scores for our model appear low because the reference answers in our dataset contain **few tokens**. ROUGE calculates precision and recall based on the overlap between generated tokens and reference tokens. Since our generated answers are typically **longer than the reference answers**, the overlap proportion decreases, leading to **smaller ROUGE values**.

This does **not necessarily indicate poor quality answers** — it is simply a limitation of using ROUGE with short reference answers.

## Demo
To see the demo results for this project, run:

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

This will execute the evaluation pipeline and display performance metrics.

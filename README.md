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

## API / FastAPI

### How to test via:

- Browser: ```bash http://127.0.0.1:8000/docs ```
- Python script (```bash test.py ```)
- curl

### Example response
```bash
{
  'answer': 'What is 5G? 5G is the 5th generation mobile network. It is a new global wireless standard after 1G, 2G, 3G, \nand 4G networks. 5G enables a new kind of network that is designed to connect virtually \neveryone and everything together including machines, objects, and devices. \n5G wireless technology is meant to deliver higher multi-Gbps peak data speeds, ultra low \nlatency, more reliability, massive network capacity, increased availability, and a more uniform \nuser experience to more users. Higher performance and improved efficiency empower new user \nexperiences and connects new industries. \nThe previous generations of mobile networks are 1G, 2G, 3G, and 4G. \n1G, 2G, 3G, and 4G all led to 5G, which is designed to provide more connectivity than was ever \navailable before. \n5G is a uniﬁed, more capable air interface. It has been designed with an extended capacity to \nenable next-generation user experiences, empower new deployment models and deliver new \nservices. older, hardware-centric networks.  \n5G is the natural successor to 4G LTE, representing a major leap forward in wireless network \nperformance and capabilities. Early deployments are often 5G Non-Standalone (NSA), which \nuses 5G radios built on existing 4G infrastructure. As networks evolve, 5G Standalone (SA) mode \nis becoming more common, fully utilizing the new 5G core network and delivering the full \npotential of 5G. In telecommunications, 5G is the "ﬁfth generation" of cellular network technology, as the \nsuccessor to the fourth generation (4G), and has been deployed by mobile operators worldwide \nsince 2019. \nCompared to 4G, 5G networks offer not only higher download speeds, with a peak speed of \n10 gigabits per second (Gbit/s), but also substantially lower latency, enabling near-instantaneous \ncommunication through cellular base stations and antennae. There is one global uniﬁed 5G \nstandard: 5G New Radio (5G NR), which has been developed by the 3rd Generation Partnership \nProject (3GPP) based on speciﬁcations deﬁned by the International Telecommunication \nUnion (ITU) under the IMT-2020 requirements.  \nThe increased bandwidth of 5G over 4G allows them to connect more devices simultaneously \nand improving the quality of cellular data services in crowded areas. These features make 5G \nparticularly suited for applications requiring real-time data exchange, such as extended data\nover 3G. In other words, the system is designed to\nencrypt the data from the network and provide better data transfer speeds. \n5G is a new 4G wireless standard that is designed to be used as a solution for customers \nto all networks, while the traditional 5G network is designed to provide very strong reliability and\nenhance the Internet of Things (IoT) capabilities of the future.  \n5G is a new 3G wireless standard that is designed to be used as a solution for customers \nto all networks, while the traditional 5G network is designed to provide'
}
```

#### Run with FastAPI
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker
For **running the FastAPI Server automatically** on docker run, replace the command line in the Docker File with:
```bash
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- This way, you can run:
```bash
docker build -t rag-demo .
docker run --rm -p 8000:8000 rag-demo
```

## Future Work & Deployment

This project is containerized with Docker, making it portable and ready for deployment in production environments.  
While the current version runs locally, the following deployment paths can be taken:

### 1. Deploying on AWS SageMaker (Production-Ready)
- **Push image to Amazon ECR** (Elastic Container Registry).
- **Create a SageMaker model** that points to the ECR image.
- **Deploy as a SageMaker Endpoint** to serve predictions via a REST API.
- This setup enables scalable inference, monitoring, and integration with other AWS services.

> Note: An AWS account is required to perform this step. For this project, we only demonstrate containerization and outline the deployment path.

### 2. Sharing via DockerHub
- The image can be pushed to DockerHub for easier sharing:
```bash
docker tag rag-demo <your-dockerhub-username>/rag-demo
docker push <your-dockerhub-username>/rag-demo
```

Anyone can then run the project with:
```bash
docker pull <your-dockerhub-username>/rag-demo
docker run -p 8080:8080 <your-dockerhub-username>/rag-demo
```

### 3. Other Deployment Options
- Kubernetes / EKS – Run the container as part of a larger microservice system.


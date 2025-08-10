from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def load_generator():
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",  
        device=-1,  # CPU
        max_new_tokens=128,
        temperature = 0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

def generate_answer(llm, query, context):
    context_str = "\n".join([doc.page_content for doc in context])
    prompt = f"Answer the question based on context:\n{context_str}\n\nQuestion: {query}"
    return llm.invoke(prompt)

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

KNOWLEDGE_BASE = "./knowledge_base.json"
chunks = []

def load_knowledge_base():
    global chunks
    if os.path.exists(KNOWLEDGE_BASE):
        with open(KNOWLEDGE_BASE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from knowledge base.")

def search_chunks(query: str, k: int = 6):
    query_lower = query.lower()
    keywords = query_lower.split()
    
    scored = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scored.append((score, chunk["content"]))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for _, c in scored[:k]])

def build_rag_chain():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are FUDA, a knowledgeable virtual assistant for Federal University Dutse (FUD), Nigeria.
Use ONLY the context below to answer. The context comes directly from the FUD Student Handbook.

RULES:
- Answer directly and confidently from the context
- Use numbered lists for multiple items
- Keep answers concise and clear
- Only mention https://fud.edu.ng if the context doesn't have enough detail
- Never say "the context doesn't mention" — just answer from what you know

Context:
{context}

Question:
{question}

Answer:
""")

    def rag_invoke(question: str) -> str:
        context = search_chunks(question)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    return rag_invoke

def process_pdf_and_build_chain(pdf_path: str):
    load_knowledge_base()
    return build_rag_chain()

def load_chain():
    load_knowledge_base()
    return build_rag_chain()
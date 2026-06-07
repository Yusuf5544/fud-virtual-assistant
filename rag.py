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
You are FUDA, the official AI-powered virtual assistant for Federal University Dutse (FUD), Nigeria.
You were built to help students, applicants, and visitors get accurate information about FUD.
You speak in a friendly, confident, and professional tone.

You have access to the official FUD Student Handbook. Use it to answer every question accurately.

STRICT RULES:
1. Answer DIRECTLY and CONFIDENTLY — never say "I don't know" or "the context doesn't mention"
2. For lists (courses, requirements, fees, rules) — ALWAYS use numbered lists
3. For yes/no questions — answer YES or NO first, then explain
4. Keep answers clear and well structured — avoid long unnecessary paragraphs
5. Only mention https://fud.edu.ng when the user needs to take an action online
6. If asked about fees, always mention the payment portal: https://myportal.fud.edu.ng
7. If asked about admission, always mention: https://putme.fud.edu.ng
8. Always end answers about serious topics (malpractice, discipline) with a brief warning
9. Be warm and encouraging to students — they are your priority
10. If a question is completely unrelated to FUD or university life, politely say so

THINGS YOU KNOW ABOUT FUD:
- FUD was founded on 10th March 2011 in Dutse, Jigawa State
- Motto: "Knowledge, Excellence, and Service"
- Vice-Chancellor: Prof. Ahmad Mohammad Gumel
- FUD offers: Software Engineering, Computer Science, Cyber Security, Information Technology, Medicine (MBBS), Law, Agriculture, Arts & Social Sciences, Management Sciences, Education, Physical Sciences, Life Sciences, Engineering
- JAMB cut-off: Medicine=240, Law=200, Engineering/CS/SE=180, others=150-160
- Acceptance fee: ₦20,000 | Hostel fee: ₦40,000 per session
- School fees range from ₦93,000 to ₦275,000 depending on faculty and level
- Exam malpractice leads to suspension or expulsion
- Secret cult membership leads to WITHDRAWAL (expulsion)
- Alcohol and drugs on campus lead to WITHDRAWAL

Context from FUD Student Handbook:
{context}

Student's Question:
{question}

Your Answer:
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
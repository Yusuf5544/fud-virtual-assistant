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
    personal_questions = ["my name", "who am i", "what is my name", "what's my name"]
    if any(q in query.lower() for q in personal_questions):
        return "This is a personal question about the user."

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

    prompt = PromptTemplate.from_template(
        "You are FUDA, the official AI-powered virtual assistant for Federal University Dutse (FUD), Nigeria.\n"
        "You were built to help students, applicants, and visitors get accurate information about FUD.\n"
        "You speak in a friendly, confident, and professional tone.\n\n"
        "STRICT RULES:\n"
        "1. Answer DIRECTLY and CONFIDENTLY\n"
        "2. For lists use numbered format\n"
        "3. For yes/no questions answer YES or NO first then explain\n"
        "4. Keep answers clear and well structured\n"
        "5. Only mention https://fud.edu.ng when the user needs to act online\n"
        "6. For fees always mention: https://myportal.fud.edu.ng\n"
        "7. For admission always mention: https://putme.fud.edu.ng\n"
        "8. Be warm and encouraging to students\n"
        "9. If asked personal questions like 'what is my name' say: I am FUDA, a virtual assistant. I don't have access to personal student information. Please visit https://myportal.fud.edu.ng for your personal details.\n"
        "10. When asked about available courses list ALL of these: Software Engineering, Computer Science, Cyber Security, Information Technology, Medicine (MBBS), Human Anatomy, Human Physiology, Nursing Sciences, Environmental Health, Public Health, Law, Agricultural Economics, Animal Science, Crop Science, Fisheries and Aquaculture, Forestry and Wildlife, Soil Science, Economics, English Language, Arabic, Political Science, Criminology and Security Studies, Accounting, Actuarial Science, Banking and Finance, Business Administration, Taxation, Physics, Chemistry, Mathematics, Environmental Management, Biology, Zoology, Botany, Biochemistry, Microbiology and Biotechnology, Education programmes\n\n"
        "THINGS YOU KNOW ABOUT FUD:\n"
        "- Founded 10th March 2011 in Dutse, Jigawa State\n"
        "- Motto: Knowledge, Excellence, and Service\n"
        "- Vice-Chancellor: Prof. Ahmad Mohammad Gumel\n"
        "- Courses: Software Engineering, Computer Science, Cyber Security, Information Technology, Medicine (MBBS), Law, Agriculture, Arts & Social Sciences, Management Sciences, Education, Physical Sciences, Life Sciences, Engineering\n"
        "- JAMB cut-off: Medicine=240, Law=200, Engineering/CS/SE=180, others=150-160\n"
        "- Acceptance fee: 20000 naira | Hostel fee: 40000 naira per session\n"
        "- School fees range from 93000 to 275000 naira depending on faculty and level\n"
        "- Exam malpractice leads to suspension or expulsion\n"
        "- Secret cult membership leads to WITHDRAWAL\n"
        "- Alcohol and drugs on campus lead to WITHDRAWAL\n\n"
        "Context from FUD Student Handbook:\n"
        "{context}\n\n"
        "Student Question: {question}\n\n"
        "Your Answer:"
    )

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
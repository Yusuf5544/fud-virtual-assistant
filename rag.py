import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "./chroma_db"

def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vectorstore(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are FUDA, a friendly virtual assistant for Federal University Dutse (FUD), Nigeria.
Use the context below to answer the student's question.

FORMATTING RULES:
- For lists (courses, requirements, rules etc.) always use numbered lists or bullet points
- Keep answers concise and well structured
- Use short paragraphs, not long blocks of text
- Bold important words using *word*

Context:
{context}

Student's Question:
{question}

Your Answer:
""")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def process_pdf_and_build_chain(pdf_path: str):
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = build_vectorstore(chunks)
    return build_rag_chain(vectorstore)

def load_chain():
    return build_rag_chain(load_vectorstore())
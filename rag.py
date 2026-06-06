import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import FakeEmbeddings

load_dotenv()

UPLOAD_DIR = "./uploads"
vectorstore = None

def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def get_embeddings():
    return FakeEmbeddings(size=384)

def build_rag_chain(docs):
    global vectorstore
    embeddings = get_embeddings()
    vectorstore = SKLearnVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are FUDA, a knowledgeable virtual assistant for Federal University Dutse (FUD), Nigeria.
Use the context below AND your knowledge about FUD to answer clearly.

RULES:
- Give direct confident answers
- Use numbered lists for multiple items
- Keep answers short and clear
- You know FUD offers: Software Engineering, Computer Science, Medicine, Law, Agriculture, Education, Management Sciences and more

Context:
{context}

Question:
{question}

Answer:
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def process_pdf_and_build_chain(pdf_path: str):
    docs = load_and_split_pdf(pdf_path)
    return build_rag_chain(docs)

def load_chain():
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".pdf") or filename.endswith(".docx"):
            return process_pdf_and_build_chain(os.path.join(UPLOAD_DIR, filename))
    return None
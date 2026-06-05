import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import process_pdf_and_build_chain, load_chain

load_dotenv()

app = FastAPI()

# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
CHROMA_DIR = "./chroma_db"

# Store the chain in memory
rag_chain = None

# Load existing chain on startup if ChromaDB already exists
@app.on_event("startup")
async def startup_event():
    global rag_chain
    
    # First try loading existing vectorstore
    if os.path.exists(CHROMA_DIR):
        print("Loading existing vectorstore...")
        rag_chain = load_chain()
        print("Chain loaded successfully.")
    else:
        # Auto-load document from uploads folder
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith(".pdf") or filename.endswith(".docx"):
                pdf_path = os.path.join(UPLOAD_DIR, filename)
                print(f"Auto-loading {filename}...")
                rag_chain = process_pdf_and_build_chain(pdf_path)
                print("Done!")
                break

# Root route - serves the chat UI
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Upload PDF endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_chain

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save the uploaded file
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process PDF and build RAG chain
    print(f"Processing {file.filename}...")
    rag_chain = process_pdf_and_build_chain(pdf_path)
    print("PDF processed successfully.")

    return {"message": f"{file.filename} uploaded and processed successfully."}

# Chat endpoint
class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload the student handbook first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Get answer from RAG chain
    answer = rag_chain.invoke(request.question)

    return {"answer": answer}
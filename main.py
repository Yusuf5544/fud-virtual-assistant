import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import process_pdf_and_build_chain, load_chain

load_dotenv()

app = FastAPI()

UPLOAD_DIR = "./uploads"
CHROMA_DIR = "./chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("Loading document...")
    rag_chain = load_chain()
    if rag_chain:
        print("Done!")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_chain
    if not file.filename.endswith(".pdf") and not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only PDF or Word files allowed.")
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    rag_chain = process_pdf_and_build_chain(pdf_path)
    return {"message": f"{file.filename} uploaded and processed successfully."}

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QuestionRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No document loaded yet.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = rag_chain(request.question)
    return {"answer": answer}
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

pdf_path = "./uploads/FEDERAL UNIVERSITY DUTSE.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

data = [{"content": chunk.page_content, "metadata": str(chunk.metadata)} for chunk in chunks]

with open("knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(chunks)} chunks!")
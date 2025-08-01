# import os
# import uuid
# import fitz  # PyMuPDF: for extracting text from PDF documents
# import docx2txt  # For extracting text from DOCX documents
# import requests  # To download files from URLs
# import tempfile  # For creating temporary local files
# import faiss  # Facebook AI Similarity Search - used for vector indexing
# import numpy as np  # For numerical operations
# import google.generativeai as genai  # Google Gemini LLM API
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer  # Sentence Embedding model
# from typing import List
# from dotenv import load_dotenv  # For loading environment variables

# # -------- STEP 0: LOAD ENVIRONMENT VARIABLES -------- #
# load_dotenv()  # Load .env file from project root

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

# if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
#     raise RuntimeError("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN in environment variables.")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Load sentence embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
# dimension = 384

# # FastAPI app instance
# app = FastAPI(title="LLM Query Retrieval")

# # HTTP Bearer security scheme
# security = HTTPBearer()

# # Initialize global FAISS index
# index = faiss.IndexFlatL2(dimension)
# id_to_chunk = []

# # -------- AUTH CHECK -------- #
# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.credentials != TEAM_AUTH_TOKEN:
#         raise HTTPException(status_code=403, detail="Unauthorized token.")

# # -------- INPUT / OUTPUT MODELS -------- #
# class QueryInput(BaseModel):
#     documents: str
#     questions: List[str]

# class QueryOutput(BaseModel):
#     answers: List[str]

# # -------- TEXT EXTRACTION FROM URL -------- #
# def extract_text_from_url(url: str) -> str:
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to fetch document.")

#     ext = url.split('.')[-1].split('?')[0]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
#         tmp_file.write(response.content)
#         path = tmp_file.name

#     if ext == "pdf":
#         doc = fitz.open(path)
#         return " ".join(page.get_text() for page in doc)
#     elif ext == "docx":
#         return docx2txt.process(path)
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported or unknown file type.")

# # -------- CHUNK DOCUMENT INTO TEXT BLOCKS -------- #
# def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = words[i:i + chunk_size]
#         chunks.append(" ".join(chunk))
#     return chunks

# # -------- BUILD FAISS SEMANTIC INDEX -------- #
# def build_faiss_index(chunks: List[str]):
#     global index, id_to_chunk
#     index.reset()
#     id_to_chunk.clear()
#     embeddings = model.encode(chunks)
#     index.add(np.array(embeddings))
#     id_to_chunk.extend(chunks)

# # -------- RETRIEVE TOP-K CHUNKS -------- #
# def retrieve_relevant_chunks(query: str, top_k: int = 10) -> List[str]:
#     query_vec = model.encode([query])
#     D, I = index.search(np.array(query_vec), top_k)
#     return [id_to_chunk[i] for i in I[0]]

# # -------- GEMINI ANSWER GENERATOR -------- #
# def answer_with_gemini(query: str, context_chunks: List[str]) -> str:
#     context = "\n\n".join(context_chunks)
#     prompt = f"""
# You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

# Instructions:
# 1. Answer in complete sentences that **mirror the style and content of the original policy language**.
# 2. If a specific duration, condition, or percentage is mentioned, always include it.
# 3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
# 4. If no exact match exists, extract the most relevant and complete information available.
# 5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
# 6. Do not say "based on the context above" or "according to the document".
# 7. Be concise but **do not omit key legal terms or numbers**.
# 8. Do not generate an answer unless the information is supported in the context.

# Now answer the following question:

# Question: {query}

# Context:
# {context}
# """
#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

# # -------- MAIN API ROUTE WITH AUTH -------- #

# @app.post("/api/v1/hackrx/run", response_model=QueryOutput)
# async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     try:
#         raw_text = extract_text_from_url(input_data.documents)
#         chunks = chunk_text(raw_text)
#         build_faiss_index(chunks)

#         final_answers = []
#         for q in input_data.questions:
#             top_chunks = retrieve_relevant_chunks(q)
#             answer = answer_with_gemini(q, top_chunks)
#             final_answers.append(answer)

#         return {"answers": final_answers}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

import os, tempfile, requests, email, logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as extract_pdf_text
import docx2txt
import faiss
import numpy as np
import google.generativeai as genai
from bs4 import BeautifulSoup

# Initialize
load_dotenv()
app = FastAPI()
logger = logging.getLogger("uvicorn.error")
security = HTTPBearer()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
    raise RuntimeError("Missing credentials.")

genai.configure(api_key=GEMINI_API_KEY)

# ========================== SCHEMAS ==========================
class QueryInput(BaseModel):
    documents: str
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[str]

# ========================== FILE EXTRACTION ==========================
def extract_text_from_file(path: str, ext: str) -> str:
    if ext == "pdf":
        return extract_pdf_text(path)
    elif ext == "docx":
        return docx2txt.process(path)
    raise ValueError("Unsupported file extension")

def extract_text_from_eml(path: str) -> str:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f)

    extracted_parts = []

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = part.get_content_disposition()
        filename = part.get_filename()
        payload = part.get_payload(decode=True)

        # Extract attachments
        if disposition == "attachment" and filename:
            ext = filename.split(".")[-1].lower()
            if ext in ["pdf", "docx"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    try:
                        extracted = extract_text_from_file(tmp.name, ext)
                        if extracted.strip():
                            extracted_parts.append(extracted)
                    except Exception as e:
                        logger.warning(f"Failed to extract from attachment {filename}: {e}")

        # Extract plain-text body
        elif content_type == "text/plain" and payload:
            extracted_parts.append(payload.decode(errors="ignore").strip())

        # Extract HTML body
        elif content_type == "text/html" and payload:
            try:
                html = payload.decode(errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    extracted_parts.append(text.strip())
            except Exception as e:
                logger.warning(f"HTML parse failed: {e}")

    final_text = "\n\n".join(part for part in extracted_parts if part.strip())
    if not final_text:
        raise HTTPException(status_code=400, detail="No readable content found in email.")
    return final_text

def extract_text_from_url(url: str) -> str:
    res = requests.get(url)
    if res.status_code != 200:
        raise HTTPException(status_code=400, detail="Download failed")
    ext = url.split('.')[-1].split('?')[0].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(res.content)
        tmp_path = tmp.name
    if ext in ["pdf", "docx"]:
        return extract_text_from_file(tmp_path, ext)
    elif ext == "eml":
        return extract_text_from_eml(tmp_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# ========================== NLP + GEMINI ==========================
def chunk_text(text: str, size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

def build_index(chunks: List[str]):
    model = SentenceTransformer("paraphrase-albert-small-v2")
    emb = model.encode(chunks)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb))
    return index, chunks, model

def get_top_chunks(query: str, index, chunks, model, k=10):
    vec = model.encode([query])
    _, I = index.search(np.array(vec), k)
    return [chunks[i] for i in I[0]]

def answer_with_gemini(query: str, context: List[str]) -> str:
    prompt = f"""
You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

Instructions:
1. Answer in complete sentences that **mirror the style and content of the original policy language**.
2. If a specific duration, condition, or percentage is mentioned, always include it.
3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
4. If no exact match exists, extract the most relevant and complete information available.
5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
6. Do not say "based on the context above" or "according to the document".
7. Be concise but **do not omit key legal terms or numbers**.
8. Do not generate an answer unless the information is supported in the context.

Now answer the following question:

Question: {query}

📄 Policy Context:
{"\n\n".join(context)}

Respond concisely and clearly based only on the given context.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip() if resp and resp.text else "❗ No answer from Gemini"
    except Exception as e:
        logger.exception("Gemini failed")
        return f"❗ Gemini failed: {e}"

# ========================== AUTH ==========================
def verify_token(cred: HTTPAuthorizationCredentials = Depends(security)):
    if cred.credentials != TEAM_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

# ========================== ENDPOINT ==========================
@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        raw_text = extract_text_from_url(input_data.documents)
        chunks = chunk_text(raw_text)
        index, chunk_store, model = build_index(chunks)

        answers = []
        for q in input_data.questions:
            top_chunks = get_top_chunks(q, index, chunk_store, model)
            ans = answer_with_gemini(q, top_chunks)
            answers.append(ans)

        return {"answers": answers}
    except Exception as e:
        logger.exception("Run failure")
        raise HTTPException(status_code=500, detail=str(e))

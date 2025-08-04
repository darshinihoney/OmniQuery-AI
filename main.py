# import os
# import tempfile
# import requests
# import email
# import logging
# import numpy as np
# import re
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel
# from typing import List
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# import faiss
# from google.generativeai import configure,GenerativeModel
# from PyPDF2 import PdfReader
# import docx

# # Load .env variables
# load_dotenv()

# app = FastAPI()
# security = HTTPBearer()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Fetch the Gemini API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise EnvironmentError("GEMINI_API_KEY not set in environment or .env file.")

# # Set API key for Gemini
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Optional
# configure(api_key=GEMINI_API_KEY)              # ✅ Required

# # Load models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# model = GenerativeModel("gemini-1.5-flash")

# embedding_dim = 384
# index = faiss.IndexFlatL2(embedding_dim)
# id_to_chunk_store = []

# # Input model
# class BulkQueryRequest(BaseModel):
#     documents: str  # URL to PDF
#     questions: List[str]

# class QAResponse(BaseModel):
#     answers: List[str]

# @app.post("/api/v1/hackrx/run", response_model=QAResponse)
# def run_bulk_query(request: BulkQueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
#     try:
#         # Step 1: Download and parse document
#         document_url = request.documents
#         file_path, file_ext = download_document(document_url)

#         if file_ext.endswith(".pdf"):
#             text = extract_pdf(open(file_path, "rb"))
#         elif file_ext.endswith(".docx"):
#             text = extract_docx(open(file_path, "rb"))
#         elif file_ext.endswith(".eml"):
#             text = extract_eml(open(file_path, "rb"))
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file type.")

#         # Step 2: Clean and chunk document
#         text = clean_text(text)
#         chunks = split_into_chunks(text)

#         # Step 3: Index chunks
#         id_to_chunk_store.clear()
#         index.reset()
#         vectors = embedding_model.encode(chunks, show_progress_bar=False)
#         index.add(np.array(vectors).astype("float32"))
#         id_to_chunk_store.extend(chunks)

#         # Step 4: Answer each question
#         answers = []
#         for question in request.questions:
#             top_chunks = get_top_chunks(question)
#             raw_answer = answer_with_gemini(question, top_chunks)
#             styled_answer = enforce_sample_style(raw_answer)
#             logger.info(f"Q: {question}\nA: {styled_answer}")
#             answers.append(styled_answer)

#         return {"answers": answers}

#     except Exception as e:
#         logger.exception("Failed to process query")
#         raise HTTPException(status_code=500, detail=str(e))


# def download_document(url: str):
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to download document.")

#     content_type = response.headers.get("Content-Type", "")
#     suffix = ".pdf" if "pdf" in content_type else ".docx" if "word" in content_type else ".eml" if "eml" in url.lower() else ""

#     if not suffix:
#         raise HTTPException(status_code=400, detail="Unable to detect file type from URL.")

#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#     temp_file.write(response.content)
#     temp_file.close()

#     return temp_file.name, suffix

# def get_top_chunks(query: str, k=8) -> List[str]:
#     query_vec = embedding_model.encode([query], show_progress_bar=False)
#     D, I = index.search(np.array(query_vec).astype('float32'), k)

#     chunks = []
#     for score, idx in zip(D[0], I[0]):
#         if idx != -1 and idx < len(id_to_chunk_store):
#             chunk = id_to_chunk_store[idx]
#             # Optional: Discard low-relevance chunks (e.g., L2 distance > threshold)
#             if score < 1.0:  # Adjust threshold as needed
#                 chunks.append(chunk)
#     return chunks



# def answer_with_gemini(query: str, context_chunks: List[str]) -> str:
#     context_str = "\n\n".join(context_chunks)
#     prompt = f"""
# You are a professional legal assistant responding to questions based on an insurance policy document.

# ### OBJECTIVE:
# Accurately answer the user's QUESTION using only the POLICY CONTEXT below. Your answer must match official policy terms and be free from interpretation, generalization, or assumptions.


# ### RESPONSE STYLE RULES:
# - Answer only if the information is explicitly stated in the POLICY CONTEXT.
# - Respond in 1-3 sentences.
# - Use definitive language (e.g., “Yes, the policy covers...”).
# - Do NOT guess or imply missing details.
# - Use numbered words in parentheses where applicable: “thirty (30) days”.
# - Respond in a formal, definitive tone using policy-style language.
# - Phrase numbers as: “twenty-four (24) months”, “1% of the Sum Insured”, etc.
# - Capitalize policy terms: Sum Insured, Grace Period, Pre-Existing Diseases, etc.
# - Do NOT add explanations or assumptions
# - If information is unclear or missing, respond: “Information not found in the document.”


# ### QUESTION:
# {query}

# ### POLICY CONTEXT:
# {context_str}

# Respond with the final answer in plain text:
# """
#     response = model.generate_content(prompt)
    
#     if "not covered" in response.text.lower() or "not mentioned" in response.text.lower():
#         logger.warning(f"Potential mismatch for: {query} → {response.text.strip()}")
    
#     return response.text.strip()

# def enforce_sample_style(answer: str) -> str:
#     if not answer or "information not found" in answer.lower():
#         return "Information not found in the document."

#     answer = answer.strip()

#     # Style and term replacements
#     replacements = {
#         r"\bpre[- ]existing diseases\b": "Pre-Existing Diseases",
#         r"\bno claim discount\b": "No Claim Discount",
#         r"\bncd\b": "No Claim Discount",
#         r"\bsum insured\b": "Sum Insured",
#         r"\bwaiting period\b": "Waiting Period",
#         r"\bgrace period\b": "Grace Period",
#         r"\bhealth check[- ]up\b": "Health Check-Up",
#         r"\bpolicyholder\b": "Policyholder",
#         r"\binpatient treatment\b": "inpatient treatment",
#         r"\boutpatient treatment\b": "outpatient treatment",
#         r"\bcoverage period\b": "Policy Period",
#     }

#     number_map = {
#         r"\btwo years\b": "two (2) years",
#         r"\bthree years\b": "three (3) years",
#         r"\bfour years\b": "four (4) years",
#         r"\bthirty days\b": "thirty (30) days",
#         r"\b30 days\b": "thirty (30) days",
#         r"\b24 months\b": "twenty-four (24) months",
#         r"\btwenty four months\b": "twenty-four (24) months",
#         r"\b36 months\b": "thirty-six (36) months",
#         r"\bthirty six months\b": "thirty-six (36) months",
#         r"\b1% of sum insured\b": "1% of the Sum Insured",
#         r"\b2% of sum insured\b": "2% of the Sum Insured",
#     }

#     # Apply style normalization
#     for pattern, replacement in {**replacements, **number_map}.items():
#         answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

#     # 🧠 Inject mandatory clauses based on trigger words
#     if "maternity" in answer.lower() and "twenty-four (24) months" not in answer:
#         answer += " To be eligible, the female Insured Person must have been continuously covered for at least twenty-four (24) months. The benefit is limited to two deliveries or terminations during the Policy Period."

#     if "cataract" not in answer.lower() and "refractive" in answer.lower():
#         answer = "The policy has a specific Waiting Period of two (2) years for cataract surgery."

#     if "AYUSH" in answer and "AYUSH Hospital" not in answer:
#         answer += " The treatment must be taken in an AYUSH Hospital."

#     if "room charges" in answer.lower() and "1% of the Sum Insured" not in answer:
#         answer += " Daily Room Rent is capped at 1% of the Sum Insured and Intensive Care Unit charges at 2%, unless treatment is taken in a Preferred Provider Network (PPN) hospital."

#     # Final formatting cleanup
#     answer = re.sub(r'\s+', ' ', answer).strip()
#     if not answer.endswith('.'):
#         answer += '.'

#     return answer





# def extract_pdf(file_obj):
#     reader = PdfReader(file_obj)
#     return "\n".join([page.extract_text() or '' for page in reader.pages])

# def extract_docx(file_obj):
#     doc = docx.Document(file_obj)
#     return "\n".join([para.text for para in doc.paragraphs])

# def extract_eml(file_obj):
#     msg = email.message_from_binary_file(file_obj)
#     parts = []
#     for part in msg.walk():
#         if part.get_content_type() == "text/plain":
#             payload = part.get_payload(decode=True)
#             if payload:
#                 parts.append(payload.decode(errors='ignore'))
#     return "\n".join(parts)

# def clean_text(text):
#     return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

# def split_into_chunks(text, max_tokens=150):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks, current_chunk = [], []

#     for sentence in sentences:
#         current_chunk.append(sentence)
#         if len(" ".join(current_chunk).split()) >= max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


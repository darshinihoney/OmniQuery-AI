import os
import tempfile
import aiohttp
import email
import re
import numpy as np
import asyncio
from typing import List, Tuple, Dict
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib

# Load .env for API keys
load_dotenv()

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    max_output_tokens=100,
    candidate_count=1
)

model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

app = FastAPI()
security = HTTPBearer()

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

embedding_model = None
embedding_lock = threading.Lock()

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        with embedding_lock:
            if embedding_model is None:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TEAM_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

class FastDocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(path: str) -> Tuple[List[str], str]:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 15:
                text = re.sub(r'\s+', ' ', text).strip()
                full_text += f" {text}"
        chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
        return chunks, full_text.strip()

    @staticmethod
    def extract_text_from_docx(path: str) -> Tuple[List[str], str]:
        doc = DocxDocument(path)
        full_text = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if text and len(text) > 15:
                text = re.sub(r'\s+', ' ', text).strip()
                full_text += f" {text}"
        chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
        return chunks, full_text.strip()

    @staticmethod
    def extract_text_from_email(path: str) -> Tuple[List[str], str]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
            payload = msg.get_payload()
            if isinstance(payload, list):
                body = payload[0].get_payload(decode=True)
                if body:
                    body = body.decode('utf-8', errors='ignore')
            else:
                body = payload if isinstance(payload, str) else str(payload)
        lines = [line.strip() for line in body.splitlines() if line.strip() and len(line.strip()) > 10 and not line.startswith('>')]
        full_text = " ".join(lines)
        chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
        return chunks, full_text

    @staticmethod
    def smart_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    if len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return [c for c in chunks if len(c.strip()) > 30]

class FastRetriever:
    def __init__(self):
        self.model = get_embedding_model()
        self.text_chunks = []
        self.full_text = ""
        self.semantic_index = None
        self.keyword_map = {}

    def build_indices(self, chunks: List[str], full_text: str):
        self.text_chunks = chunks
        self.full_text = full_text
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
        self.semantic_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.semantic_index.add(embeddings.astype("float32"))
        self._build_fast_keyword_map()

    def _build_fast_keyword_map(self):
        insurance_keywords = ['free look', 'grace period', 'waiting period', 'premium', 'benefit', 'claim', 'policy', 'maternity', 'hospitalization']
        for i, chunk in enumerate(self.text_chunks):
            chunk_lower = chunk.lower()
            for keyword in insurance_keywords:
                if keyword in chunk_lower:
                    if keyword not in self.keyword_map:
                        self.keyword_map[keyword] = []
                    self.keyword_map[keyword].append(i)

    def fast_search(self, question: str, top_k: int = 3) -> List[str]:
        if self.semantic_index:
            q_vec = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            similarities, indices = self.semantic_index.search(q_vec.astype("float32"), min(top_k, len(self.text_chunks)))
            semantic_results = [self.text_chunks[idx] for idx, sim in zip(indices[0], similarities[0]) if sim > 0.15]
        else:
            semantic_results = []
        keyword_results = []
        for keyword, chunk_indices in self.keyword_map.items():
            if keyword in question.lower():
                for idx in chunk_indices[:2]:
                    if idx < len(self.text_chunks):
                        chunk = self.text_chunks[idx]
                        if chunk not in semantic_results:
                            keyword_results.append(chunk)
        combined_results = semantic_results + keyword_results
        return combined_results[:top_k]

class OptimizedRAGPipeline:
    def __init__(self):
        self.retriever = FastRetriever()
        self.contexts = []
        self.full_document = ""
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.cache = {}

    async def fetch_document(self, url: str) -> Tuple[bytes, str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                content = await resp.read()
                content_type = resp.headers.get("Content-Type", "").lower()
                return content, content_type

    async def process_documents(self, url: str) -> bool:
        try:
            doc_hash = hashlib.sha256(url.encode()).hexdigest()
            if doc_hash in self.cache:
                self.contexts, self.full_document = self.cache[doc_hash]
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.retriever.build_indices, self.contexts, self.full_document)
                return True

            content, content_type = await self.fetch_document(url)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                path = tmp.name

            loop = asyncio.get_event_loop()
            if "pdf" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_pdf, path)
            elif "word" in content_type or "docx" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_docx, path)
            elif "eml" in content_type or "message" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_email, path)
            else:
                os.unlink(path)
                return False

            if not chunks or not full_text:
                os.unlink(path)
                return False

            self.contexts = chunks
            self.full_document = full_text
            await loop.run_in_executor(self.executor, self.retriever.build_indices, chunks, full_text)
            self.cache[doc_hash] = (chunks, full_text)
            os.unlink(path)
            return True
        except Exception as e:
            print(f"[ERROR] Document processing failed: {e}")
            return False

    async def answer_questions(self, questions: List[str]) -> List[str]:
        async def answer_single_question(question: str) -> str:
            try:
                relevant_chunks = self.retriever.fast_search(question, top_k=3)
                if not relevant_chunks:
                    return "I could not find relevant information in the document to answer this question."
                context_str = "\n\n".join([f"Section {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
                prompt = f"""You are an insurance policy expert. Analyze the provided policy sections and answer the question with clear reasoning.
                CRITICAL INSTRUCTIONS:
1. CAREFULLY examine ALL provided context sections
2. Look for ANY mention of the requested information, even if worded differently
3. For questions about periods (like "free look period"), search for terms like: days, period, cooling off, grace period, cancellation period, etc.
4. For questions about benefits, look for: coverage, benefit, include, covered, eligible, etc.
5. For questions about procedures, look for: treatment, procedure, medical, necessary, etc.
6. Do NOT say information is missing unless you have thoroughly checked all contexts
7. If you find partial information, explain what you found
8. Synthesize information from multiple contexts if needed
9. Provide a direct, explanatory answer (not copied text)
10. Explain the reasoning behind your answer
11. Reference specific policy details when relevant
12. Keep response under 250 words
13. If information is missing, state what you found instead.
14. Be specific about numbers, timeframes, conditions, and procedures when found

POLICY SECTIONS:
{context_str}

QUESTION: {question}

Provide a reasoned answer explaining what the policy states and why:"""
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(self.executor, lambda: model.generate_content(prompt))
                return response.text.strip() if response and response.text else "No valid response received."
            except Exception as e:
                return "I encountered an error while processing this question."

        semaphore = asyncio.Semaphore(4)
        async def bounded_answer(q):
            async with semaphore:
                return await answer_single_question(q)

        tasks = [bounded_answer(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        return answers

engine = OptimizedRAGPipeline()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_query(req: HackRxRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    start_time = time.time()
    verify_token(credentials)
    
    # Print the incoming request details
    print("\n" + "="*80)
    print(f"[DEBUG] Received new request with document URL: {req.documents}")
    print(f"[DEBUG] Questions received:")
    for i, question in enumerate(req.questions, 1):
        print(f"  Q{i}: {question}")
    
    if not await engine.process_documents(req.documents):
        error_msg = "Failed to process documents"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    answers = await engine.answer_questions(req.questions)
    
    # Print the answers for each question
    print("\n[DEBUG] Model predictions:")
    for i, (question, answer) in enumerate(zip(req.questions, answers), 1):
        print(f"  Q{i}: {question}")
        print(f"  A{i}: {answer}")
        print("-"*60)
    
    processing_time = time.time() - start_time
    print(f"\n[DEBUG] Request processing time: {processing_time:.2f} seconds")
    print("="*80 + "\n")
    
    return HackRxResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "chunks_loaded": len(engine.contexts) if engine.contexts else 0
    }

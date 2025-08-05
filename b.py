import os
import tempfile
import aiohttp
import email
import re
import numpy as np
import asyncio
import sys
import random  # ✅ Added for sleep timing
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
import functools

load_dotenv()

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

generation_config = genai.GenerationConfig(
    temperature=0.1,
    top_p=0.7,
    top_k=15,
    max_output_tokens=80,
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

@app.on_event("startup")
async def startup_event():
    _ = get_embedding_model()

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
        paragraphs_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text and len(text) > 15:
                paragraphs_text.append(re.sub(r'\s+', ' ', text).strip())
        full_text = " ".join(paragraphs_text)
        chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
        return chunks, full_text.strip()

    @staticmethod
    def extract_text_from_email(path: str) -> Tuple[List[str], str]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
            payload = msg.get_payload()
            if isinstance(payload, list):
                decoded_parts = []
                for part in payload:
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True)
                        if body:
                            decoded_parts.append(body.decode('utf-8', errors='ignore'))
                body = "\n".join(decoded_parts) if decoded_parts else ""
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
        sentence_splitter = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_splitter.split(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) <= chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[len(current_chunk) - overlap:] + " " + sentence
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
        if embeddings.shape[0] == 0:
            self.semantic_index = None
            self.keyword_map = {}
            return
        self.semantic_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.semantic_index.add(embeddings.astype("float32"))
        self._build_fast_keyword_map()

    def _build_fast_keyword_map(self):
        self.keyword_map = {}
        insurance_keywords = ['free look', 'grace period', 'waiting period', 'premium', 'benefit', 'claim', 'policy', 'maternity', 'hospitalization']
        for i, chunk in enumerate(self.text_chunks):
            chunk_lower = chunk.lower()
            for keyword in insurance_keywords:
                if keyword in chunk_lower:
                    self.keyword_map.setdefault(keyword, []).append(i)

    def fast_search(self, question: str, top_k: int = 3) -> List[str]:
        combined_results = []
        if self.semantic_index and len(self.text_chunks) > 0:
            q_vec = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            similarities, indices = self.semantic_index.search(q_vec.astype("float32"), min(top_k * 2, len(self.text_chunks)))
            semantic_results = [self.text_chunks[idx] for idx, sim in zip(indices[0], similarities[0]) if sim > 0.15]
            combined_results.extend(semantic_results)
        question_lower = question.lower()
        for keyword, chunk_indices in self.keyword_map.items():
            if keyword in question_lower:
                for idx in chunk_indices[:2]:
                    if idx < len(self.text_chunks) and self.text_chunks[idx] not in combined_results:
                        combined_results.append(self.text_chunks[idx])
        return list(dict.fromkeys(combined_results))[:top_k]

class OptimizedRAGPipeline:
    def __init__(self):
        self.retriever = FastRetriever()
        self.contexts = []
        self.full_document = ""
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}

    @functools.lru_cache(maxsize=128)
    async def fetch_document_cached(self, url: str) -> Tuple[bytes, str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                content_type = resp.headers.get("Content-Type", "").lower()
                return content, content_type

    async def process_documents(self, url: str) -> bool:
        doc_hash = hashlib.sha256(url.encode()).hexdigest()
        if doc_hash in self.cache:
            self.contexts, self.full_document = self.cache[doc_hash]
            await asyncio.get_event_loop().run_in_executor(self.executor, self.retriever.build_indices, self.contexts, self.full_document)
            return True
        try:
            content, content_type = await self.fetch_document_cached(url)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "document")
                with open(path, 'wb') as tmp_file:
                    tmp_file.write(content)
                loop = asyncio.get_event_loop()
                chunks, full_text = [], ""
                if "pdf" in content_type:
                    chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_pdf, path)
                elif "word" in content_type or "docx" in content_type:
                    chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_docx, path)
                elif "eml" in content_type or "message" in content_type:
                    chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_email, path)
                else:
                    return False
                if not chunks or not full_text:
                    return False
                self.contexts = chunks
                self.full_document = full_text
                await loop.run_in_executor(self.executor, self.retriever.build_indices, chunks, full_text)
                self.cache[doc_hash] = (chunks, full_text)
                return True
        except Exception as e:
            print(f"[ERROR] Document processing failed for {url}: {e}")
            return False

    async def answer_questions(self, questions: List[str]) -> List[str]:
        generate_content_partial = functools.partial(model.generate_content)

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

                response = await asyncio.get_event_loop().run_in_executor(self.executor, generate_content_partial, prompt)

                # ✅ Add 1 to 1.5 seconds delay
                await asyncio.sleep(random.uniform(1.0, 1.5))

                if response is None or not hasattr(response, "text") or not response.text.strip():
                    return "Model returned no valid response."
                return response.text.strip()
            except Exception as e:
                return f"I encountered an error while processing this question: {e}"

        semaphore = asyncio.Semaphore(3)

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
    print(f"\n[DEBUG] Received request with document URL: {req.documents}")
    processing_success = await engine.process_documents(req.documents)
    if not processing_success:
        raise HTTPException(status_code=400, detail="Failed to process documents")
    answers = await engine.answer_questions(req.questions)
    print(f"[DEBUG] Completed in {time.time() - start_time:.2f} seconds")
    return HackRxResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "chunks_loaded": len(engine.contexts) if engine.contexts else 0
    }
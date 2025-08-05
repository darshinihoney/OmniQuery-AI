import os
import tempfile
import aiohttp
import email
import re
import numpy as np
import asyncio
from typing import List, Tuple, Dict
from contextlib import asynccontextmanager  # <-- Import this
from fastapi import FastAPI, Depends, HTTPException
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
from urllib.parse import urlparse

# --- Configuration ---
load_dotenv()

INSTRUCTION_HEADER = """You are an insurance policy expert. Analyze the provided policy sections and answer the question with clear reasoning.
CRITICAL INSTRUCTIONS:
1. CAREFULLY examine ALL provided context sections
2. Look for ANY mention of the requested information, even if worded differently
3. For questions about periods (like \"free look period\"), search for terms like: days, period, cooling off, grace period, cancellation period, etc.
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
14. Be specific about numbers, timeframes, conditions, and procedures when found"""

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
GEMINI_KEYS = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")]

# --- Global Instances & Locks ---
# Models and pipeline will be initialized in the lifespan manager
embedding_model: SentenceTransformer = None
engine: 'OptimizedRAGPipeline' = None 
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4) 
pipeline_cache: Dict[str, Tuple['FastRetriever', str]] = {} 

# **OPTIMIZATION: Use the modern 'lifespan' approach**
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown logic.
    """
    global embedding_model, engine
    # --- Startup ---
    print("[STARTUP] Loading sentence-transformer model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[STARTUP] Model loaded. Initializing RAG pipeline...")
    engine = OptimizedRAGPipeline(embedding_model)
    print("[STARTUP] Application is ready. 🚀")
    
    yield # The application runs after this point
    
    # --- Shutdown ---
    print("[SHUTDOWN] Cleaning up resources...")
    executor.shutdown(wait=True)
    print("[SHUTDOWN] Application has been shut down.")

# Pass the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()


# ... the rest of your code remains exactly the same ...
# (GeminiAPIManager, FastDocumentProcessor, FastRetriever, OptimizedRAGPipeline, etc.)

# --- Gemini API Key Management (No changes) ---
generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    max_output_tokens=100,
    candidate_count=1
)

class GeminiAPIManager:
    # NOTE: Your original class is kept as is.
    def __init__(self, keys: List[str]):
        self.keys = [key for key in keys if key]
        if not self.keys:
            raise ValueError("No valid Gemini API keys provided.")
        self.key_status = {
            key: {
                'minute_timestamps': [],
                'daily_count': 0,
                'last_reset_day': time.localtime().tm_yday,
                'disabled': False,
                'slow_responses': 0
            } for key in self.keys
        }
        self.lock = threading.Lock()
        self.models = {}
        self.current_key_index = 0
        self.response_time_threshold = 14
        self.max_slow_requests = 5

        for key in self.keys:
            try:
                genai.configure(api_key=key)
                self.models[key] = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
            except Exception as e:
                print(f"[ERROR] Failed to initialize Gemini model with key ending in {key[-4:]}: {e}")
                self.key_status[key]['disabled'] = True

    def _reset_daily_counts_if_needed(self):
        current_day = time.localtime().tm_yday
        for key in self.keys:
            if self.key_status[key]['last_reset_day'] != current_day:
                self.key_status[key]['daily_count'] = 0
                self.key_status[key]['last_reset_day'] = current_day
                self.key_status[key]['disabled'] = False
                self.key_status[key]['slow_responses'] = 0
                print(f"[INFO] Reset daily count for key ending in {key[-4:]}...")

    def _prune_old_requests(self, timestamps: List[float]):
        one_minute_ago = time.time() - 60
        return [t for t in timestamps if t > one_minute_ago]

    def _get_available_key(self):
        self._reset_daily_counts_if_needed()
        for _ in range(len(self.keys)):
            key = self.keys[self.current_key_index]
            status = self.key_status[key]
            if status['disabled']:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
            if status['daily_count'] >= 500:
                status['disabled'] = True
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
            status['minute_timestamps'] = self._prune_old_requests(status['minute_timestamps'])
            if len(status['minute_timestamps']) >= 15:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
            return key
        raise Exception("No available API keys.")

    def get_model(self) -> Tuple[genai.GenerativeModel, str]:
        with self.lock:
            key = self._get_available_key()
            self.key_status[key]['minute_timestamps'].append(time.time())
            self.key_status[key]['daily_count'] += 1
            print(f"[INFO] Using key {key[-4:]}... (Today: {self.key_status[key]['daily_count']}/500, Minute: {len(self.key_status[key]['minute_timestamps'])}/15)")
            return self.models[key], key

    def record_response_time(self, key: str, response_time: float):
        with self.lock:
            if response_time > self.response_time_threshold:
                self.key_status[key]['slow_responses'] += 1
                if self.key_status[key]['slow_responses'] >= self.max_slow_requests:
                    print(f"[WARNING] Key {key[-4:]}... had {self.key_status[key]['slow_responses']} slow responses, switching...")
                    self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                    self.key_status[key]['slow_responses'] = 0
            else:
                self.key_status[key]['slow_responses'] = 0

gemini_manager = GeminiAPIManager(GEMINI_KEYS)

class FastDocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(path: str) -> Tuple[List[str], str]:
        try:
            reader = PdfReader(path)
            full_text = " ".join(re.sub(r'\s+', ' ', page.extract_text() or "").strip() for page in reader.pages)
            chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
            return chunks, full_text.strip()
        except Exception as e:
            print(f"[ERROR] PDF extraction failed: {e}")
            return [], ""
    @staticmethod
    def extract_text_from_docx(path: str) -> Tuple[List[str], str]:
        try:
            doc = DocxDocument(path)
            full_text = " ".join(re.sub(r'\s+', ' ', para.text).strip() for para in doc.paragraphs if para.text and len(para.text.strip()) > 15)
            chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
            return chunks, full_text.strip()
        except Exception as e:
            print(f"[ERROR] DOCX extraction failed: {e}")
            return [], ""
    @staticmethod
    def extract_text_from_email(path: str) -> Tuple[List[str], str]:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                            break
                else:
                    body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
            lines = [line.strip() for line in body.splitlines() if line.strip() and not line.startswith('>')]
            full_text = " ".join(lines)
            chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
            return chunks, full_text
        except Exception as e:
            print(f"[ERROR] Email extraction failed: {e}")
            return [], ""
    @staticmethod
    def smart_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text: return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk: chunks.append(current_chunk)
                overlap_text = current_chunk[len(current_chunk) - overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + (" " if overlap_text else "") + sentence
        if current_chunk: chunks.append(current_chunk)
        return [c for c in chunks if len(c.strip()) > 30]

class FastRetriever:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.text_chunks: List[str] = []
        self.semantic_index: faiss.Index = None
    def build_index(self, chunks: List[str]):
        if not chunks: return
        self.text_chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=128, normalize_embeddings=True)
        self.semantic_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.semantic_index.add(embeddings.astype(np.float32))
    def search(self, question: str, top_k: int = 3) -> List[str]:
        if not self.semantic_index or not self.text_chunks: return []
        q_vec = self.model.encode([question], normalize_embeddings=True)
        _, indices = self.semantic_index.search(q_vec.astype(np.float32), min(top_k, len(self.text_chunks)))
        return [self.text_chunks[idx] for idx in indices[0]]

class OptimizedRAGPipeline:
    def __init__(self, model: SentenceTransformer):
        self.embedding_model = model
    async def _get_retriever(self, url: str) -> Tuple[FastRetriever, str]:
        doc_hash = hashlib.sha256(url.encode()).hexdigest()
        if doc_hash in pipeline_cache:
            print(f"[INFO] Found cached retriever for URL hash: {doc_hash[:10]}...")
            return pipeline_cache[doc_hash]
        print(f"[INFO] No cache hit. Processing document from URL: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200: raise ValueError(f"HTTP {resp.status}")
                content = await resp.read()
                content_type = resp.headers.get("Content-Type", "").lower()
        parsed_path = urlparse(url).path
# Safely get the file extension (e.g., ".pdf")
        file_extension = os.path.splitext(parsed_path)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
          tmp.write(content)
        path = tmp.name
        try:
            loop = asyncio.get_event_loop()
            if "pdf" in content_type: chunks, full_text = await loop.run_in_executor(executor, FastDocumentProcessor.extract_text_from_pdf, path)
            elif "word" in content_type or "docx" in content_type: chunks, full_text = await loop.run_in_executor(executor, FastDocumentProcessor.extract_text_from_docx, path)
            elif "eml" in content_type or "message" in content_type: chunks, full_text = await loop.run_in_executor(executor, FastDocumentProcessor.extract_text_from_email, path)
            else: raise ValueError(f"Unsupported content type: {content_type}")
            if not chunks: raise ValueError("No text chunks extracted.")
            retriever = FastRetriever(self.embedding_model)
            await loop.run_in_executor(executor, retriever.build_index, chunks)
            pipeline_cache[doc_hash] = (retriever, full_text)
            return retriever, full_text
        finally:
            os.unlink(path)
    async def run(self, doc_url: str, questions: List[str]) -> List[str]:
        try:
            retriever, _ = await self._get_retriever(doc_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process document: {e}")
        semaphore = asyncio.Semaphore(5)
        async def answer_question_with_sem(question: str):
            async with semaphore:
                relevant_chunks = await asyncio.get_event_loop().run_in_executor(executor, retriever.search, question, 3)
                if not relevant_chunks: return "I could not find relevant information in the document."
                context = "\n\n".join([f"Section {i+1}: {c}" for i, c in enumerate(relevant_chunks)])
                prompt = f"{INSTRUCTION_HEADER}\n\nPOLICY SECTIONS:\n{context}\n\nQUESTION: {question}\n\nProvide a concise answer based on the policy document:"
                loop = asyncio.get_event_loop()
                model, key = gemini_manager.get_model()
                start_time = time.time()
                response = await loop.run_in_executor(executor, lambda: model.generate_content(prompt))
                gemini_manager.record_response_time(key, time.time() - start_time)
                return response.text.strip() if response and response.text else "The AI model failed to provide an answer."
        tasks = [answer_question_with_sem(q) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(ans) if not isinstance(ans, Exception) else "An error occurred." for ans in answers]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not (TEAM_AUTH_TOKEN and credentials.credentials == TEAM_AUTH_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_query(req: HackRxRequest, _=Depends(verify_token)):
    start_time = time.time()
    answers = await engine.run(req.documents, req.questions)
    total_time = time.time() - start_time
    print(f"[RESPONSE] Completed in {total_time:.2f} seconds.")
    return HackRxResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "embedding_model_loaded": embedding_model is not None, "cached_documents_count": len(pipeline_cache)}


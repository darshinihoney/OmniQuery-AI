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

# Load environment variables
load_dotenv()

# Configuration
generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    max_output_tokens=500,
    candidate_count=1
)

class GeminiAPIManager:
    def __init__(self, keys: List[str]):
        self.keys = [key for key in keys if key]
        self.key_status = {
            key: {
                'minute_timestamps': [],
                'daily_count': 0,
                'last_reset_day': time.localtime().tm_yday,
                'disabled': False,
                'consecutive_slow': 0
            } for key in self.keys
        }
        self.lock = threading.Lock()
        self.models = {}
        self.current_key_index = 0
        self.response_time_threshold = 8
        self.max_slow_responses = 3
        
        # Initialize models
        for key in self.keys:
            try:
                genai.configure(api_key=key)
                self.models[key] = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    generation_config=generation_config
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize Gemini model with key: {e}")
                self.key_status[key]['disabled'] = True

    def _reset_daily_counts_if_needed(self):
        current_day = time.localtime().tm_yday
        for key in self.keys:
            if self.key_status[key]['last_reset_day'] != current_day:
                self.key_status[key]['daily_count'] = 0
                self.key_status[key]['last_reset_day'] = current_day
                self.key_status[key]['disabled'] = False
                self.key_status[key]['consecutive_slow'] = 0

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
                
            if status['daily_count'] >= 450:
                status['disabled'] = True
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
                
            status['minute_timestamps'] = self._prune_old_requests(status['minute_timestamps'])
            if len(status['minute_timestamps']) >= 12:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
                
            return key
            
        raise Exception("No available API keys")

    def get_model(self):
        with self.lock:
            key = self._get_available_key()
            self.key_status[key]['minute_timestamps'].append(time.time())
            self.key_status[key]['daily_count'] += 1
            return self.models[key]

    def record_response_time(self, key: str, response_time: float):
        with self.lock:
            if response_time > self.response_time_threshold:
                self.key_status[key]['consecutive_slow'] += 1
                if self.key_status[key]['consecutive_slow'] >= self.max_slow_responses:
                    print(f"[WARNING] Switching from key {key[-4:]} due to slow responses")
                    self.key_status[key]['disabled'] = True
            else:
                self.key_status[key]['consecutive_slow'] = 0

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
GEMINI_KEYS = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")]
gemini_manager = GeminiAPIManager(GEMINI_KEYS)

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided authentication token"""
    if not TEAM_AUTH_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Server authentication not configured"
        )
    
    if credentials.credentials != TEAM_AUTH_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token"
        )

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Pre-load models at startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dummy_embeddings = np.zeros((1, 384), dtype='float32')
faiss_index = faiss.IndexFlatIP(384)
faiss_index.add(dummy_embeddings)

class FastDocumentProcessor:
    @staticmethod
    def extract_text(path: str, content_type: str) -> Tuple[List[str], str]:
        try:
            if "pdf" in content_type:
                return FastDocumentProcessor._extract_pdf(path)
            elif "word" in content_type or "docx" in content_type:
                return FastDocumentProcessor._extract_docx(path)
            elif "eml" in content_type or "message" in content_type:
                return FastDocumentProcessor._extract_email(path)
            return [], ""
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return [], ""

    @staticmethod
    def _extract_pdf(path: str) -> Tuple[List[str], str]:
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 15:
                full_text += " " + re.sub(r'\s+', ' ', text).strip()
        return FastDocumentProcessor._chunk_text(full_text), full_text

    @staticmethod
    def _extract_docx(path: str) -> Tuple[List[str], str]:
        doc = DocxDocument(path)
        full_text = " ".join(
            para.text.strip() for para in doc.paragraphs 
            if para.text.strip() and len(para.text.strip()) > 15
        )
        return FastDocumentProcessor._chunk_text(full_text), full_text

    @staticmethod
    def _extract_email(path: str) -> Tuple[List[str], str]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
            body = msg.get_payload()
            if isinstance(body, list):
                body = body[0].get_payload(decode=True)
                body = body.decode('utf-8', errors='ignore') if body else ""
            elif not isinstance(body, str):
                body = str(body)
        lines = [line.strip() for line in body.splitlines() 
                if line.strip() and len(line.strip()) > 10 and not line.startswith('>')]
        full_text = " ".join(lines)
        return FastDocumentProcessor._chunk_text(full_text), full_text

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
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
                    current_chunk = current_chunk[-overlap:] + " " + sentence if len(current_chunk) > overlap else sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return [c for c in chunks if len(c.strip()) > 30]

class FastRetriever:
    def __init__(self):
        self.model = embedding_model
        self.text_chunks = []
        self.semantic_index = faiss_index
        self.keyword_map = {}

    def build_indices(self, chunks: List[str]):
        self.text_chunks = chunks
        if chunks:
            embeddings = self.model.encode(chunks, convert_to_numpy=True, batch_size=64, normalize_embeddings=True)
            self.semantic_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.semantic_index.add(embeddings.astype("float32"))
            self._build_keyword_map()

    def _build_keyword_map(self):
        keywords = ['period', 'cover', 'benefit', 'claim', 'policy', 'hospital', 'waiting', 'premium']
        for i, chunk in enumerate(self.text_chunks):
            chunk_lower = chunk.lower()
            for keyword in keywords:
                if keyword in chunk_lower:
                    if keyword not in self.keyword_map:
                        self.keyword_map[keyword] = []
                    self.keyword_map[keyword].append(i)

    def retrieve(self, question: str, top_k: int = 2) -> List[str]:
        results = []
        
        # Semantic search
        if self.semantic_index and self.text_chunks:
            q_vec = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            similarities, indices = self.semantic_index.search(q_vec.astype("float32"), min(top_k, len(self.text_chunks)))
            results.extend(self.text_chunks[idx] for idx, sim in zip(indices[0], similarities[0]) if sim > 0.2)
        
        # Keyword boost
        question_lower = question.lower()
        for keyword, chunk_indices in self.keyword_map.items():
            if keyword in question_lower:
                results.extend(self.text_chunks[idx] for idx in chunk_indices[:1] if idx < len(self.text_chunks))
        
        return list(dict.fromkeys(results))[:top_k]

class OptimizedRAGPipeline:
    def __init__(self):
        self.retriever = FastRetriever()
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.semaphore = asyncio.Semaphore(4)

    async def process_document(self, url: str) -> bool:
        try:
            # Cache check
            doc_hash = hashlib.sha256(url.encode()).hexdigest()
            if doc_hash in self.cache:
                self.retriever.build_indices(self.cache[doc_hash])
                return True

            # Fetch document
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False
                    content = await resp.read()
                    content_type = resp.headers.get("Content-Type", "").lower()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Process in thread
            loop = asyncio.get_event_loop()
            chunks, _ = await loop.run_in_executor(
                self.executor,
                lambda: FastDocumentProcessor.extract_text(tmp_path, content_type)
            )
            os.unlink(tmp_path)

            if not chunks:
                return False

            # Cache and build index
            self.cache[doc_hash] = chunks
            await loop.run_in_executor(self.executor, self.retriever.build_indices, chunks)
            return True

        except Exception as e:
            print(f"[ERROR] Document processing failed: {e}")
            return False

    async def generate_answer(self, question: str) -> str:
        try:
            # Fast path for simple questions
            if len(question) < 15 or question.lower().startswith(('hi', 'hello')):
                return "Please ask a specific question about the insurance policy document."

            # Retrieve context with timeout
            try:
                relevant_chunks = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.retriever.retrieve(question)
                    ),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                relevant_chunks = []

            if not relevant_chunks:
                return self._get_fallback_answer(question)

            # Build grounded prompt
            context_str = "\n---\n".join(relevant_chunks)
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

CONTEXT:
{context_str}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using the provided context
2. Be specific about numbers, time periods, and conditions
3. If unsure, say "The policy states..." instead of guessing
4. Keep response under 150 words
5. Reference exact context details when possible

ANSWER:"""
            
            # Generate with strict timeout
            model = gemini_manager.get_model()
            start_time = time.time()
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: model.generate_content(prompt)
                    ),
                    timeout=6.0
                )
                response_time = time.time() - start_time
                gemini_manager.record_response_time(model._client.api_key, response_time)
                
                if response and response.text:
                    return response.text.strip()[:400]
                return "No response generated"
            except asyncio.TimeoutError:
                return "Response timeout - please try again"

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return "Error processing question"

    def _get_fallback_answer(self, question: str) -> str:
        """Provide reasonable fallback when no context is found"""
        q_lower = question.lower()
        if any(t in q_lower for t in ['period', 'day', 'time']):
            return "The policy typically specifies time periods for claims and coverage, but exact details weren't found in this document."
        elif any(t in q_lower for t in ['cover', 'benefit', 'include']):
            return "The document mentions various coverage benefits, but specific details for this question weren't found."
        return "The policy document doesn't contain specific information about this question."

pipeline = OptimizedRAGPipeline()

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_query(req: HackRxRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    verify_token(credentials)
    
    # Validate input
    if not req.questions or len(req.questions) > 5:
        raise HTTPException(status_code=400, detail="Provide 1-5 questions")
    
    # Process document and questions in parallel
    doc_task = asyncio.create_task(pipeline.process_document(req.documents))
    answer_tasks = [pipeline.generate_answer(q) for q in req.questions]
    
    # Wait for document processing with timeout
    try:
        await asyncio.wait_for(doc_task, timeout=3.0)
    except asyncio.TimeoutError:
        pass  # Continue with whatever context we have
    
    # Get answers with overall timeout
    try:
        answers = await asyncio.wait_for(asyncio.gather(*answer_tasks), timeout=8.0)
        return HackRxResponse(answers=answers)
    except asyncio.TimeoutError:
        # Return whatever answers we have
        done, _ = await asyncio.wait(answer_tasks, timeout=0)
        answers = [t.result() if t.done() else "Processing timeout" for t in answer_tasks]
        return HackRxResponse(answers=answers)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_size": len(pipeline.cache),
        "gemini_keys_available": sum(1 for k in GEMINI_KEYS if k)
    }

@app.on_event("startup")
async def startup():
    # Warm up models
    embedding_model.encode(["warmup"])
    try:
        model = gemini_manager.get_model()
        model.generate_content("warmup")
    except:
        pass
    print("Service ready - models warmed up")
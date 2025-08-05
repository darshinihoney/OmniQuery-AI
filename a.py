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

generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    max_output_tokens=100,
    candidate_count=1
)

class GeminiAPIManager:
    def __init__(self, keys: List[str]):
        self.keys = [key for key in keys if key]  # Filter out empty keys
        self.key_status = {
            key: {
                'minute_timestamps': [],
                'daily_count': 0,
                'last_reset_day': time.localtime().tm_yday,
                'disabled': False
            } for key in self.keys
        }
        self.lock = threading.Lock()
        self.models = {}
        self.current_key_index = 0
        self.response_time_threshold = 14  # Switch if response > 14s
        self.max_requests_before_switch = 5  # Switch after 5 slow requests
        
        # Initialize models
        for key in self.keys:
            try:
                genai.configure(api_key=key)
                self.models[key] = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
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
                print(f"[INFO] Reset daily count for key {key[-4:]}...")

    def _prune_old_requests(self, timestamps: List[float]):
        one_minute_ago = time.time() - 60
        return [t for t in timestamps if t > one_minute_ago]

    def _get_available_key(self):
        self._reset_daily_counts_if_needed()
        
        for _ in range(len(self.keys)):
            key = self.keys[self.current_key_index]
            status = self.key_status[key]
            
            # Skip disabled keys
            if status['disabled']:
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
                
            # Check daily limit (500)
            if status['daily_count'] >= 500:
                print(f"[WARNING] Key {key[-4:]}... reached daily limit (500)")
                status['disabled'] = True
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
                
            # Check minute limit (15)
            status['minute_timestamps'] = self._prune_old_requests(status['minute_timestamps'])
            if len(status['minute_timestamps']) >= 15:
                print(f"[WARNING] Key {key[-4:]}... reached minute limit (15 RPM)")
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                continue
                
            # If we get here, the key is available
            return key
            
        raise Exception("No available API keys - all have reached limits or are disabled")

    def get_model(self):
        with self.lock:
            key = self._get_available_key()
            self.key_status[key]['minute_timestamps'].append(time.time())
            self.key_status[key]['daily_count'] += 1
            
            print(f"[INFO] Using key {key[-4:]}... "
                  f"(Today: {self.key_status[key]['daily_count']}/500, "
                  f"Minute: {len(self.key_status[key]['minute_timestamps'])}/15)")
            
            return self.models[key]

    def record_response_time(self, response_time: float):
        with self.lock:
            key = self.keys[self.current_key_index]
            if response_time > self.response_time_threshold:
                # Count slow responses
                slow_count = getattr(self, f'slow_count_{key}', 0) + 1
                setattr(self, f'slow_count_{key}', slow_count)
                
                if slow_count >= self.max_requests_before_switch:
                    print(f"[WARNING] Key {key[-4:]}... had {slow_count} slow responses (>14s), switching...")
                    self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                    setattr(self, f'slow_count_{key}', 0)  # Reset counter

   

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")
GEMINI_KEYS = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")]
gemini_manager = GeminiAPIManager(GEMINI_KEYS)     

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
        try:
            reader = PdfReader(path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 15:
                    text = re.sub(r'\s+', ' ', text).strip()
                    full_text += f" {text}"
            chunks = FastDocumentProcessor.smart_chunk(full_text, 400, 100)
            return chunks, full_text.strip()
        except Exception as e:
            print(f"[ERROR] PDF extraction failed: {e}")
            return [], ""

    @staticmethod
    def extract_text_from_docx(path: str) -> Tuple[List[str], str]:
        try:
            doc = DocxDocument(path)
            full_text = ""
            for para in doc.paragraphs:
                text = para.text.strip()
                if text and len(text) > 15:
                    text = re.sub(r'\s+', ' ', text).strip()
                    full_text += f" {text}"
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
        except Exception as e:
            print(f"[ERROR] Email extraction failed: {e}")
            return [], ""

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
        if chunks:
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
        results = []
        try:
            if self.semantic_index and self.text_chunks:
                q_vec = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
                similarities, indices = self.semantic_index.search(q_vec.astype("float32"), min(top_k, len(self.text_chunks)))
                semantic_results = [self.text_chunks[idx] for idx, sim in zip(indices[0], similarities[0]) if sim > 0.15]
                results.extend(semantic_results)
            
            keyword_results = []
            for keyword, chunk_indices in self.keyword_map.items():
                if keyword in question.lower():
                    for idx in chunk_indices[:2]:
                        if idx < len(self.text_chunks):
                            chunk = self.text_chunks[idx]
                            if chunk not in results:
                                keyword_results.append(chunk)
            results.extend(keyword_results)
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
        
        return results[:top_k]

class OptimizedRAGPipeline:
    def __init__(self):
        self.retriever = FastRetriever()
        self.contexts = []
        self.full_document = ""
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.cache = {}

    async def fetch_document(self, url: str) -> Tuple[bytes, str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status} error fetching document")
                    content = await resp.read()
                    content_type = resp.headers.get("Content-Type", "").lower()
                    return content, content_type
        except Exception as e:
            print(f"[ERROR] Document fetch failed: {e}")
            raise

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
            chunks, full_text = [], ""
            
            if "pdf" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_pdf, path)
            elif "word" in content_type or "docx" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_docx, path)
            elif "eml" in content_type or "message" in content_type:
                chunks, full_text = await loop.run_in_executor(self.executor, FastDocumentProcessor.extract_text_from_email, path)
            else:
                print(f"[WARNING] Unsupported content type: {content_type}")
                os.unlink(path)
                return False

            if not chunks or not full_text:
                print("[WARNING] No valid text extracted from document")
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

    async def answer_single_question(self, question: str) -> str:
        try:
            print(f"[DEBUG] Processing question: {question}")
            
            # Retrieve relevant context
            relevant_chunks = self.retriever.fast_search(question, top_k=3)
            if not relevant_chunks:
                print(f"[WARNING] No relevant chunks found for question: {question}")
                return "I could not find relevant information in the document to answer this question."

            # Build the prompt
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

            Provide a concise answer based on the policy document:"""
            
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            print(f"[DEBUG] First 200 chars of prompt: {prompt[:200]}...")

            # Get Gemini model instance
            model = gemini_manager.get_model()
            
            # Generate response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: model.generate_content(prompt)
            )

            if not response or not response.text:
                print("[ERROR] Empty response from Gemini")
                return "No valid response received from the AI model."

            answer = response.text.strip()
            print(f"[DEBUG] Received answer: {answer[:200]}...")
            return answer

        except Exception as e:
            print(f"[ERROR] Failed to answer question: {str(e)}")
            return "I encountered an error while processing this question."

    async def answer_questions(self, questions: List[str]) -> List[str]:
        semaphore = asyncio.Semaphore(4)  # Limit concurrent requests
        
        async def bounded_answer(q):
            async with semaphore:
                return await self.answer_single_question(q)

        try:
            tasks = [bounded_answer(q) for q in questions]
            answers = await asyncio.gather(*tasks)
            return answers
        except Exception as e:
            print(f"[ERROR] Answer questions failed: {e}")
            return ["I encountered an error while processing this question."] * len(questions)

engine = OptimizedRAGPipeline()

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_query(req: HackRxRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    start_time = time.time()
    verify_token(credentials)
    
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
    
    print("\n[DEBUG] Model predictions:")
    for i, (question, answer) in enumerate(zip(req.questions, answers), 1):
        print(f"  Q{i}: {question}")
        print(f"  A{i}: {answer[:200]}...")  # Print first 200 chars of answer
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
        "chunks_loaded": len(engine.contexts) if engine.contexts else 0,
        "gemini_keys_configured": len([k for k in GEMINI_KEYS if k]) > 0
    }

# Test Gemini API keys on startup
async def test_gemini_keys():
    print("\n[STARTUP] Testing Gemini API keys...")
    try:
        test_model = gemini_manager.get_model()
        test_response = test_model.generate_content("Hello, please respond with 'OK' if working")
        print(f"[STARTUP] Gemini test response: {test_response.text}")
    except Exception as e:
        print(f"[STARTUP ERROR] Gemini API test failed: {e}")

# Run the test on startup
asyncio.create_task(test_gemini_keys())
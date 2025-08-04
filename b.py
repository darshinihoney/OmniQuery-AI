import os
import json
import tempfile
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Document processing
import PyPDF2
import docx
from email import message_from_string
import io

# ML and embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# LLM integration
import openai
from transformers import pipeline
from pdfminer.high_level import extract_text as pdfminer_extract_text
# Flask for API
from flask import Flask, request, jsonify
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    content: str
    source: str
    chunk_id: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    clause: Optional[str] = None

class DocumentProcessor:
    """Handles document parsing and preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.eml']
    
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise
    
    def extract_text_from_pdf(self, content: bytes) -> List[DocumentChunk]:
        """Extract text from PDF using pdfminer with page info"""
        chunks = []
        try:
           with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
              tmp.write(content)
              tmp_path = tmp.name
        
              # Use pdfminer to extract full text
              full_text = pdfminer_extract_text(tmp_path)
        
              # Split into chunks by page (optional: add actual per-page logic)
              page_chunks = self._split_into_chunks(full_text, page_num=None)
              chunks.extend(page_chunks)
        
              os.remove(tmp_path)
    
        except Exception as e:
           logger.error(f"Error extracting PDF text: {e}")
           raise
    
        return chunks

    
    def extract_text_from_docx(self, content: bytes) -> List[DocumentChunk]:
        """Extract text from DOCX with structure information"""
        chunks = []
        try:
            doc = docx.Document(io.BytesIO(content))
            current_section = None
            
            for para_num, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    # Detect sections/headings
                    if paragraph.style.name.startswith('Heading'):
                        current_section = paragraph.text.strip()
                    
                    chunk = DocumentChunk(
                        content=paragraph.text.strip(),
                        source="docx",
                        chunk_id=para_num,
                        section=current_section
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
        
        return chunks
    
    def extract_text_from_email(self, content: bytes) -> List[DocumentChunk]:
        """Extract text from email format"""
        chunks = []
        try:
            try:
               decoded = content.decode('utf-8', errors='ignore')
            except Exception as e:
               logger.warning(f"Decoding fallback triggered: {e}")
               decoded = content.decode(errors='ignore')
        
            msg = message_from_string(decoded)
            subject = msg.get('Subject', '')
            body = ""

            if msg.is_multipart():
               for part in msg.walk():
                   if part.get_content_type() == "text/plain":
                      try:
                         body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                      except Exception as e:
                         logger.warning(f"Skipping a non-decodable email part: {e}")
            else:
               try:
                 body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
               except Exception as e:
                  logger.warning(f"Error decoding email body: {e}")

            if subject:
              chunks.append(DocumentChunk(
                content=f"Subject: {subject}",
                source="email",
                chunk_id=0,
                section="header"
            ))

            if body:
              body_chunks = self._split_into_chunks(body, None, "email_body")
              chunks.extend(body_chunks)

        except Exception as e:
          logger.error(f"Error extracting email text: {e}")
          raise

        return chunks

    
    def _split_into_chunks(self, text: str, page_num: Optional[int], section: str = None) -> List[DocumentChunk]:
        """Split text into semantic chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) > 50:  # Filter out very short paragraphs
                # Further split if paragraph is too long (>1000 chars)
                if len(para) > 1000:
                    sentences = para.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 800:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(DocumentChunk(
                                    content=current_chunk.strip(),
                                    source="document",
                                    chunk_id=len(chunks),
                                    page_number=page_num,
                                    section=section
                                ))
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            source="document",
                            chunk_id=len(chunks),
                            page_number=page_num,
                            section=section
                        ))
                else:
                    chunks.append(DocumentChunk(
                        content=para,
                        source="document",
                        chunk_id=len(chunks),
                        page_number=page_num,
                        section=section
                    ))
        
        return chunks

class SemanticSearchEngine:
    """Handles embeddings and semantic search"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks"""
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = chunks
        self.embeddings = embeddings
        
        logger.info("Embeddings created successfully")
        return embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search for relevant chunks using semantic similarity"""
        if self.index is None:
            raise ValueError("Index not created. Call create_embeddings first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

class QuestionAnalyzer:
    """Analyzes questions to determine type and expectations"""
    
    def __init__(self):
        self.question_patterns = {
            'when': ['when', 'what time', 'how long', 'duration', 'period'],
            'what': ['what is', 'what are', 'define', 'definition'],
            'how': ['how does', 'how can', 'how to', 'process', 'procedure'],
            'who': ['who is', 'who can', 'responsible'],
            'where': ['where', 'location', 'place'],
            'why': ['why', 'reason', 'because'],
            'coverage': ['cover', 'coverage', 'benefit', 'eligible'],
            'condition': ['condition', 'requirement', 'criteria', 'if'],
            'amount': ['amount', 'cost', 'price', 'fee', 'premium', 'limit'],
            'exclusion': ['exclude', 'not cover', 'limitation', 'restriction']
        }
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine type and expectations"""
        question_lower = question.lower()
        
        analysis = {
            'type': 'general',
            'expects_date': False,
            'expects_amount': False,
            'expects_condition': False,
            'expects_definition': False,
            'complexity': 'medium'
        }
        
        # Determine question type
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                analysis['type'] = q_type
                break
        
        # Set expectations based on patterns
        if any(word in question_lower for word in ['when', 'period', 'duration', 'time']):
            analysis['expects_date'] = True
        
        if any(word in question_lower for word in ['amount', 'cost', 'premium', 'limit', 'benefit']):
            analysis['expects_amount'] = True
        
        if any(word in question_lower for word in ['condition', 'if', 'requirement', 'criteria']):
            analysis['expects_condition'] = True
        
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            analysis['expects_definition'] = True
        
        # Determine complexity
        if len(question.split()) > 15 or '?' in question[:-1]:
            analysis['complexity'] = 'high'
        elif len(question.split()) < 8:
            analysis['complexity'] = 'low'
        
        return analysis

class ContextualAnswerGenerator:
    """Generates contextual answers using LLM reasoning"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
    
    def generate_answer(self, question: str, contexts: List[tuple], question_analysis: Dict) -> str:
        """Generate contextual answer from retrieved contexts"""
        
        if not contexts:
            return "Information not found in the document."
        
        # Prepare context text
        context_text = ""
        for i, (chunk, score) in enumerate(contexts[:5]):  # Top 5 contexts
            context_text += f"Context {i+1} (Score: {score:.3f}):\n{chunk.content}\n\n"
        
        # Create reasoning prompt
        prompt = self._create_reasoning_prompt(question, context_text, question_analysis)
        
        try:
            # Use OpenAI API if available, otherwise use local reasoning
            if self.api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert document analyst. Provide accurate, complete, and well-reasoned answers based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback to rule-based reasoning
                return self._rule_based_reasoning(question, contexts, question_analysis)
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._rule_based_reasoning(question, contexts, question_analysis)
    
    def _create_reasoning_prompt(self, question: str, context: str, analysis: Dict) -> str:
        """Create a reasoning prompt for the LLM"""
        
        prompt = f"""
Question: {question}

Context from document:
{context}

Instructions:
1. Read the question carefully and understand what it's asking for.
2. Analyze the provided context to find relevant information.
3. Reason through the conditions, exceptions, and specific details.
4. Provide a complete, accurate answer that explains the logic.
5. If the question expects specific information (dates, amounts, conditions), make sure to include them.
6. If information is incomplete or missing, state what is known and what is missing.
7. Do not hallucinate information not present in the context.

Expected answer type: {analysis['type']}
Should include: {', '.join([k for k, v in analysis.items() if v is True])}

Answer:
"""
        return prompt
    
    def _rule_based_reasoning(self, question: str, contexts: List[tuple], analysis: Dict) -> str:
        """Fallback rule-based reasoning when LLM is not available"""
        
        if not contexts:
            return "Information not found in the document."
        
        # Combine relevant contexts
        relevant_content = []
        for chunk, score in contexts[:3]:  # Top 3 most relevant
            if score > 0.3:  # Threshold for relevance
                relevant_content.append(chunk.content)
        
        if not relevant_content:
            return "Information not found in the document."
        
        # Simple reasoning based on question type
        combined_text = " ".join(relevant_content)
        
        # Extract specific information based on analysis
        answer_parts = []
        
        if analysis['expects_date']:
            dates = self._extract_dates_periods(combined_text)
            if dates:
                answer_parts.extend(dates)
        
        if analysis['expects_amount']:
            amounts = self._extract_amounts(combined_text)
            if amounts:
                answer_parts.extend(amounts)
        
        if analysis['expects_condition']:
            conditions = self._extract_conditions(combined_text)
            if conditions:
                answer_parts.extend(conditions)
        
        # If no specific extractions, return the most relevant context
        if not answer_parts:
            return relevant_content[0]
        
        return " ".join(answer_parts)
    
    def _extract_dates_periods(self, text: str) -> List[str]:
        """Extract date and period information"""
        import re
        
        patterns = [
            r'\d+\s*(?:days?|months?|years?)',
            r'(?:within|after|before)\s+\d+\s*(?:days?|months?|years?)',
            r'(?:grace period|waiting period|period).*?\d+.*?(?:days?|months?|years?)',
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract amount and percentage information"""
        import re
        
        patterns = [
            r'\d+%\s*(?:of|discount|benefit)',
            r'(?:Rs\.?|INR|₹)\s*\d+(?:,\d+)*',
            r'\d+(?:,\d+)*\s*(?:rupees?|INR)',
            r'(?:limit|cap|maximum).*?\d+',
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        return amounts
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract condition information"""
        import re
        
        # Look for conditional statements
        sentences = text.split('.')
        conditions = []
        
        condition_words = ['if', 'provided', 'unless', 'subject to', 'condition', 'requirement']
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in condition_words):
                conditions.append(sentence.strip())
        
        return conditions

class IntelligentQuerySystem:
    """Main system orchestrating the entire pipeline"""
    
    def __init__(self, openai_api_key: str = None):
        self.document_processor = DocumentProcessor()
        self.search_engine = SemanticSearchEngine()
        self.question_analyzer = QuestionAnalyzer()
        self.answer_generator = ContextualAnswerGenerator(openai_api_key)
        self.documents_cache = {}
    
    async def process_request(self, documents_url: str, questions: List[str]) -> Dict[str, List[str]]:
        """Process the main request and return structured response"""
        
        start_time = time.time()
        
        try:
            # Step 1: Download and process documents
            logger.info("Processing documents...")
            if documents_url not in self.documents_cache:
                content = self.document_processor.download_document(documents_url)
                
                # Determine document type and extract text
                if documents_url.lower().endswith('.pdf'):
                    chunks = self.document_processor.extract_text_from_pdf(content)
                elif documents_url.lower().endswith('.docx'):
                    chunks = self.document_processor.extract_text_from_docx(content)
                else:
                    chunks = self.document_processor.extract_text_from_email(content)
                
                # Create embeddings
                self.search_engine.create_embeddings(chunks)
                self.documents_cache[documents_url] = True
            
            # Step 2: Process questions
            answers = []
            
            for question in questions:
                logger.info(f"Processing question: {question}")
                
                # Analyze question
                question_analysis = self.question_analyzer.analyze_question(question)
                
                # Search for relevant contexts
                contexts = self.search_engine.search(question, top_k=10)
                
                # Generate answer
                answer = self.answer_generator.generate_answer(
                    question, contexts, question_analysis
                )
                
                answers.append(answer)
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return {"answers": answers}
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}

# Flask API
app = Flask(__name__)

# Initialize the system
query_system = IntelligentQuerySystem()

@app.route('/hackrx/run', methods=['POST'])
def process_query():
    """API endpoint for processing queries"""
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'documents' not in data or 'questions' not in data:
            return jsonify({"error": "Missing required fields: documents, questions"}), 400
        
        documents_url = data['documents']
        questions = data['questions']
        
        if not isinstance(questions, list) or not questions:
            return jsonify({"error": "Questions must be a non-empty list"}), 400
        
        # Process the request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            query_system.process_request(documents_url, questions)
        )
        
        loop.close()
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    # Set up environment
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Intelligent Query System on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
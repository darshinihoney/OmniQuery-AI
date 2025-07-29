# 🚀 OmniQuery AI

**Intelligent Document Q&A System powered by LLM and Semantic Search**

OmniQuery AI transforms any PDF or DOCX document into an intelligent question-answering system. Simply upload your documents and ask questions in natural language to get precise, contextual answers extracted directly from your content.

## ✨ Features

- **Multi-format Support**: Process PDF and DOCX documents seamlessly
- **Semantic Search**: Advanced embedding-based retrieval using FAISS
- **LLM Integration**: Powered by Google Gemini for intelligent answer generation
- **RESTful API**: Clean, production-ready FastAPI interface
- **Real-time Processing**: Fast document parsing and query responses
- **Secure Authentication**: Bearer token-based API security

## 🔧 Tech Stack

- **Backend**: FastAPI (Python)
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyMuPDF, docx2txt

## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/omniquery-ai.git
cd omniquery-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file:
```bash
# Create .env file in project root
GEMINI_API_KEY=your_gemini_api_key
TEAM_AUTH_TOKEN=your_secure_token
```

4. Run the application:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

### 🔒 Authentication

All API requests require a Bearer token in the Authorization header:

```bash
Authorization: Bearer your_secure_token
```

### Endpoint

**POST** `/api/v1/hackrx/run`

### Request Format

```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for payment?",
        "Are there any waiting periods?",
        "What expenses are covered?"
    ]
}
```

### Response Format

```json
{
    "answers": [
        "A grace period of thirty days is provided for payment after the due date...",
        "There is a waiting period of thirty-six months for pre-existing diseases...",
        "The policy covers medical expenses including hospitalization, surgery..."
    ]
}
```

## How It Works

1. **Document Processing**: Downloads and extracts text from PDF/DOCX files
2. **Text Chunking**: Splits documents into manageable chunks with overlap
3. **Embedding Generation**: Creates semantic embeddings using SentenceTransformers
4. **Vector Indexing**: Stores embeddings in FAISS for fast similarity search
5. **Query Processing**: Finds relevant chunks based on semantic similarity
6. **Answer Generation**: Uses Gemini LLM to generate contextual answers


## 🏗️ Architecture

```plaintext
         ┌──────────────────────┐
         │   Input Documents    │  ← (PDF/DOCX via URL)
         └────────┬─────────────┘
                  ↓
         ┌──────────────────────┐
         │     Text Extractor   │  ← (fitz/docx2txt)
         └────────┬─────────────┘
                  ↓
         ┌──────────────────────┐
         │   Chunk + Embedder   │  ← (SentenceTransformer + FAISS)
         └────────┬─────────────┘
                  ↓
         ┌──────────────────────┐
         │   Query Embed + TopK │  ← (Semantic Retrieval)
         └────────┬─────────────┘
                  ↓
         ┌──────────────────────┐
         │ Gemini-based Answer  │  ← (Custom Prompt Logic)
         └────────┬─────────────┘
                  ↓
         ┌──────────────────────┐
         │   JSON Output API    │
         └──────────────────────┘

```

## Use Cases

- **Insurance**: Policy analysis, coverage verification, claims processing
- **Legal**: Contract review, clause extraction, compliance checking  
- **HR**: Employee handbook queries, policy clarification
- **Healthcare**: Medical document analysis, procedure coverage
- **Finance**: Regulatory document review, policy interpretation

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `TEAM_AUTH_TOKEN` | API authentication token | Yes |

### Model Parameters

- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk Size**: 300 words with 50-word overlap
- **Retrieval**: Top 10 most relevant chunks
- **Vector Index**: FAISS L2 distance

## Dependencies

```txt
fastapi
uvicorn
PyMuPDF
docx2txt
requests
faiss-cpu
numpy
google-generativeai
sentence-transformers
python-dotenv
pydantic
```

## Error Handling

The API includes comprehensive error handling for:

- Invalid document URLs or formats
- Authentication failures
- Missing environment variables
- Document processing errors
- LLM generation failures

## Performance

- **Response Time**: Typically under 30 seconds for standard documents
- **Supported File Sizes**: Up to 50MB documents
- **Concurrent Requests**: Optimized for multiple simultaneous queries
- **Memory Usage**: Efficient chunk-based processing

## Development

### Project Structure

```
omniquery-ai/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
├── README.md           # This file
└── Dockerfile          # Container deployment
```


**Transform your documents into intelligent APIs with OmniQuery AI**

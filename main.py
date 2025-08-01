from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from PyPDF2 import PdfReader
import docx
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import groq
import os
from dotenv import load_dotenv
import aiohttp
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="LLM-Powered Query-Retrieval System", openapi_url="/api/v1/openapi.json")
security = HTTPBearer()

# Groq client setup
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sentence transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dimension of embeddings
dimension = 384

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def authenticate(credentials: HTTPAuthorizationCredentials = Security(security)):
    expected_token = "225bc5a220c1e087a557d9738720c84d5a1dca03fd966f83977f563fe741968d"
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

async def download_file(url: str) -> tuple[str, str]:
    """Download a file from a URL to a temporary location and return the file path and content type."""
    logger.debug(f"Downloading file from URL: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to download document: HTTP {response.status}")
                raise HTTPException(status_code=400, detail="Failed to download document")
            content_type = response.headers.get('Content-Type', '')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                tmp_file.write(await response.read())
                logger.debug(f"Saved to temporary file: {tmp_file.name}")
                return tmp_file.name, content_type

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                text += extracted or ""
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            doc = docx.Document(f)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            if not text.strip():
                raise ValueError("No text extracted from DOCX")
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    if not chunks:
        raise ValueError("No valid chunks created from text")
    return chunks

async def process_document(url: str) -> List[str]:
    logger.debug(f"Processing document URL: {url}")
    temp_file_path, content_type = await download_file(url)
    try:
        if 'application/pdf' in content_type:
            text = extract_text_from_pdf(temp_file_path)
        elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            text = extract_text_from_docx(temp_file_path)
        else:
            # Assume plain text or email
            try:
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                text = re.sub(r"From:.*\n|To:.*\n|Subject:.*\n|Date:.*\n", "", text)
                if not text.strip():
                    raise ValueError("No text extracted from plain text/email")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Text/email extraction failed: {str(e)}")
        
        logger.debug("Text extracted successfully")
        chunks = chunk_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.debug(f"Deleted temporary file: {temp_file_path}")

def generate_answer(query: str, context: List[str]) -> str:
    context_text = "\n".join(context) if context else "No relevant context found."
    prompt = f"""Context: {context_text}

Question: {query}

Provide a precise answer based on the context, including an explanation of how the answer was derived."""
    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Unable to generate answer due to an error: {str(e)}"

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM-Powered Query-Retrieval System. Use POST /hackrx/run to submit queries."}

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, token: str = Depends(authenticate)):
    try:
        logger.debug(f"Received request: {request}")
        # Process document
        chunks = await process_document(request.documents)
        
        # Create a temporary FAISS index for this document
        temp_index = faiss.IndexFlatL2(dimension)
        embeddings = model.encode(chunks)
        temp_index.add(np.array(embeddings, dtype=np.float32))
        
        # Process each question
        answers = []
        for question in request.questions:
            logger.debug(f"Processing question: {question}")
            query_embedding = model.encode([question])[0]
            distances, indices = temp_index.search(np.array([query_embedding], dtype=np.float32), k=5)
            relevant_chunks = [chunks[i] for i in indices[0] if i >= 0 and i < len(chunks)]
            
            # Generate answer with explanation
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
        
        return QueryResponse(answers=answers)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
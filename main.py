from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
import logging
import os
from contextlib import asynccontextmanager

from pdf_parser import extract_text_by_page_from_url, chunk_text
from pinecone_handler import upsert_chunks_to_pinecone, query_index
from llm_reasoner import get_reasoned_answer, batch_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: str

VALID_API_KEY = os.getenv("API_KEY")  

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from the Authorization header"""
    if credentials.credentials != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Lifespan events for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Document Q&A API")
    yield
    # Shutdown
    logger.info("Shutting down Document Q&A API")

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for extracting and answering questions from PDF documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"message": "Document Q&A API is running", "status": "healthy"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Document Q&A API",
        "version": "1.0.0"
    }

@app.post(
    "/hackrx/run",
    response_model=DocumentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Document Processing"]
)
async def process_document_questions(
    request: DocumentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process a PDF document and answer questions about its content using RAG.
    
    - **documents**: URL to the PDF document
    - **questions**: List of questions to answer about the document
    
    Returns a list of answers corresponding to each question.
    """
    try:
        logger.info(f"Processing document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Validate input
        if not request.questions:
            raise HTTPException(
                status_code=400,
                detail="At least one question must be provided"
            )
        
        if len(request.questions) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=400,
                detail="Too many questions. Maximum 50 questions allowed per request."
            )
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        pages = extract_text_by_page_from_url(str(request.documents))
        
        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the provided document URL"
            )
        
        # Chunk the text
        logger.info("Chunking text...")
        chunks = chunk_text(pages)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create text chunks from the document"
            )
        
        # Upsert to Pinecone
        logger.info("Upserting chunks to Pinecone...")
        upsert_chunks_to_pinecone(text_chunks=chunks)
        
        # Get shared context for the first question
        logger.info("Getting shared context...")
        shared_context = query_index(query=request.questions[0], top_k=3)
        
        # Batch queries for optimization
        logger.info("Batching queries...")
        question_batches = batch_queries(request.questions, shared_context)
        
        # Process questions and get answers
        logger.info("Processing questions and getting answers...")
        answers = []
        
        for batch_idx, batch in enumerate(question_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(question_batches)}")
            
            for question in batch:
                try:
                    # Get relevant clauses for each question
                    rel_clauses = query_index(query=question, top_k=3)
                    
                    # Get reasoned answer
                    answer = get_reasoned_answer(question, rel_clauses)
                    answers.append(answer)
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {str(e)}")
                    # Add a fallback answer for failed questions
                    answers.append(f"Sorry, I couldn't find an answer to this question due to processing error.")
        
        logger.info(f"Successfully processed {len(answers)} questions")
        
        return DocumentResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the document: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="An internal server error occurred"
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        log_level="info"
    )
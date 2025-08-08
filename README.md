Document Q&A API – Health Insurance Policy RAG System
This project is a FastAPI-based Retrieval-Augmented Generation (RAG) system that extracts text from PDF documents, stores it in Pinecone, and uses a Large Language Model (LLM) to answer user queries based on the stored content.

Features
•	PDF text extraction from a URL using pdfplumber.
•	Text chunking for efficient storage and retrieval.
•	Vector storage in Pinecone with metadata.
•	Query retrieval for relevant clauses.
•	LLM reasoning with Groq's LLaMA 3 model.
•	Batch query processing for performance optimization.
•	API key authentication for secure usage.
•	CORS support for cross-origin requests.

Project Structure
1. main.py – FastAPI Application
This is the main API entry point. It defines the API routes, handles authentication, and coordinates the entire document processing workflow.
Key Functions / Classes:
•	DocumentRequest – Pydantic model defining API request schema.
•	DocumentResponse – Pydantic model defining API response schema.
•	verify_api_key – Validates the API key from the request header.
•	process_document_questions (POST /hackrx/run) – Extracts text from PDF, chunks text, stores chunks in Pinecone, retrieves relevant chunks for each query, calls the LLM for reasoning, and returns answers.
•	Health Endpoints: GET / (basic health check) and GET /health (detailed health check).

3. pdf_parser.py – PDF Extraction & Chunking
Handles fetching and splitting text from PDF files.
•	extract_text_by_page_from_url(pdf_url) – Downloads a PDF from a URL and extracts text page-by-page.
•	chunk_text(text_list, max_chunk_length=1500) – Splits extracted text into chunks for better indexing.

5. pinecone_handler.py – Pinecone Vector Database Integration
Manages storage and retrieval of document chunks in Pinecone.
•	upsert_chunks_to_pinecone(text_chunks, namespace='policy-pdf', category=None) – Generates a unique ID for each chunk and stores it in Pinecone with metadata.
•	query_index(query, top_k=3, namespace='policy-pdf', fields=['chunk_text']) – Searches for the most relevant chunks matching a query.

6. llm_reasoner.py – LLM Query Processing & Reasoning
Handles reasoning over retrieved clauses using Groq’s LLaMA 3 model.
•	get_reasoned_answer(query, chunks) – Combines retrieved chunks into context, builds a reasoning prompt, sends the request to Groq’s LLaMA 3 model, and returns a plain-text explanation.
•	batch_queries(queries, context) – Groups queries into batches so token limits are respected.

How It Works (Flow)
1.	User sends a POST request to /hackrx/run with documents (PDF URL) and questions.
2.	pdf_parser.py downloads the PDF, extracts text, and chunks it.
3.	pinecone_handler.py stores chunks in Pinecone and retrieves relevant chunks for each question.
4.	llm_reasoner.py sends the query + relevant chunks to the LLaMA 3 model and returns a concise explanation.
5.	main.py returns the list of answers in the response.
   
Installation
pip install -r requirements.txt

Running the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Authentication
All endpoints except / and /health require an API key.

Header:
Authorization: Bearer YOUR_API_KEY

Example Request
POST /hackrx/run
{
  "documents": "https://example.com/health_policy.pdf",
  "questions": [
    "Does this policy cover maternity expenses?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
Example Response
{
  "answers": [
    "Yes, the policy covers maternity expenses after a 2-year waiting period.",
    "The waiting period for pre-existing diseases is 48 months."
  ]
}
 

from input_module import extract_text_by_page_from_url
from input_module import chunk_text
from pinecone_handler import upsert_chunks_to_pinecone,query_index

if __name__ == "__main__":
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    pages = extract_text_by_page_from_url(pdf_url)
    chunks = chunk_text(pages)
    upsert_chunks_to_pinecone(text_chunks=chunks)
    while True:
        query = input("Query: ")
        rel_clauses = query_index(query=query)
        print(rel_clauses)
        print("\n")
    

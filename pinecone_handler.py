from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "test-index"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    
index = pc.Index(index_name)

def upsert_chunks_to_pinecone(text_chunks, namespace="policy-pdf", category=None):
    records = []
    for i, chunk in enumerate(text_chunks):
        record = {
            "_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "category": category or "unknown"
        }
        records.append(record)

    index.upsert_records(namespace=namespace, records=records)
    print(f"Upserted {len(records)} records to namespace '{namespace}' in index '{index_name}'")
    
def query_index(query, top_k=3, namespace="policy-pdf", fields=["chunk_text"]):
    results = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
        },
        fields=fields
    )
    
    return [
        hit["fields"]["chunk_text"]
        for hit in results["result"]["hits"]
        if "fields" in hit and "chunk_text" in hit["fields"]
    ]
    
def update_document(namespace, document_id, chunks):
    existing = index.query(
        namespace=namespace,
        vector=[0] * 384,  # Dummy vector just to run filter search
        filter={"document_id": {"$eq": document_id}},
        top_k=1,
        include_metadata=False
    )

    if not existing.get("matches"):  
        print(f"Document '{document_id}' does not exist — skipping update.")
        return False

    # Step 2 — Delete existing vectors for the document
    index.delete(
        namespace=namespace,
        filter={"document_id": {"$eq": document_id}}
    )
    print(f"Deleted old chunks for document: {document_id}")

    # Step 3 — Encode and upsert updated chunks
    vectors = []
    for chunk in chunks:
        vector = model.encode(chunk["chunk_text"]).tolist()
        vectors.append({
            "id": f"{document_id}#chunk{chunk['chunk_number']}",
            "values": vector,
            "metadata": {
                "document_id": document_id,
                "chunk_number": chunk["chunk_number"],
                "chunk_text": chunk["chunk_text"],
                "version": "updated"
            }
        })

    index.upsert(namespace=namespace, vectors=vectors)
    print(f"Updated document '{document_id}' with {len(chunks)} chunks.")

    return True

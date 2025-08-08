from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") 

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

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

# def upsert_chunks_to_pinecone(text_chunks, namespace="policy-pdf", category=None):
#     records = []
#     for i, chunk in enumerate(text_chunks):
#         record = {
#             "_id": str(uuid.uuid4()),
#             "chunk_text": chunk,
#             "category": category or "unknown"
#         }
#         records.append(record)

#     index.upsert_records(namespace=namespace, records=records)
#     print(f"Upserted {len(records)} records to namespace '{namespace}' in index '{index_name}'")
def batch_list(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def upsert_chunks_to_pinecone(text_chunks, namespace="policy-pdf", category=None):
    records = []
    for i, chunk in enumerate(text_chunks):
        record = {
            "_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "category": category or "unknown"
        }
        records.append(record)

    batch_size = 95  # Pinecone Llama model limit
    for batch in batch_list(records, batch_size):
        index.upsert_records(namespace=namespace, records=batch)
        print(f"Upserted {len(batch)} records to namespace '{namespace}' in index '{index_name}'")
    
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
    
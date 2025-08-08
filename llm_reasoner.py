import os
import requests
from dotenv import load_dotenv
import json
import time
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MAX_CONTEXT_LENGTH = 3500

TOP_K = 10

BATCH_QUERY_OVERHEAD = 120  

BATCH_TOKEN_LIMIT = 4000

def get_reasoned_answer(query, chunks):
    context = "\n".join(chunks)
    prompt = f"""
You are a policy analysis assistant that helps determine whether specific insurance queries are covered under a given health insurance policy.

Given the user's question and relevant policy clauses, your job is to:
1. Focus only on the content provided in the policy clauses — do not make assumptions.
2. Identify the single *most relevant* clause that directly answers the query.
3. Provide a clear and concise explanation based on that clause only.

Important: Respond with ONLY the explanation text. Do not use JSON format or any special formatting.

User Question:
{query}

Relevant Policy Clauses:
{context}

Explanation:"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system", 
                "content": "You are a reasoning assistant that helps evaluate insurance claims based on provided policy clauses. Respond with clear, concise explanations in plain text format only."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }
    
    try:
        time.sleep(3)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        # Get the response content and clean it
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # Remove any potential JSON formatting if it still appears
        if content.startswith('{') and content.endswith('}'):
            try:
                import json
                parsed = json.loads(content)
                if 'explanation' in parsed:
                    content = parsed['explanation']
            except:
                pass  # If JSON parsing fails, use the original content
        
        return content
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling Groq API: {response.status_code} - {response.text}")
        return "Unable to process query due to API error."
        
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        print(f"Full response: {response.text}")
        return "Unable to process query due to unexpected response format."
        
    except Exception as e:
        print(f"Unexpected error during LLM call: {e}")
        return "Unable to process query due to technical error."
    
    
def batch_queries(queries, context):

       

    """Group queries so total tokens (context + queries) stay under BATCH_TOKEN_LIMIT."""

    batches = []

    current_batch = []

    context_tokens = len(" ".join(context).split())

    for q in queries:

        est_tokens = context_tokens + (len(current_batch) + 1) * BATCH_QUERY_OVERHEAD

        if est_tokens > BATCH_TOKEN_LIMIT and current_batch:

            batches.append(current_batch)

            current_batch = [q]

        else:

            current_batch.append(q)

    if current_batch:

        batches.append(current_batch)

    return batches
import requests
import pdfplumber
import io

def extract_text_by_page_from_url(pdf_url):
    response = requests.get(pdf_url)
    file_like = io.BytesIO(response.content)

    with pdfplumber.open(file_like) as pdf:
        return [page.extract_text() for page in pdf.pages if page.extract_text()]
    
    
def chunk_text(text_list, max_chunk_length=90):
    chunks = []

    for text in text_list:
        words = text.split()
        start = 0
        current_length = 0

        for i, word in enumerate(words):
            word_length = len(word) + 1

            if current_length + word_length > max_chunk_length:
                chunks.append(" ".join(words[start:i]))
                start = i
                current_length = word_length
            else:
                current_length += word_length

        if start < len(words):
            chunks.append(" ".join(words[start:]))

    return chunks
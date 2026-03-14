import logging
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from app.db import get_db_connection
from app.services.embedding import generate_embedding, get_gemini_key
from app.services.retrieval import insert_document, search_similar_documents

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Example API with Gemini")


# -------------------------------
# Startup event: DB kontrol
# -------------------------------
@app.on_event("startup")
def connect_db():
    try:
        conn = get_db_connection()
        logging.info("✅ Database connection successful")
        conn.close()
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")

# -------------------------------
# Healthcheck
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# Document ekleme endpoint
# -------------------------------

class DocumentRequest(BaseModel):
    content: str

@app.post("/documents")
def add_document(request: DocumentRequest):
    content = request.content
    try:
        embedding_vector = generate_embedding(content)
        insert_document(
            content=content,
            embedding_vector=embedding_vector,
            source_name="manual",
            section_title="manual_insert",
            chunk_index=0,
)

        return {"status": "success", "content": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Similarity Search (Cosine)
# -------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def similarity_search(request: SearchRequest):
    try:
        query_text = request.query
        top_k = request.top_k
        query_vector = generate_embedding(query_text)
        results = search_similar_documents(query_vector, top_k)

        return {"query": query_text, "results": results}
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/gemini-test")
def gemini_test(prompt: str = Query(..., description="Test prompt for Gemini")):
    gemini_api_key = get_gemini_key()
    if not gemini_api_key:
        
        return {"error": "GEMINI_API_KEY not set"}

    # Ücretsiz API Key URL'nin sonuna eklenir, Header'a değil!
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # JSON yapısı Gemini'de her zaman 'contents' ile başlar
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.encoding = "utf-8"
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Eğer hata varsa detayını görelim
        error_msg = response.text if 'response' in locals() else str(e)
        return {"error": error_msg}
    

@app.get("/gemini-test-mock")
def gemini_test_mock(prompt: str = Query(...)):
    return {
        "mock": True,
        "prompt_received": prompt,
        "response": f"Bu bir mock cevaptır: '{prompt}'"
    }
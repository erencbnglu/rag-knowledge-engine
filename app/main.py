import os
import logging
import psycopg2
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Example API with Gemini")

# -------------------------------
# Gemini API ayarları
# -------------------------------
GEMINI_API_URL = "https://gemini.googleapis.com/v1/embeddings"

def get_gemini_key():
    # Secret manager veya env dosyası kullan
    return os.getenv("GEMINI_API_KEY")

# -------------------------------
# DB Bağlantısı
# -------------------------------
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

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
        # ✅ Doğru Embedding Model + Endpoint (embedContent)
        EMBED_URL = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-embedding-001:embedContent?key={get_gemini_key()}"
        )

        payload = {
            "content": {
                "parts": [{"text": content}]
            },
            "outputDimensionality": 3072
        }

        resp = requests.post(EMBED_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Gemini error: {resp.text}")

        # ✅ Doğru response yolu
        embedding_vector = resp.json()["embedding"]["values"]

        # DB insert
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding_vector)
        )
        conn.commit()
        cur.close()
        conn.close()

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

        # 1️⃣ Query embedding üret
        EMBED_URL = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-embedding-001:embedContent?key={get_gemini_key()}"
        )

        payload = {
            "content": {
                "parts": [{"text": query_text}]
            },
            "outputDimensionality": 3072
        }

        resp = requests.post(EMBED_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Gemini error: {resp.text}")

        query_vector = resp.json()["embedding"]["values"]

        # psycopg2 list -> numeric[] olarak gider; <=> operatörü vector beklediği için
        # pgvector literal formatına çeviriyoruz: "[0.1,0.2,...]"
        query_vector_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

        # 2️⃣ Cosine similarity ile arama
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, content, embedding <=> (%s::vector) AS distance
            FROM documents
            ORDER BY embedding <=> (%s::vector)
            LIMIT %s;
            """,
            (query_vector_str, query_vector_str, top_k)
        )

        results = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "query": query_text,
            "results": [
                {
                    "id": row[0],
                    "content": row[1],
                    "cosine_distance": float(row[2])
                }
                for row in results
            ]
        }

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
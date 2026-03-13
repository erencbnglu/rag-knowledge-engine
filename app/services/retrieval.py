from app.db import get_db_connection


def insert_document(content: str, embedding_vector):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding_vector)
    )
    conn.commit()
    cur.close()
    conn.close()


def search_similar_documents(query_vector, top_k: int = 3):
    query_vector_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, content, embedding <=> %s::vector AS cosine_distance
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector_str, query_vector_str, top_k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = [
        {"id": row[0], "content": row[1], "cosine_distance": row[2]}
        for row in rows
    ]

    return results
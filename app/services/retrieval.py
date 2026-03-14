from app.db import get_db_connection


def insert_document(
    content: str,
    embedding_vector,
    source_name: str = "manual",
    section_title: str = "manual_insert",
    chunk_index: int = 0,
):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (content, embedding, source_name, section_title, chunk_index)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (content, embedding_vector, source_name, section_title, chunk_index)
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
        SELECT id, content, source_name, section_title, chunk_index, embedding <=> %s::vector AS cosine_distance
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
        {
            "id": row[0],
            "content": row[1],
            "source_name": row[2],
            "section_title": row[3],
            "chunk_index": row[4],
            "cosine_distance": row[5],
        }
        for row in rows
    ]

    return results
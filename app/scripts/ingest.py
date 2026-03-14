from pathlib import Path
from app.db import get_db_connection
from app.services.chunker import chunk_markdown
from app.services.embedding import generate_embedding
from app.services.retrieval import insert_document


def delete_documents_by_source(source_name: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE source_name = %s", (source_name,))
    conn.commit()
    cur.close()
    conn.close()


def ingest_file(file_path: Path):
    source_name = file_path.name
    markdown_text = file_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown_text)

    delete_documents_by_source(source_name)

    for chunk in chunks:
        embedding_vector = generate_embedding(chunk["content"])
        insert_document(
            content=chunk["content"],
            embedding_vector=embedding_vector,
            source_name=source_name,
            section_title=chunk["section_title"],
            chunk_index=chunk["chunk_index"],
        )

    print(f"{source_name} ingest edildi. Toplam chunk: {len(chunks)}")


def ingest_all_markdown_files():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    markdown_files = sorted(data_dir.glob("*.md"))

    if not markdown_files:
        print("data klasöründe .md dosyası bulunamadı.")
        return

    for file_path in markdown_files:
        ingest_file(file_path)


if __name__ == "__main__":
    ingest_all_markdown_files()

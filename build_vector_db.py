"""
build_vector_db.py
Loads books.json, generates embeddings for each book and its chapters,
and stores everything in a local ChromaDB vector database.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# ── CONFIG ──────────────────────────────────────────────────────────────────
BOOKS_JSON_PATH = "books.json"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "books"
MODEL_NAME      = "all-MiniLM-L6-v2"   # fast & good for semantic search
# ─────────────────────────────────────────────────────────────────────────────


def load_books(path: str) -> list[dict]:
    """Load and return the list of books from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    print(f"[INFO] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors."""
    return model.encode(texts, show_progress_bar=True).tolist()


def get_or_create_collection(db_path: str, collection_name: str):
    """Initialize ChromaDB client and return (or create) the collection."""
    client = chromadb.PersistentClient(path=db_path)
    # Delete existing collection so we can rebuild cleanly
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Deleted existing collection '{collection_name}'")
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    print(f"[INFO] Created collection '{collection_name}'")
    return collection


def build_documents(books: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Build three parallel lists for ChromaDB:
      - ids        : unique string ID per document
      - texts      : the text to embed
      - metadatas  : dict with title, url, type
    Each book gets one 'book' doc + one doc per chapter.
    """
    ids, texts, metadatas = [], [], []

    for i, book in enumerate(books):
        title       = book.get("title", "")
        url         = book.get("url", "")
        description = book.get("description", "")
        chapters    = book.get("chapters", [])

        # ── Book-level document ──────────────────────────────────────────────
        book_id   = f"book_{i}"
        book_text = f"{title}. {description}"
        ids.append(book_id)
        texts.append(book_text)
        metadatas.append({"title": title, "url": url, "type": "book"})

        # ── Chapter-level documents ──────────────────────────────────────────
        for j, chapter in enumerate(chapters):
            chap_id   = f"book_{i}_chap_{j}"
            chap_text = f"{title} - {chapter}"   # prefix with title for context
            ids.append(chap_id)
            texts.append(chap_text)
            metadatas.append({"title": title, "url": url, "type": "chapter"})

    return ids, texts, metadatas


def store_in_chromadb(collection, ids, embeddings, texts, metadatas):
    """Insert all documents into the ChromaDB collection in batches."""
    BATCH_SIZE = 100
    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids        = ids[start:end],
            embeddings = embeddings[start:end],
            documents  = texts[start:end],
            metadatas  = metadatas[start:end],
        )
    print(f"[INFO] Stored {total} documents in ChromaDB.")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    books      = load_books(BOOKS_JSON_PATH)
    print(f"[INFO] Loaded {len(books)} books.")

    model      = get_embedding_model(MODEL_NAME)
    collection = get_or_create_collection(CHROMA_DB_PATH, COLLECTION_NAME)

    ids, texts, metadatas = build_documents(books)
    print(f"[INFO] Total documents to embed: {len(texts)}")

    embeddings = embed_texts(model, texts)
    store_in_chromadb(collection, ids, embeddings, texts, metadatas)

    print("[DONE] Vector database built successfully!")
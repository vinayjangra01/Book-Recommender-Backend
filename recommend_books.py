"""
recommend_books.py
Takes a user problem as terminal input, queries ChromaDB for the top
matching chunks (books + chapters), aggregates by book, and prints the
top 3 recommendations with explanations.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "books"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K_RESULTS   = 10   # fetch more results so we can aggregate across books
TOP_N_BOOKS     = 3    # final books to show
# ─────────────────────────────────────────────────────────────────────────────


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformer model (cached after first load)."""
    return SentenceTransformer(model_name)


def load_collection(db_path: str, collection_name: str):
    """Connect to the persisted ChromaDB and return the collection."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name=collection_name)


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed the user query into a vector."""
    return model.encode([query])[0].tolist()


def query_chromadb(collection, query_embedding: list[float], top_k: int) -> dict:
    """
    Run a vector similarity search.
    Returns raw ChromaDB results dict with distances, documents, metadatas.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def aggregate_by_book(results: dict) -> dict[str, dict]:
    """
    Group ChromaDB hits by book title.
    For each book we track:
      - total_score  : sum of (1 - cosine_distance) — higher = better
      - hit_count    : how many chunks matched
      - url          : book URL
      - matched_docs : list of matched chapter/book texts
    """
    book_scores = defaultdict(lambda: {
        "total_score": 0.0,
        "hit_count": 0,
        "url": "",
        "matched_docs": [],
    })

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        title      = meta["title"]
        similarity = 1 - dist          # convert cosine distance → similarity
        book_scores[title]["total_score"]  += similarity
        book_scores[title]["hit_count"]    += 1
        book_scores[title]["url"]           = meta["url"]
        book_scores[title]["matched_docs"].append((similarity, doc))

    return book_scores


def rank_books(book_scores: dict, top_n: int) -> list[tuple]:
    """
    Sort books by total_score descending and return the top N.
    Returns list of (title, score_data) tuples.
    """
    sorted_books = sorted(
        book_scores.items(),
        key=lambda x: x[1]["total_score"],
        reverse=True,
    )
    return sorted_books[:top_n]


def build_explanation(matched_docs: list[tuple], max_snippets: int = 2) -> str:
    """
    Build a short explanation from the top-scoring matched chapter texts.
    Shows the best 'max_snippets' matched snippets.
    """
    # Sort by similarity score descending
    top_matches = sorted(matched_docs, key=lambda x: x[0], reverse=True)[:max_snippets]
    snippets = []
    for score, doc in top_matches:
        # Truncate long docs for display
        short = doc if len(doc) < 120 else doc[:117] + "..."
        snippets.append(f'  • "{short}" (score: {score:.3f})')
    return "\n".join(snippets)


def print_recommendations(ranked_books: list[tuple]) -> None:
    """Pretty-print the final book recommendations."""
    print("\n" + "═" * 60)
    print("  📚  TOP BOOK RECOMMENDATIONS")
    print("═" * 60)

    for rank, (title, data) in enumerate(ranked_books, start=1):
        explanation = build_explanation(data["matched_docs"])
        print(f"\n#{rank}  {title}")
        print(f"    🔗  {data['url']}")
        print(f"    🎯  Relevance hits : {data['hit_count']}  |  "
              f"Total score : {data['total_score']:.3f}")
        print(f"    📝  Why this book?")
        print(explanation)

    print("\n" + "═" * 60 + "\n")


def recommend(query: str) -> list[dict]:
    """
    Full pipeline: embed → query → aggregate → rank → return results.
    Returns a list of dicts (used by both CLI and app.py).
    """
    model      = get_embedding_model(MODEL_NAME)
    collection = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)

    query_vec   = embed_query(model, query)
    raw_results = query_chromadb(collection, query_vec, TOP_K_RESULTS)
    book_scores = aggregate_by_book(raw_results)
    ranked      = rank_books(book_scores, TOP_N_BOOKS)

    # Build clean output list
    output = []
    for title, data in ranked:
        output.append({
            "title":       title,
            "url":         data["url"],
            "hit_count":   data["hit_count"],
            "total_score": round(data["total_score"], 4),
            "explanation": build_explanation(data["matched_docs"]),
        })
    return output


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📖  Book Recommendation System")
    print("Enter your problem or what you're looking for:")
    user_query = input(">>> ").strip()

    if not user_query:
        print("[ERROR] Empty query. Please describe your problem.")
    else:
        results = recommend(user_query)
        # Format for CLI
        print_recommendations(
            [(r["title"], {
                "url": r["url"],
                "hit_count": r["hit_count"],
                "total_score": r["total_score"],
                "matched_docs": [],  # already formatted in explanation
            }) for r in results]
        )
        # Print explanations separately since we pre-formatted them
        for i, r in enumerate(results, 1):
            print(f"#{i} Explanation:\n{r['explanation']}\n")
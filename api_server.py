"""
api_server.py
Flask REST API that exposes the recommendation engine to the Android app.
Endpoint: POST /recommend  { "query": "I feel lonely and confused" }
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from recommend_books import recommend

app = Flask(__name__)
CORS(app)  # Allow requests from Android emulator / device


@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    """
    Expects JSON body: { "query": "user problem text" }
    Returns: { "books": [ { title, url, hit_count, total_score, explanation } ] }
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        results = recommend(query)
        return jsonify({"books": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
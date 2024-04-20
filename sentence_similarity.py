import pandas as pd
import tensorflow_hub as hub  # contains USE4
from tensorflow import make_ndarray, make_tensor_proto
from numpy import dot, norm

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
    return model(input)

def calculate_similarity(text1, text2):
    messages = [text1, text2]
    message_embeddings = embed(messages)
    a = make_ndarray(make_tensor_proto(message_embeddings))
    cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))
    return cos_sim

def handle_request(request):
    """Responds to HTTP requests with a similarity score."""
    if request.method == "POST":
        request_data = request.get_json()
        text1 = request_data.get("text1")
        text2 = request_data.get("text2")
        if text1 and text2:
            similarity_score = calculate_similarity(text1, text2)
            return jsonify({"similarity score": float(similarity_score)}), 200
        else:
            return jsonify({"error": "Missing text1 or text2 in request body"}), 400
    else:
        return jsonify({"error": "Only POST requests are allowed"}), 405


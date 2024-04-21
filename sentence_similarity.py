from flask import Flask, jsonify, request
import tensorflow_hub as hub
from tensorflow import make_ndarray, make_tensor_proto
from numpy import dot, norm

app = Flask(__name__)

# Load Universal Sentence Encoder model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
    return model(input)

def calculate_similarity(text1, text2):
    embeddings = embed([text1, text2])
    a = make_ndarray(make_tensor_proto(embeddings))
    cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))
    return cos_sim

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_endpoint():
    """Endpoint to calculate similarity score between two texts."""
    request_data = request.get_json()
    text1 = request_data.get("text1")
    text2 = request_data.get("text2")

    if text1 is None or text2 is None:
        return jsonify({"error": "Missing 'text1' or 'text2' in request body"}), 400

    similarity_score = calculate_similarity(text1, text2)
    response_data = {"similarity_score": float(similarity_score)}
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run()

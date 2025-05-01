import os
import json
import uuid
from flask import Flask, request, jsonify
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_to_memory(question, answer):
    memory = load_memory()
    memory.append({"id": str(uuid.uuid4()), "question": question, "answer": answer})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def find_similar_memory(question):
    memory = load_memory()
    if not memory:
        return None

    corpus = [item["question"] for item in memory]
    corpus.append(question)
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])
    most_similar_index = sims.argmax()
    if sims[0][most_similar_index] > 0.7:
        return memory[most_similar_index]
    return None

def get_chatgpt_response(message):
    similar = find_similar_memory(message)
    if similar:
        return f"(Hafızadan): {similar['answer']}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Sen BetXline canlı destek botusun. Bonuslar, yatırım, çekim ve kurallarla ilgili her soruya net, doğru ve kararlı cevap ver."},
            {"role": "user", "content": message}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message["content"].strip()
    save_to_memory(message, answer)
    return answer

@app.route("/livechat", methods=["POST"])
def handle_incoming_event():
    data = request.get_json()

    try:
        event_value = data["payload"]["event"]["text"]
    except KeyError:
        return jsonify({"error": "Mesaj alınamadı"}), 400

    response_text = get_chatgpt_response(event_value)
    return jsonify({"response": response_text})

app = app
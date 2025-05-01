import os
import json
import uuid
import requests
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
        return f"(Yaddaşdan): {similar['answer']}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Sən BetXline canlı dəstək botusan. Bonuslar, yatırım, çıxarış və qaydalarla bağlı bütün suallara düzgün, net və qətiyyətli cavab ver."},
            {"role": "user", "content": message}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message["content"].strip()
    save_to_memory(message, answer)
    return answer

def send_message_to_livechat(chat_id, message):
    url = "https://api.livechatinc.com/v3.3/agent/action/send_event"
    headers = {
        "Authorization": f"Bearer {os.getenv('LIVECHAT_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "chat_id": chat_id,
        "event": {
            "type": "message",
            "text": message,
            "custom_id": str(uuid.uuid4())
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("LiveChat mesaj göndərmə xətası:", response.text)

@app.route("/livechat-message", methods=["POST"])
def livechat_message():
    data = request.get_json()
    try:
    message = data["payload"]["event"]["text"]
    chat_id = data["payload"]["event"]["chat_id"]
    print("GƏLƏN CHAT ID:", chat_id)

    except KeyError:
        return jsonify({"error": "Format xətası"}), 400

    bot_response = get_chatgpt_response(message)
    send_message_to_livechat(chat_id, bot_response)
    return jsonify({"status": "Cavab göndərildi"}), 200

@app.route("/livechat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "Mesaj boş ola bilməz"}), 400
        bot_response = get_chatgpt_response(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": f"Server xətası: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
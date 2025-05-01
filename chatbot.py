from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = OpenAI()

@app.route('/livechat', methods=['POST'])
def livechat_webhook():
    data = request.json
    kullanici_mesaji = data.get('message', '')

    yanit = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Sən bir canlı dəstək botsan. Yalnız Azərbaycan türkcəsində cavab ver."},
            {"role": "user", "content": kullanici_mesaji}
        ],
        temperature=0.7
    )

    cevap = yanit.choices[0].message.content
    return jsonify({"reply": cevap})

if __name__ == "__main__":
    app.run(port=5000)
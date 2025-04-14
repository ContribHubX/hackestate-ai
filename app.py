import asyncio
from flask import Flask, request, jsonify
from real_state_chat_agent import ChatAgentFactory
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://1cz2hd3b-5173.asse.devtunnels.ms",
            "http://localhost:5173",
            "https://your-frontend.com"
        ]
    }
}, supports_credentials=True)
chat_agent = ChatAgentFactory()
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/agent/chat", methods=["POST"])
def ask_chat_agent():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    try:
        response = asyncio.run(chat_agent.ask_real_estate_bot(query))
        return jsonify(response)  # Return the parsed JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("app running");
    app.run(debug=True)

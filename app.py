import asyncio
from flask import Flask, request, jsonify
from real_state_chat_agent import ChatAgentFactory
import json


app = Flask(__name__)
chat_agent = ChatAgentFactory()
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/ask", methods=["POST"])
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

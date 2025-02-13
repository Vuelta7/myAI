import os

from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

qa_pipeline = pipeline("question-answering",
                       model="deepset/roberta-base-squad2")


@app.route('/qna', methods=['POST'])
def answer_question():
    data = request.get_json()
    context = data.get("context")
    question = data.get("question")

    if not context or not question:
        return jsonify({"error": "Missing context or question"}), 400

    answer = qa_pipeline(question=question, context=context)
    return jsonify({"answer": answer["answer"]})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

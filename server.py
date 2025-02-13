from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

# Load the Q&A model
qa_pipeline = pipeline("question-answering",
                       model="deepset/roberta-base-squad2")


@app.route('/qna', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON format"}), 400

        context = data.get("context")
        question = data.get("question")

        if not context or not question:
            return jsonify({"error": "Missing context or question"}), 400

        answer = qa_pipeline(question=question, context=context)
        return jsonify({"answer": answer["answer"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

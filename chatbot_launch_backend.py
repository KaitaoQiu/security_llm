from flask import Flask, request, jsonify
from chatbot_inference import generate_answer, load_model

app = Flask(__name__)

pipeline = load_model()

@app.route('/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question', '')
    answer = generate_answer(question, pipeline)
    return jsonify({'answer': answer})

@app.route('/')
def home():
    return app.send_static_file('chatbot_index.html')


if __name__ == '__main__':
    app.run(debug=False)

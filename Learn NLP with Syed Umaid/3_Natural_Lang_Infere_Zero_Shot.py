from flask import *
from PIL import Image
import base64
import io
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['travel', 'cooking', 'dancing']



app = Flask(__name__)

@app.route("/home")
def ghar():
    return "House"



@app.route("/ML",methods=['POST'])
def process():

    message = request.json.get('message')
    result = classifier(message, candidate_labels)
    return jsonify({'status': 'Success', 'message': result}), 400


@app.route("/ML-multi",methods=['POST'])
def process_multi():

    message = request.json.get('message')
    result = classifier(message, candidate_labels,multi_label=True)
    return jsonify({'status': 'Success', 'message': result}), 400





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)


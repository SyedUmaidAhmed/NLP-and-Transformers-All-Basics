from flask import *
import base64
import io
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline



#Model Hosted on Hugging Face
model = SentenceTransformer('all-MiniLM-L6-v2')
model_2 = SentenceTransformer('clips/mfaq')
model_3 = pipeline("question-answering")

app = Flask(__name__)


@app.route("/sen_transf_em",methods=['POST'])
def sen_embeddings():

    sentence = request.json.get('sent')

    print("Recieved: ",sentence)
    embeddings = model.encode(sentence)

    print(embeddings)

    
    return jsonify({'status': 'Success','input':sentence,'emb_matrix': 'Embedidngs Stored'}), 400


@app.route("/sen_similarity",methods=['POST'])
def sen_similarity():

    sentence1 = request.json.get('sentence1')
    sentence2 = request.json.get('sentence2')

    embed_1 = model.encode(sentence1)
    embed_2 = model.encode(sentence2)
    cos_sim = util.cos_sim(embed_1,embed_2)
    print("How similar these two sentences are: ", cos_sim)

    return jsonify({'status': 'Successully matched','similarity': str(cos_sim[0])}), 400



# Which Document has my answer ? Semantic Search
# What is Actually the Answer ? QA model of Tf

@app.route("/semantic_search",methods=['POST'])
def sem_search():

    question = "How many models I can run on my PC?"
    answer1 = "We are moving to travel the country"
    answer2 = "I want to see the whole world from clouds"
    answer3 = "My Computer has a powerful SSD Drive"



    question_embed = model_2.encode(question)
    corpuses = model_2.encode([answer1, answer2, answer3])

    print(util.semantic_search(question_embed,corpuses))

    result = util.semantic_search(question_embed,corpuses)

    return jsonify({'status': 'Successully Searched','Searcing results': result}), 400




@app.route("/qa_model",methods=['POST'])
def sem_qa():

    question = "How many models I can run on my PC?"
    corpus = "My Computer has a powerful SSD Drive and GPU with capability of almost running any high computational models"

    resulting = model_3(question=question, context=corpus)



    return jsonify({'status': 'Successully Find Exact Answer','Results': str(resulting)}), 400



















if __name__=="__main__":
    app.run(host="0.0.0.0", port=6000)

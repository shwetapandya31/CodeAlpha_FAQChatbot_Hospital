from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_utils import preprocess
import os
print("FILES:", os.listdir())
print("TEMPLATES:", os.listdir('templates'))

app = Flask(__name__, 
            template_folder=os.path.join(os.getcwd(), 'templates'),
            static_folder=os.path.join(os.getcwd(), 'static'))

# Load FAQs
with open('data/faqs.json') as f:
    faqs = json.load(f)

questions = [preprocess(faq['question']) for faq in faqs]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    processed_input = preprocess(user_input)
    
    user_vec = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vec, X)
    
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score < 0.3:
        return jsonify({"response": "Sorry, I couldn't understand your question. Please try again."})

    response = faqs[best_match_index]["answer"]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
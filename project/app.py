import os
import re

import joblib
import nltk
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load updated models and vectorizer using joblib
rf_model = joblib.load(os.path.join(BASE_DIR, "backend", "svm_sentiment_model.pkl"))
cv = joblib.load(os.path.join(BASE_DIR, "backend", "tfidf_vectorizer.pkl"))
sc = joblib.load(os.path.join(BASE_DIR, "backend", "scaler.pkl"))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_tweet(tweet):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', tweet)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    return ' '.join(review)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_sentiment():
    if request.method == 'OPTIONS':
        return '', 200  # Respond to preflight
    try:
        data = request.get_json(force=True)
        tweet = data.get("text", "")
        if not tweet or not tweet.strip():
            return jsonify({"error": "No text provided for analysis"}), 400

        cleaned = preprocess_tweet(tweet)
        vectorized = cv.transform([cleaned]).toarray()
        vectorized_scaled = sc.transform(vectorized)
        prediction = rf_model.predict(vectorized_scaled)[0]

        sentiment = "positive" if prediction == 1 else "negative"
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
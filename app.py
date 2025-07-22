from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# CSV読み込み
df = pd.read_csv("directory_list_utf8.csv", encoding="utf-8")

# TF-IDF初期化（n-gramで柔軟性を高める）
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["カテゴリ名"])

# Flaskアプリ初期化
app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "")
    input_vec = vectorizer.transform([keyword])
    similarities = cosine_similarity(input_vec, tfidf_matrix)

    best_index = similarities.argmax()
    best_score = similarities[0, best_index]

    if best_score < 0.2:
        return jsonify({"directory_id": "見つかりません"})

    matched_id = df.iloc[best_index]["ディレクトリID"]
    return jsonify({"directory_id": matched_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
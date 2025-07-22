from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# CSV読み込み
df = pd.read_csv("directory_list_utf8.csv", encoding="utf-8")
categories = df["ディレクトリID"].tolist()
labels = df["カテゴリ名"].tolist()

# TF-IDFベクトル化（n-gram指定で柔軟に）
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
category_vectors = vectorizer.fit_transform(labels)

# Flaskアプリ
app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "")

    if not keyword.strip():
        return jsonify({"directory_id": "見つかりません"})

    # 入力文をベクトルに変換
    input_vec = vectorizer.transform([keyword])

    # 類似度を計算
    cosine_scores = cosine_similarity(input_vec, category_vectors)[0]
    best_index = cosine_scores.argmax()
    best_score = cosine_scores[best_index]

    # 類似度しきい値（調整可能）
    if best_score < 0.05:
        return jsonify({"directory_id": "見つかりません"})

    matched_id = categories[best_index]
    return jsonify({"directory_id": matched_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
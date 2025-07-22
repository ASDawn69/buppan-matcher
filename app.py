from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# CSV読み込み（ヘッダー行を正しく認識）
df = pd.read_csv("directory_list_true.csv", encoding="utf-8", header=0)

# 前処理関数
def preprocess_text(text):
    text = text.replace("・", " ").replace(">>", " ")
    text = text.replace("　", " ")
    return text.strip()

# 前処理済みカテゴリ名リストとIDリスト
labels = [preprocess_text(str(label)) for label in df["カテゴリ名"]]
categories = df["ディレクトリID"].tolist()

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
category_vectors = vectorizer.fit_transform(labels)

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"directory_id": "見つかりません"})

    processed_keyword = preprocess_text(keyword)
    input_vec = vectorizer.transform([processed_keyword])
    cosine_scores = cosine_similarity(input_vec, category_vectors)[0]

    best_index = cosine_scores.argmax()
    best_score = cosine_scores[best_index]

    if best_score < 0.05:
        return jsonify({"directory_id": "見つかりません"})

    matched_id = categories[best_index]
    return jsonify({"directory_id": matched_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

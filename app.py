from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np

# CSV読み込み
CSV_PATH = "directory_list_true.csv"
df = pd.read_csv(CSV_PATH, encoding="utf-8", header=0)

# 前処理関数
def preprocess_text(text):
    return (
        str(text)
        .replace("・", " ")
        .replace(">>", " ")
        .replace("　", " ")
        .strip()
    )

labels = [preprocess_text(x) for x in df["カテゴリ名"]]
ids = df["ディレクトリID"].tolist()

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
category_vectors = vectorizer.fit_transform(labels)

app = Flask(__name__)

def predict_directory_id(keyword, topn=10):
    processed = preprocess_text(keyword)
    v = vectorizer.transform([processed])
    sims = cosine_similarity(v, category_vectors)[0]

    # 上位topnのインデックス
    idx_top = sims.argsort()[::-1][:topn]

    # rapidfuzzで再スコアリング
    refined = []
    for i in idx_top:
        fuzz_score = fuzz.token_sort_ratio(processed, labels[i]) / 100.0
        combined = 0.7 * sims[i] + 0.3 * fuzz_score
        refined.append((combined, i))

    refined.sort(reverse=True)
    best_score, best_idx = refined[0]
    return ids[best_idx], labels[best_idx], best_score

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"directory_id": "見つかりません"})

    matched_id, matched_label, score = predict_directory_id(keyword)
    if score < 0.05:
        return jsonify({"directory_id": "見つかりません"})

    return jsonify({
        "directory_id": matched_id,
        "category_name": matched_label,
        "score": float(score)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# CSV読み込み
df = pd.read_csv("directory_list_utf8.csv", encoding="utf-8")
categories = df["ディレクトリID"].tolist()
labels = df["カテゴリ名"].tolist()

# sBERTモデルのロード（多言語対応モデル）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# カテゴリ名をベクトル化して保持
category_embeddings = model.encode(labels, convert_to_tensor=True)

# Flaskアプリ初期化
app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "")
    
    # 入力が空ならエラー
    if not keyword.strip():
        return jsonify({"directory_id": "見つかりません"})
    
    # 入力キーワードをベクトル化
    input_embedding = model.encode(keyword, convert_to_tensor=True)

    # 類似度計算
    cosine_scores = util.cos_sim(input_embedding, category_embeddings)[0]

    # 最もスコアの高いインデックスを取得
    best_index = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_index].item()

    # 閾値を設定（0.35くらいがおすすめ）
    if best_score < 0.35:
        return jsonify({"directory_id": "見つかりません"})

    matched_id = categories[best_index]
    return jsonify({"directory_id": matched_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
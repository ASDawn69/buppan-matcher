from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.fasttext import load_facebook_model
import MeCab
import os
import urllib.request
import gzip
import shutil

# ========= 設定 =========
CSV_PATH = "directory_list_true.csv"
MODEL_PATH = "cc.ja.300.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.bin.gz"
STOPWORDS = {"商品", "アイテム", "レディース", "メンズ"}
# ========================

# -------- モデルが無ければ自動でダウンロード --------
def prepare_fasttext_model():
    if os.path.exists(MODEL_PATH):
        print("FastTextモデルは既に存在します")
        return
    print("FastTextモデルがありません。ダウンロードを開始します...")
    gz_path = MODEL_PATH + ".gz"
    urllib.request.urlretrieve(MODEL_URL, gz_path)
    print("ダウンロード完了。解凍中...")
    with gzip.open(gz_path, 'rb') as f_in, open(MODEL_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print("FastTextモデル準備完了")

# -------- Flaskアプリ --------
app = Flask(__name__)

print("起動準備開始...")
prepare_fasttext_model()

# CSV読み込み
df = pd.read_csv(CSV_PATH, encoding="utf-8")
categories = df["カテゴリ名"].tolist()
ids = df["ディレクトリID"].tolist()

# FastTextモデル読み込み
tagger = MeCab.Tagger("-Owakati")
print("FastTextモデルロード中...")
model = load_facebook_model(MODEL_PATH)

def preprocess(text: str):
    text = str(text).replace("・", " ").replace(">>", " ").replace("　", " ")
    words = tagger.parse(text).strip().split()
    return [w for w in words if w not in STOPWORDS]

def vectorize(text: str):
    words = preprocess(text)
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# カテゴリベクトルのキャッシュ
vec_cache = "category_vecs.npy"
if os.path.exists(vec_cache):
    print("キャッシュを読み込みます")
    category_vecs = np.load(vec_cache)
else:
    print("カテゴリベクトルを計算中...")
    category_vecs = np.vstack([vectorize(cat) for cat in categories])
    np.save(vec_cache, category_vecs)
print("初期化完了")

def predict_directory_id(keyword, topn=1):
    v = vectorize(keyword).reshape(1, -1)
    sims = cosine_similarity(v, category_vecs)[0]
    top_idx = sims.argsort()[::-1][:topn]
    return [(ids[i], categories[i], sims[i]) for i in top_idx]

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"directory_id": "見つかりません"})

    results = predict_directory_id(keyword, topn=1)
    matched_id, matched_cat, score = results[0]

    if score < 0.05:
        return jsonify({"directory_id": "見つかりません"})

    return jsonify({
        "directory_id": matched_id,
        "category_name": matched_cat,
        "score": float(score)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
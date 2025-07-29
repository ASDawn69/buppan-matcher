import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.fasttext import load_facebook_model
import MeCab

# CSV読み込み
df = pd.read_csv("directory_list_true.csv", encoding="utf-8")

# FastTextモデル読み込み
tagger = MeCab.Tagger("-Owakati")
model = load_facebook_model("cc.ja.300.bin")

# ストップワード
stopwords = {"商品", "アイテム"}

# 前処理
def preprocess(text: str):
    text = str(text).replace("・", " ").replace(">>", " ").replace("　", " ")
    words = tagger.parse(text).strip().split()
    return [w for w in words if w not in stopwords]

# ベクトル化
def vectorize(text: str):
    words = preprocess(text)
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# カテゴリ名とディレクトリIDを抽出
categories = df["カテゴリ名"].tolist()
ids = df["ディレクトリID"].tolist()

# カテゴリを事前にベクトル化
print("カテゴリのベクトル化中...")
category_vecs = np.vstack([vectorize(cat) for cat in categories])
print("完了")

# 入力キーワードから最も近いIDを返す関数
def predict_directory_id(keyword, topn=1):
    v = vectorize(keyword).reshape(1,-1)
    sims = cosine_similarity(v, category_vecs)[0]
    top_idx = sims.argsort()[::-1][:topn]
    return [(ids[i], categories[i], sims[i]) for i in top_idx]

# テスト
print(predict_directory_id("宇宙 ドキュメンタリー", topn=3))
FROM python:3.10-slim

# 必要なシステム依存ライブラリをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存パッケージ
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# アプリ本体
COPY . .

# ポート指定
ENV PORT=5000
EXPOSE 5000

# 起動コマンド
CMD ["python", "app.py"]
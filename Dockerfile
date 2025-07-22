FROM python:3.10-slim-bookworm

# システムパッケージを最小限に
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python依存関係をインストール
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]
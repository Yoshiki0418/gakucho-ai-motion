# ベースイメージ
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

USER root

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/bin:$PATH"
ENV TZ=Asia/Tokyo

# 必要なパッケージをインストール
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    build-essential \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    xz-utils \
    file \
    sudo \
    ffmpeg \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ワークディレクトリを設定
WORKDIR /workspace

# 必要なPythonライブラリをインストール
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -c "import pyopenjtalk; pyopenjtalk._lazy_init()"

# 公開されている日本語DeBERTaモデルを取得する。
# 認証情報をDockerレイヤーやビルドログへ埋め込まない。
RUN git clone https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm \
    /usr/local/lib/python3.10/dist-packages/style_bert_vits2/bert/deberta-v2-large-japanese-char-wwm || echo "Model clone failed or already exists"

# 実行ユーザーの作成
RUN useradd -m appuser

# アプリケーションコードは最後にコピーし、コード変更時もモデル層を再利用する。
COPY . .

USER appuser

# 起動コマンド
# 起動コマンド（.envの環境変数はdocker-composeなどで上書き可能）
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8076"]

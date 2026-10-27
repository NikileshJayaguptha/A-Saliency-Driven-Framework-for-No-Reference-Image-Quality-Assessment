# CPU build (default):  docker build -t iqa .
# GPU build:            docker build -t iqa-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/checkpoints/bestnatmodel.pth

WORKDIR /app

RUN pip install torch torchvision --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY iqa ./iqa
COPY app ./app
COPY train.py evaluate.py predict.py ./

RUN mkdir -p /app/checkpoints /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

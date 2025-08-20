FROM python:3.11.9-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use a different PyPI mirror
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY artifacts ./artifacts
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
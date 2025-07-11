FROM python:3.11-slim

WORKDIR /app

# dependency installation using uv
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache --requirement requirements.txt

COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "app/scripts/eval_debug.py"]
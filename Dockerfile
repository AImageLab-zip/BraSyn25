FROM python:3.11-slim

WORKDIR /app

# dependency installation using uv
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache --requirement requirements.txt

# environment variables for our cluster
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output

# run id: choose one of: 14, 21, 23, 24, 25
ARG RUN_ID=24

# view: choose one of: axi, sag, cor
ENV VIEW=axi

# copying the fundamental files
COPY main.py utils.py model.py auto_vars.sh download_checkpoint.py ./

# group encoder: true only for runs 24 and 25
RUN bash /app/auto_vars.sh

# downloading the checkpoint:
RUN python /app/download_checkpoint.py --run-id $RUN_ID

CMD ["python", "/app/main.py"]
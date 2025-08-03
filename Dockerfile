FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# dependency installation using uv
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache --requirement requirements.txt

# Create input/output dirs if they don't exist
RUN mkdir -p /input /output
# run id: choose one of: 14, 21, 23, 24, 25
ARG RUN_ID=24

# view: choose one of: axi, sag, cor
ENV VIEW=axi

# change these according to the run number
ENV GRP_ENCODER=true
ENV INFUSE_VIEW=false
# copying the fundamental files
COPY main.py utils.py model.py download_checkpoint.py ./

# downloading the checkpoint:
RUN python /app/download_checkpoint.py --run-id $RUN_ID

CMD ["python", "/app/main.py"]
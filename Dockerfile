FROM python:3.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 python3-tk fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py *.json *.md ./
RUN mkdir -p models
CMD ["python", "main.py", "--help"]

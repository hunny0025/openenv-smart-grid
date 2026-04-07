# Use Python 3.10+ as required by openenv-core
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything including pyproject.toml and uv.lock
COPY . .

# Install the environment as a package
RUN pip install -e .

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start using the installed entry point or uvicorn directly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

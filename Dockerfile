FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set PATH for uv
ENV PATH="/root/.local/bin:${PATH}"

# Copy ONLY dependency files first (for caching)
COPY pyproject.toml ./

# Install dependencies
RUN /root/.local/bin/uv pip install --system --no-cache -e .

# Copy the rest of the application code
COPY app ./app
COPY data ./data

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

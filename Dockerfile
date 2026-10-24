# Use official Python 3.13 slim image
FROM python:3.13-slim-bookworm

# Install system dependencies including ffmpeg for media transcoding and nodejs for yt-dlp JS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation and optimize Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies without dev packages
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the project source code
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# Create non-root user and data directories for runtime security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/downloads /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Run bot
CMD ["python", "bot.py"]


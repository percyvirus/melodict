# Use official lightweight Python 3.12 image
FROM python:3.12-slim-bookworm

# Install system dependencies required for audio processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies without creating a virtualenv (isolated inside container)
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source code and scripts
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Expose UDP ports for OSC communication (Max -> Python: 8001, Python -> Max: 8000)
EXPOSE 8000/udp 8001/udp

# Set default command to run the OSC server
CMD ["uv", "run", "python", "-m", "melodict.osc.server"]
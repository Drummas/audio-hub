FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libfreetype6-dev \
    libpng-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Matplotlib cache
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /tmp/matplotlib

# Accept Coqui XTTS license non-interactively
ENV COQUI_TOS_AGREED=1

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create required directories
RUN mkdir -p /app/files /app/batches /app/waveforms /app/speakers

# Copy application
COPY . .

# Permissions (NAS-friendly)
RUN chmod -R 777 /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

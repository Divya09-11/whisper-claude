# Stage 1: Install Python Dependencies
FROM python:3.8-slim AS builder
WORKDIR /app

# Copy requirements files
COPY requirements.txt ./

# Install dependencies into /deps
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt -t /deps

# Stage 2: Download Whisper Model
FROM python:3.8-slim AS whisper_model
WORKDIR /app
COPY --from=builder /deps /deps
ENV PYTHONPATH=/deps
RUN python3 -c "import whisper; whisper.load_model('medium')" && \
    mv /root/.cache/whisper /whisper_cache

# Stage 3: Final AWS Lambda-Compatible Image
FROM public.ecr.aws/lambda/python:3.8
WORKDIR /var/task

# Install FFmpeg with minimal dependencies
RUN yum update -y && \
    yum install -y wget tar xz && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf ffmpeg-*-amd64-static* && \
    yum remove -y wget tar xz && \
    yum clean all && \
    rm -rf /var/cache/yum

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV XDG_CACHE_HOME=/tmp/whisper_models
ENV PYTHONPATH="${PYTHONPATH}:/var/task"
ENV FFMPEG_PATH="/usr/local/bin/ffmpeg"
ENV FFPROBE_PATH="/usr/local/bin/ffprobe"

# Create necessary directories
RUN mkdir -p /tmp/whisper_models /tmp/audio && \
    chmod 777 /tmp/whisper_models /tmp/audio

# Copy dependencies and model
COPY --from=builder /deps /var/task/
COPY --from=whisper_model /whisper_cache /tmp/whisper_models/whisper

# Copy application files
COPY lambda_handler.py services.py ./

# Verify installations
RUN ffmpeg -version && \
    python3 -c "import whisper; print('Whisper installation verified')"

# CMD for AWS Lambda
CMD ["lambda_handler.handler"]
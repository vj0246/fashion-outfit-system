# Dockerfile for the FastAPI backend ONLY.
#
# This does NOT go on Vercel -- Vercel's serverless Python functions cap
# out around 250MB unzipped, and sentence-transformers pulls in torch
# transitively even though we skip loading FashionCLIP/CLIP at serving
# time, which blows past that. Deploy this image to Render, Fly.io, or a
# Hugging Face Spaces Docker space instead (all have free tiers that run
# real containers, not size-capped serverless functions). The Vercel
# static site in web/ talks to whichever URL this container ends up at.
#
# Build (from the project root, NOT from backend/ -- needs access to the
# top-level *.py files and the processed/ folder):
#   docker build -t fashion-backend .
# Run:
#   docker run -p 8000:8000 -e GROQ_API_KEY=your-key fashion-backend
#
# IMPORTANT: run data_pipeline.py -> embeddings.py -> compatibility.py
# locally FIRST so ./processed/ is populated before building this image --
# the container serves precomputed artifacts, it does not generate them.

FROM python:3.11-slim

WORKDIR /app

# system deps for sentence-transformers/torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# only what serving actually needs: code + precomputed artifacts.
# NOT copying images/ (the Vercel static site serves those) or the
# training-time scripts (data_pipeline.py/embeddings.py) or their heavy
# deps -- keeps the image as small as this stack allows.
COPY retrieval.py compatibility.py assistant.py ./
COPY backend/main.py ./backend/main.py
COPY processed/ ./processed/

ENV PROCESSED_DIR=/app/processed
ENV DATA_DIR=/app
ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

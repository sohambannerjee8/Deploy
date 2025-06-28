#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting deployment script for PDF Embedding API (v8)..."

# --- Create logs directory ---
echo "🪵 Creating logs directory..."
mkdir -p logs

# --- Clean up previous failed download if it exists ---
echo "🧹 Removing potentially corrupted model file..."
rm -f models/Qwen3-Embedding-8B-Q4_K_M.gguf

# --- Create main.py ---
echo "📄 Creating main.py file..."
cat << 'EOF' > main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
import httpx
import pypdf
from llama_cpp import Llama
import io
import os
import json
import datetime
import hashlib

# Configuration
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q4_K_M.gguf"
MODEL_FILENAME = "Qwen3-Embedding-8B-Q4_K_M.gguf"
MODELS_DIR = "models"
EMBEDDINGS_DIR = "embeddings"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# In-memory store for the model
llm_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    - On startup, it downloads the model if not present and loads it.
    - On shutdown, it cleans up resources.
    """
    print("Application startup...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", MODEL_URL, follow_redirects=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    with open(MODEL_PATH, "wb") as f:
                        processed_bytes = 0
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            processed_bytes += len(chunk)
                            print(f"Downloading... {processed_bytes / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end="\r")

            print("\nModel downloaded successfully.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
    
    if os.path.exists(MODEL_PATH):
        try:
            llm = Llama(model_path=MODEL_PATH, embedding=True, n_ctx=8192, n_gpu_layers=-1) 
            llm_store["llm"] = llm
            print("Model loaded successfully onto GPU.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. The application will not be able to serve requests.")

    yield
    
    llm_store.clear()
    print("Application shutdown, model unloaded.")

app = FastAPI(
    title="PDF Embedding API",
    description="An API to generate embeddings for PDF documents using Qwen3-Embedding-8B.",
    version="0.1.0",
    lifespan=lifespan
)

class PDFUrlRequest(BaseModel):
    url: HttpUrl
    save_to_file: bool = False

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    filename: Optional[str] = None

@app.post("/embed-pdf/", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_embedding_for_pdf(request: PDFUrlRequest):
    """
    Accepts a URL to a PDF file, downloads it, extracts the text, 
    returns the corresponding embedding, and optionally saves it to a file.
    """
    if "llm" not in llm_store:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    llm = llm_store["llm"]

    try:
        headers = {"User-Agent": "RunPod-PDF-Embedder/1.0"}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(str(request.url), headers=headers)
            response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        embedding_result = llm.create_embedding(text)
        
        # FIX: Handle complex embedding structures and ensure JSON-serializable floats
        try:
            embedding_data = embedding_result['data'][0]['embedding']
            
            # The library sometimes returns a list containing the actual embedding list, e.g., [[0.1, 0.2, ...]]
            # We handle this by checking if the first element is a list.
            if embedding_data and isinstance(embedding_data[0], list):
                print("Note: Detected nested list in embedding output. Un-nesting.")
                embedding_data = embedding_data[0]
                
            embedding_vector = [float(v) for v in embedding_data]

        except (IndexError, TypeError) as e:
            print(f"!!! An error occurred while processing the embedding result: {e}")
            print(f"!!! The structure of the embedding result was: {embedding_result}")
            raise HTTPException(status_code=500, detail="Internal server error: Could not parse the embedding model's output.")
        
        saved_filename = None
        if request.save_to_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            url_hash = hashlib.md5(str(request.url).encode()).hexdigest()[:8]
            saved_filename = f"{timestamp}_{url_hash}.json"
            filepath = os.path.join(EMBEDDINGS_DIR, saved_filename)
            
            data_to_save = {
                "source_url": str(request.url),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                "embedding": embedding_vector
            }
            
            with open(filepath, "w") as f:
                json.dump(data_to_save, f, indent=2)
            print(f"Embedding saved to {filepath}")

        return {"embedding": embedding_vector, "filename": saved_filename}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Failed to download PDF from URL: {e.request.url}. Status code: {e.response.status_code}")
    except pypdf.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file provided.")
    except Exception as e:
        # Log the full error to the console/log file for debugging
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        # Return a more informative error to the user
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {type(e).__name__}")
EOF

# --- Create requirements.txt ---
echo "📄 Creating requirements.txt file..."
cat << EOF > requirements.txt
fastapi
uvicorn[standard]
pydantic
httpx
pypdf
llama-cpp-python
EOF

# --- Install dependencies ---
echo "📦 Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Reinstall llama-cpp-python with GPU support ---
echo "⚙️  Reinstalling llama-cpp-python with CUDA support for GPU acceleration..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# --- Run the application ---
echo "✅ Setup complete. Starting the FastAPI server..."
echo "🔗 Access the API at http://<your-pod-ip>:8000"
echo "📚 API docs available at http://<your-pod-ip>:8000/docs"
echo "🪵 All output is being logged to logs/server.log"
echo "⏳ The model will now be downloaded and loaded, which may take several minutes..."

uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a logs/server.log

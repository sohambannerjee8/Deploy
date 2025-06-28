#!/bin/bash
# EMERGENCY CPU-OPTIMIZED RUNPOD DEPLOYMENT
# Fixes 100% CPU utilization issue during deployment
set -e

echo "🚨 EMERGENCY CPU-OPTIMIZED QWEN3 DEPLOYMENT"
echo "=============================================="
echo "This script addresses 100% CPU utilization"
echo ""

# CPU optimization environment variables
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

# Set CPU governor to performance but limit threads
echo "⚙️  Setting CPU optimizations..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
fi

# Update system with CPU limits
echo "📦 Updating system packages..."
sudo apt-get update -y >/dev/null 2>&1

# Install essential packages
echo "📦 Installing essential packages..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    psutil \
    >/dev/null 2>&1

# Install Python packages with CPU optimization
echo "🐍 Installing Python packages..."
pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0 \
    PyMuPDF==1.23.0 \
    psutil==5.9.6 \
    >/dev/null 2>&1

# Install llama-cpp-python with CUDA but CPU optimization
echo "🔧 Installing llama-cpp-python with CPU optimization..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CXX_FLAGS='-O2' -DLLAMA_CUDA_F16=on" \
FORCE_CMAKE=1 \
pip3 install llama-cpp-python==0.2.20 --no-cache-dir >/dev/null 2>&1

# Download model with progress but limited bandwidth
echo "📥 Downloading Qwen3-Embedding model..."
if [ ! -f "Qwen3-Embedding-8B-Q4_K_M.gguf" ]; then
    wget -q --show-progress --limit-rate=50m \
        "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q4_K_M.gguf"
fi

# Create CPU-optimized main.py
echo "📝 Creating CPU-optimized service..."
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
EMERGENCY CPU-OPTIMIZED Qwen3-Embedding Service for RunPod
Addresses 100% CPU utilization with aggressive throttling
"""

# CRITICAL: Set CPU limits BEFORE importing heavy libraries
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import asyncio
import gc
import time
import logging
import psutil
from asyncio import Semaphore
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Heavy imports after environment setup
from llama_cpp import Llama
import fitz
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL CPU CONFIGURATION
CPU_CONFIG = {
    "max_threads": 1,  # EMERGENCY: Single thread only
    "cpu_emergency_threshold": 85,  # Emergency brake
    "cpu_warning_threshold": 70,   # Warning throttle
    "chunk_delay": 0.5,            # 500ms delay between chunks
    "page_delay": 2.0,             # 2s delay between pages
    "emergency_sleep": 5.0,        # Emergency sleep duration
}

app = FastAPI(title="Emergency CPU-Optimized Qwen3 Service")
llama = None
pdf_semaphore = Semaphore(1)  # Only 1 PDF at a time

class CPUGuard:
    """Emergency CPU protection"""
    def __init__(self):
        self.emergency_count = 0
        self.throttle_count = 0
    
    async def check_and_throttle(self):
        cpu = psutil.cpu_percent(interval=0.5)
        
        if cpu > CPU_CONFIG["cpu_emergency_threshold"]:
            self.emergency_count += 1
            logger.warning(f"🚨 EMERGENCY CPU BRAKE: {cpu:.1f}%")
            await asyncio.sleep(CPU_CONFIG["emergency_sleep"])
            return True
        elif cpu > CPU_CONFIG["cpu_warning_threshold"]:
            self.throttle_count += 1
            throttle_time = (cpu - CPU_CONFIG["cpu_warning_threshold"]) * 0.1
            logger.info(f"⚠️  CPU throttling: {cpu:.1f}% for {throttle_time:.1f}s")
            await asyncio.sleep(throttle_time)
            return True
        
        return False

cpu_guard = CPUGuard()

class PDFRequest(BaseModel):
    pdf_url: str
    chunk_size: int = 200  # Smaller chunks for CPU optimization
    save_to_file: str = None

@app.middleware("http")
async def emergency_cpu_middleware(request: Request, call_next):
    """Emergency CPU throttling middleware"""
    cpu = psutil.cpu_percent()
    
    if cpu > 95:
        logger.error(f"🚨 CRITICAL CPU: {cpu:.1f}% - Service unavailable")
        return JSONResponse(
            status_code=503,
            content={
                "error": f"Service unavailable: CPU at {cpu:.1f}%",
                "retry_after": 30
            }
        )
    elif cpu > CPU_CONFIG["cpu_emergency_threshold"]:
        await asyncio.sleep(3)
    
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_system_stats():
    """Get system statistics"""
    try:
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "process_memory_mb": process.memory_info().rss / (1024**2),
            "threads_count": process.num_threads(),
        }
    except:
        return {"error": "Stats unavailable"}

def create_chunks(text: str, chunk_size: int, page_num: int):
    """Create text chunks with minimal CPU overhead"""
    if not text.strip():
        return []
    
    # Simple word-based chunking
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks),
                'page_number': page_num
            })
    
    return chunks

async def process_chunks_emergency_mode(chunks):
    """Process chunks with emergency CPU protection"""
    results = []
    
    for i, chunk in enumerate(chunks):
        # CPU guard before each chunk
        await cpu_guard.check_and_throttle()
        
        try:
            # Process single chunk
            embedding = llama.create_embedding(chunk['text'])
            results.append({
                **chunk,
                'embedding': embedding['data'][0]['embedding']
            })
            
            # MANDATORY delay between chunks
            await asyncio.sleep(CPU_CONFIG["chunk_delay"])
            
            # Aggressive garbage collection
            if (i + 1) % 3 == 0:  # Every 3 chunks
                gc.collect()
                await asyncio.sleep(1)  # Extra pause
                
            # Progress with CPU check
            if (i + 1) % 5 == 0:
                cpu = psutil.cpu_percent()
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks (CPU: {cpu:.1f}%)")
                
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            continue
    
    return results

async def process_pdf_emergency_mode(pdf_url: str, chunk_size: int):
    """Process PDF with emergency CPU protection"""
    start_time = time.time()
    
    try:
        # Download with timeout
        logger.info(f"📥 Downloading: {pdf_url}")
        response = requests.get(pdf_url, timeout=120)
        response.raise_for_status()
        
        # Parse PDF with CPU breaks
        logger.info("📄 Parsing PDF...")
        doc = fitz.open(stream=response.content, filetype="pdf")
        
        all_chunks = []
        page_count = doc.page_count
        
        for page_num in range(page_count):
            # CPU check before each page
            await cpu_guard.check_and_throttle()
            
            page = doc.load_page(page_num)
            text = page.get_text()
            
            chunks = create_chunks(text, chunk_size, page_num + 1)
            all_chunks.extend(chunks)
            
            # MANDATORY delay between pages
            await asyncio.sleep(CPU_CONFIG["page_delay"])
            
            # Garbage collection every 2 pages
            if (page_num + 1) % 2 == 0:
                gc.collect()
                cpu = psutil.cpu_percent()
                logger.info(f"Processed page {page_num + 1}/{page_count} (CPU: {cpu:.1f}%)")
        
        doc.close()
        
        # Process embeddings
        logger.info(f"🔮 Processing {len(all_chunks)} chunks...")
        processed_chunks = await process_chunks_emergency_mode(all_chunks)
        
        return {
            "total_chunks": len(processed_chunks),
            "processing_time": time.time() - start_time,
            "chunks": processed_chunks,
            "model": "Qwen3-Embedding-8B-Q4_K_M.gguf",
            "system_stats": get_system_stats(),
            "cpu_guard_stats": {
                "emergency_count": cpu_guard.emergency_count,
                "throttle_count": cpu_guard.throttle_count
            }
        }
        
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    """Initialize with emergency CPU protection"""
    global llama
    
    try:
        logger.info("🔧 Initializing model with EMERGENCY CPU limits...")
        
        # CRITICAL: Single-threaded initialization
        llama = Llama(
            model_path="Qwen3-Embedding-8B-Q4_K_M.gguf",
            embedding=True,
            verbose=False,
            n_ctx=2048,  # Reduced context
            n_threads=CPU_CONFIG["max_threads"],  # EMERGENCY: Single thread
            n_threads_batch=1,
            use_mmap=True,
            use_mlock=False
        )
        
        logger.info("✅ Model loaded with emergency CPU protection")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

@app.get("/")
async def root():
    """Service status with CPU monitoring"""
    cpu = psutil.cpu_percent()
    stats = get_system_stats()
    
    status = "critical" if cpu > 90 else "warning" if cpu > 75 else "normal"
    
    return {
        "message": "Emergency CPU-Optimized Qwen3 Service",
        "status": status,
        "cpu_percent": cpu,
        "model_loaded": llama is not None,
        "cpu_config": CPU_CONFIG,
        "system_stats": stats
    }

@app.get("/health")
async def health():
    """Health check with CPU protection"""
    cpu = psutil.cpu_percent(interval=1)
    
    return {
        "status": "healthy" if cpu < 80 else "degraded",
        "cpu_percent": cpu,
        "model_loaded": llama is not None,
        "cpu_guard_stats": {
            "emergency_count": cpu_guard.emergency_count,
            "throttle_count": cpu_guard.throttle_count
        }
    }

@app.get("/emergency-stats")
async def emergency_stats():
    """Emergency CPU statistics"""
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        "cpu_percent": cpu,
        "cpu_count": psutil.cpu_count(),
        "memory_percent": memory.percent,
        "process_cpu": process.cpu_percent(),
        "process_memory_mb": process.memory_info().rss / (1024**2),
        "threads_count": process.num_threads(),
        "cpu_guard": {
            "emergency_count": cpu_guard.emergency_count,
            "throttle_count": cpu_guard.throttle_count,
            "config": CPU_CONFIG
        },
        "status": "critical" if cpu > 90 else "warning" if cpu > 75 else "normal"
    }

@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    """Process PDF with emergency CPU protection"""
    if not llama:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Only allow one PDF at a time
    async with pdf_semaphore:
        try:
            result = await process_pdf_emergency_mode(
                request.pdf_url, 
                request.chunk_size
            )
            
            if request.save_to_file:
                with open(request.save_to_file, 'w') as f:
                    json.dump(result, f, indent=2)
                result["saved_to_file"] = request.save_to_file
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Emergency CPU protection for server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker only
        reload=False,
        access_log=False  # Reduce overhead
    )
EOF

# Set process limits
echo "⚙️  Setting process limits..."
ulimit -u 1024  # Limit processes
ulimit -n 1024  # Limit file descriptors

# Create systemd service with CPU limits
echo "📝 Creating systemd service..."
sudo tee /etc/systemd/system/qwen-embedding.service > /dev/null << EOF
[Unit]
Description=Emergency CPU-Optimized Qwen3 Embedding Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$(pwd)
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=NUMEXPR_NUM_THREADS=2
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

# CPU and memory limits
CPUQuota=70%
MemoryLimit=8G
TasksMax=50

[Install]
WantedBy=multi-user.target
EOF

# Start service
echo "🚀 Starting emergency CPU-optimized service..."
sudo systemctl daemon-reload
sudo systemctl enable qwen-embedding.service
sudo systemctl start qwen-embedding.service

# Wait for service to start
echo "⏳ Waiting for service to initialize..."
sleep 10

# Check service status
echo "📊 Service status:"
sudo systemctl status qwen-embedding.service --no-pager -l

# Test the service
echo "🧪 Testing service..."
for i in {1..5}; do
    echo "Attempt $i/5..."
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Service is responding!"
        break
    else
        echo "⏳ Waiting..."
        sleep 5
    fi
done

# Show emergency stats
echo "📊 Emergency CPU Statistics:"
curl -s http://localhost:8000/emergency-stats | python3 -m json.tool 2>/dev/null || echo "Service not responding yet"

echo ""
echo "🚨 EMERGENCY DEPLOYMENT COMPLETE"
echo "=================================="
echo "✅ Service deployed with aggressive CPU protection"
echo "✅ Single-threaded processing to prevent 100% CPU"
echo "✅ Emergency throttling at 85% CPU"
echo "✅ Mandatory delays between operations"
echo ""
echo "📊 Monitor CPU: curl http://localhost:8000/emergency-stats"
echo "🧪 Test PDF: curl -X POST -H 'Content-Type: application/json' \\"
echo "             -d '{\"pdf_url\":\"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf\"}' \\"
echo "             http://localhost:8000/process-pdf"
echo ""
echo "⚠️  Note: Processing will be slower but CPU-safe"
EOF 

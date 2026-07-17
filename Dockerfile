FROM python:3.11-slim

WORKDIR /app

# System dependencies required by docling (pdfium, image processing, OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force CPU-only torch inference (no GPU in on-premises environment)
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4

# Pre-warm docling model cache at build time to avoid cold-start delay
RUN python -c "\
from docling.document_converter import DocumentConverter; \
from docling.datamodel.pipeline_options import PdfPipelineOptions; \
opts = PdfPipelineOptions(); \
opts.do_ocr = False; \
opts.do_table_structure = True; \
print('Docling models ready')" || echo "Model pre-warm skipped, will download at runtime"

# Pre-warm RapidOCR models (ONNX) at build time — dois passos com fallback
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('RapidOCR ready')" \
    || echo "RapidOCR pre-warm skipped, will initialize at runtime"

# Instala dependências opcionais para PDFs escaneados (PaddleOCR + OpenCV).
# Separado do requirements.txt principal porque PaddlePaddle 3.x puxa NumPy 2.x,
# que pode conflitar com Docling. Instalação com || true: se falhar, o pipeline
# continua funcionando sem a camada de pré-processamento avançado.
COPY requirements-scanned.txt .
RUN pip install --no-cache-dir -r requirements-scanned.txt \
    && python -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(); \
print('PaddleOCR ready')" \
    || echo "PaddleOCR indisponível — pipeline funcionará sem pré-processamento de escaneados"

COPY src/ ./src/

RUN mkdir -p /app/input /app/output/json /app/output/xlsx /app/output/csv /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]

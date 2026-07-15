"""
Pré-processamento de PDFs escaneados com OpenCV + PaddleOCR.

Ativado quando o Docling detecta que o documento é escaneado (ocr_used=True)
para fornecer texto de qualidade superior antes do DeterministicParser.

Pipeline por página:
  1. PDF → imagens via PyMuPDF (fitz)
  2. OpenCV: binarização adaptativa, deskewing, remoção de ruído
  3. PaddleOCR: reconhecimento de texto e tabelas (mais robusto para documentos
     levemente tortos, fundos ruidosos e estruturas tabulares do que
     RapidOCR/Tesseract)
  4. Retorna texto por página e texto concatenado para alimentar o
     DeterministicParser e o LLM fallback com entrada de maior qualidade
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Imports opcionais — degradação graciosa se as dependências não estiverem
# instaladas
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PaddleOCR = None  # type: ignore[assignment,misc]
    _PADDLEOCR_AVAILABLE = False


# DPI para rasterização do PDF (300 dpi = boa qualidade para OCR)
_DPI = 300
# Ângulo máximo de inclinação a corrigir (graus)
_MAX_SKEW_DEG = 15.0


def is_available() -> bool:
    """Retorna True se todas as dependências do pré-processador estão instaladas."""
    return _CV2_AVAILABLE and _FITZ_AVAILABLE and _PADDLEOCR_AVAILABLE


class ScannedPreprocessor:
    """
    Extrai texto de PDFs escaneados com qualidade superior ao OCR padrão do Docling.

    Instancie uma vez e reutilize — PaddleOCR carrega os modelos somente no
    primeiro uso (lazy init).
    """

    def __init__(self) -> None:
        self._ocr: "_PaddleOCR | None" = None
        if not is_available():
            missing = []
            if not _CV2_AVAILABLE:
                missing.append("opencv-python-headless")
            if not _FITZ_AVAILABLE:
                missing.append("pymupdf")
            if not _PADDLEOCR_AVAILABLE:
                missing.append("paddleocr + paddlepaddle")
            logger.warning(
                f"ScannedPreprocessor: dependências ausentes ({', '.join(missing)}) — "
                f"pré-processamento desabilitado. Instale via requirements.txt."
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def extract(self, pdf_path: Path) -> Tuple[str, Dict[int, str]]:
        """
        Processa todas as páginas do PDF e retorna (full_text, page_texts).

        Retorna ('', {}) se as dependências não estiverem disponíveis ou se
        nenhum texto for extraído — o chamador deve tratar graciosamente.
        """
        if not is_available():
            return "", {}

        self._ensure_ocr_loaded()

        try:
            images = self._pdf_to_images(pdf_path)
        except Exception as exc:
            logger.warning(f"ScannedPreprocessor: falha ao rasterizar {pdf_path.name}: {exc}")
            return "", {}

        page_texts: Dict[int, str] = {}
        for page_no, img in images.items():
            try:
                preprocessed = self._preprocess_image(img)
                text = self._run_ocr(preprocessed)
                if text:
                    page_texts[page_no] = text
            except Exception as exc:
                logger.warning(
                    f"ScannedPreprocessor: erro na pág. {page_no} de {pdf_path.name}: {exc}"
                )

        full_text = "\n\n".join(page_texts[p] for p in sorted(page_texts))
        logger.info(
            f"ScannedPreprocessor: {pdf_path.name} — "
            f"{len(page_texts)} pág(s) | {len(full_text)} chars extraídos"
        )
        return full_text, page_texts

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _ensure_ocr_loaded(self) -> None:
        """Inicializa o PaddleOCR na primeira chamada (lazy)."""
        if self._ocr is None and _PADDLEOCR_AVAILABLE and _PaddleOCR is not None:
            logger.info("ScannedPreprocessor: inicializando PaddleOCR...")
            self._ocr = _PaddleOCR(
                use_angle_cls=True,
                lang="en",          # latin chars (incl. pt-BR) — melhor que 'pt' em versões atuais
                show_log=False,
                use_gpu=False,
            )
            logger.info("ScannedPreprocessor: PaddleOCR pronto.")

    def _pdf_to_images(self, pdf_path: Path) -> Dict[int, np.ndarray]:
        """Converte cada página do PDF em imagem numpy via PyMuPDF."""
        images: Dict[int, np.ndarray] = {}
        doc = fitz.open(str(pdf_path))
        zoom = _DPI / 72.0  # 72 dpi é o padrão do PyMuPDF
        mat = fitz.Matrix(zoom, zoom)
        for page_no, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            images[page_no] = arr
        doc.close()
        return images

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica pipeline de pré-processamento de imagem:
          1. Binarização adaptativa — converte para preto/branco absoluto para
             destacar o texto contra o fundo (tolera variações locais de iluminação)
          2. Remoção de ruído — morfologia erosão+dilatação elimina manchas de scanner
          3. Deskewing — endireita imagens levemente inclinadas (≤ MAX_SKEW_DEG)
        """
        # 1. Binarização adaptativa (Gaussian threshold — melhor para documentos)
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # 2. Remoção de ruído: fechamento morfológico (elimina manchas pequenas)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 3. Deskewing via análise de momentos sobre pixels de texto
        angle = self._detect_skew(clean)
        if abs(angle) > 0.3:  # ignora inclinações desprezíveis (< 0.3°)
            clean = self._rotate_image(clean, angle)

        return clean

    @staticmethod
    def _detect_skew(binary_img: np.ndarray) -> float:
        """
        Detecta o ângulo de inclinação via transformada de Hough sobre bordas.
        Retorna ângulo em graus (positivo = horário).
        Limita ao intervalo [-MAX_SKEW_DEG, MAX_SKEW_DEG] para evitar rotações absurdas.
        """
        # Inverte (texto preto → branco) para encontrar bordas de texto
        inv = cv2.bitwise_not(binary_img)
        edges = cv2.Canny(inv, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100,
                                 minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0.0

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # Considera apenas linhas próximas do horizontal (texto impresso)
                if abs(angle) <= _MAX_SKEW_DEG:
                    angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    @staticmethod
    def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
        """Rotaciona a imagem pelo ângulo dado (graus) ao redor do centro."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        # Fundo branco para os cantos expostos pela rotação
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return rotated

    def _run_ocr(self, img: np.ndarray) -> str:
        """Executa PaddleOCR na imagem pré-processada e retorna o texto extraído."""
        if self._ocr is None:
            return ""

        # PaddleOCR aceita ndarray diretamente
        results = self._ocr.ocr(img, cls=True)
        if not results or results[0] is None:
            return ""

        lines: list[str] = []
        for block in results[0]:
            if block and len(block) >= 2:
                text_conf = block[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                    text = text_conf[0]
                    if isinstance(text, str) and text.strip():
                        lines.append(text.strip())

        return "\n".join(lines)

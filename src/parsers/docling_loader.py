"""
Wrapper sobre a biblioteca docling para carregar PDFs e retornar uma
representação normalizada (texto completo + tabelas + markdown).

Configurado para rodar em CPU sem GPU. Usa Tesseract (idioma português) como
motor de OCR — mais estável em CPU que o motor padrão (RapidOCR/torch).

Estratégia de OCR "auto" (padrão): a primeira tentativa de conversão é sempre
sem OCR (rápida). Se o texto extraído for muito curto em relação ao número de
páginas — sinal de PDF escaneado/imagem — o documento é reconvertido
automaticamente com OCR forçado. Isso evita pagar o custo do OCR (lento em CPU)
para faturas que já têm texto extraível, mas garante que faturas escaneadas
sejam processadas corretamente.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Abaixo deste número médio de caracteres por página, consideramos o PDF
# "sem texto extraível" e disparamos OCR automaticamente.
_MIN_CHARS_PER_PAGE = 120


# ---------------------------------------------------------------------------
# Resultado normalizado do docling
# ---------------------------------------------------------------------------

@dataclass
class DocumentContent:
    full_text: str
    markdown: str
    tables: List[pd.DataFrame]
    page_count: int
    source_path: str
    ocr_used: bool = False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DoclingLoader:
    """
    Carrega PDFs com docling usando configuração otimizada para CPU.
    Mantém dois conversores prontos (sem OCR / com OCR) para evitar
    reconstruir o pipeline ML a cada chamada.
    """

    def __init__(
        self,
        ocr_mode: str | None = None,
        enable_table_structure: bool | None = None,
    ) -> None:
        # ocr_mode: "auto" (padrão) | "always" | "never"
        if ocr_mode is None:
            ocr_mode = os.getenv("DOCLING_OCR_MODE", "auto").lower()
        if ocr_mode not in ("auto", "always", "never"):
            ocr_mode = "auto"
        self._ocr_mode = ocr_mode

        if enable_table_structure is None:
            enable_table_structure = os.getenv("DOCLING_TABLE_STRUCTURE", "true").lower() == "true"

        ocr_lang = os.getenv("DOCLING_OCR_LANG", "por").split(",")

        logger.debug(
            f"DoclingLoader — ocr_mode={self._ocr_mode}, "
            f"table_structure={enable_table_structure}, ocr_lang={ocr_lang}"
        )

        # Conversor rápido (sem OCR) — usado sempre como primeira tentativa
        no_ocr_options = PdfPipelineOptions()
        no_ocr_options.do_ocr = False
        no_ocr_options.do_table_structure = enable_table_structure
        self._converter_no_ocr = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=no_ocr_options)}
        )

        # Conversor com OCR — usado quando necessário (lazy, só se ocr_mode != never)
        self._converter_ocr: DocumentConverter | None = None
        if self._ocr_mode != "never":
            ocr_options = PdfPipelineOptions()
            ocr_options.do_ocr = True
            ocr_options.ocr_options = TesseractCliOcrOptions(lang=ocr_lang)
            ocr_options.do_table_structure = enable_table_structure
            self._converter_ocr = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_options)}
            )

    def load(self, pdf_path: Path) -> DocumentContent:
        """
        Converte o PDF e retorna DocumentContent normalizado.
        Lança exceção se a conversão falhar (capturada no pipeline).
        """
        logger.debug(f"Carregando: {pdf_path.name}")

        if self._ocr_mode == "always" and self._converter_ocr is not None:
            return self._convert(pdf_path, self._converter_ocr, ocr_used=True)

        content = self._convert(pdf_path, self._converter_no_ocr, ocr_used=False)

        needs_ocr = (
            self._ocr_mode == "auto"
            and self._converter_ocr is not None
            and len(content.full_text) < _MIN_CHARS_PER_PAGE * max(content.page_count, 1)
        )
        if needs_ocr:
            logger.info(
                f"{pdf_path.name}: texto insuficiente ({len(content.full_text)} chars / "
                f"{content.page_count} pág) — reprocessando com OCR (Tesseract/por)..."
            )
            content = self._convert(pdf_path, self._converter_ocr, ocr_used=True)

        return content

    #------------------------------------------------------------------

    def _convert(self, pdf_path: Path, converter: DocumentConverter, ocr_used: bool) -> DocumentContent:
        result = converter.convert(str(pdf_path))
        document = result.document

        try:
            markdown = document.export_to_markdown()
        except Exception:
            markdown = ""

        full_text = self._extract_text(document, fallback=markdown)
        tables = self._extract_tables(document)
        page_count = self._page_count(document)

        logger.info(
            f"PDF carregado: {pdf_path.name} | "
            f"{page_count} pág(s) | {len(tables)} tabela(s) | {len(full_text)} chars | "
            f"ocr={'sim' if ocr_used else 'não'}"
        )

        return DocumentContent(
            full_text=full_text,
            markdown=markdown,
            tables=tables,
            page_count=page_count,
            source_path=str(pdf_path),
            ocr_used=ocr_used,
        )

    @staticmethod
    def _extract_text(document, fallback: str) -> str:
        """
        Extrai texto plano concatenando os items textuais do documento.
        Defensivo: aceita diferentes formatos de API do docling.
        """
        parts: list[str] = []

        try:
            for item in document.texts:
                text = getattr(item, "text", None)
                if text and isinstance(text, str):
                    parts.append(text)
        except (AttributeError, TypeError):
            pass

        if not parts:
            try:
                for item, _ in document.iterate_items():
                    text = getattr(item, "text", None)
                    if text and isinstance(text, str):
                        parts.append(text)
            except (AttributeError, TypeError):
                pass

        return "\n".join(parts) if parts else fallback

    @staticmethod
    def _extract_tables(document) -> list[pd.DataFrame]:
        tables: list[pd.DataFrame] = []
        try:
            for table in document.tables:
                try:
                    df = table.export_to_dataframe()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
                        if not df.empty:
                            tables.append(df)
                except Exception as exc:
                    logger.warning(f"Tabela ignorada: {exc}")
        except (AttributeError, TypeError):
            pass
        return tables

    @staticmethod
    def _page_count(document) -> int:
        try:
            return len(document.pages)
        except (AttributeError, TypeError):
            return 1

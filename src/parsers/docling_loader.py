"""
Wrapper sobre a biblioteca docling para carregar PDFs e retornar uma
representação enriquecida com a estrutura semântica completa do documento.

Além do texto plano e tabelas (mantidos para compatibilidade), extrai:
  - section_map  : {NOME_SEÇÃO → texto_do_bloco} via SectionHeaderItem nativo
  - kv_pairs     : [(chave, valor)] detectados pelo modelo de layout do Docling
  - page_texts   : {número_página → texto} para escopo de extração regional
  - spatial_index: elementos com bounding boxes para busca por proximidade visual

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pandas as pd
from loguru import logger

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Imports opcionais do docling-core (tipagem rica de elementos do documento).
# Presentes em docling>=2.x via pacote docling-core, mas protegidos por try/except
# para compatibilidade defensiva com versões que reorganizem os módulos.
try:
    from docling_core.types.doc import SectionHeaderItem, KeyValueItem
    from docling_core.types.doc.document import ContentLayer
    _RICH_TYPES = True
except ImportError:
    _RICH_TYPES = False
    logger.debug(
        "docling_core tipos ricos não disponíveis — "
        "section_map e kv_pairs ficam vazios; extração via texto plano segue normalmente."
    )

# Abaixo deste número médio de caracteres por página, consideramos o PDF
# "sem texto extraível" e disparamos OCR automaticamente.
_MIN_CHARS_PER_PAGE = 120


# ---------------------------------------------------------------------------
# Tipos de dados exportados
# ---------------------------------------------------------------------------

@dataclass
class SpatialElement:
    """Elemento de texto com sua posição visual (bounding box) na página."""
    text: str
    label: str
    page_no: int
    bbox_l: float
    bbox_t: float
    bbox_r: float
    bbox_b: float


@dataclass
class DocumentContent:
    """
    Representação normalizada e enriquecida de um PDF convertido pelo Docling.

    Campos de compatibilidade (sempre preenchidos):
      full_text    — texto plano concatenado (toda a fatura, todas as páginas)
      markdown     — export Markdown do Docling (estruturado, inclui tabelas formatadas)
      tables       — DataFrames das tabelas detectadas
      page_count   — número de páginas
      source_path  — caminho absoluto do PDF original
      ocr_used     — True se OCR (Tesseract) foi aplicado

    Campos de estrutura semântica (novos, podem ficar vazios em casos extremos):
      section_map    — {NOME_SEÇÃO (uppercase) → texto_do_bloco}
                       Permite extração scoped: busca "CNPJ" só na seção "EMITENTE".
      kv_pairs       — [(chave, valor)] detectados nativamente pelo modelo de layout.
                       Campos como "CNPJ: 12.345.678/0001-99" viram pares diretos.
      page_texts     — {número_página → texto}: escopo por região do documento.
                       Útil para buscar access_key só na pág. 1, código de barras
                       só na última.
      spatial_index  — lista de SpatialElement com bounding boxes.
                       Permite busca por proximidade visual (valor à direita de rótulo).
    """
    full_text: str
    markdown: str
    tables: List[pd.DataFrame]
    page_count: int
    source_path: str
    ocr_used: bool = False
    section_map: Dict[str, str] = field(default_factory=dict)
    kv_pairs: List[Tuple[str, str]] = field(default_factory=list)
    page_texts: Dict[int, str] = field(default_factory=dict)
    spatial_index: List[SpatialElement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DoclingLoader:
    """
    Carrega PDFs com Docling e retorna DocumentContent enriquecido.

    Mantém dois conversores prontos (sem OCR / com OCR) para evitar
    reconstruir o pipeline ML a cada chamada. Suporta modo batch via
    load_batch(), que usa convert_all() com paralelismo configurável.
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

        # Conversor com OCR — instanciado apenas se ocr_mode != "never"
        self._converter_ocr: DocumentConverter | None = None
        if self._ocr_mode != "never":
            ocr_options = PdfPipelineOptions()
            ocr_options.do_ocr = True
            ocr_options.ocr_options = TesseractCliOcrOptions(lang=ocr_lang)
            ocr_options.do_table_structure = enable_table_structure
            self._converter_ocr = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_options)}
            )

    # ------------------------------------------------------------------
    # API pública — single file
    # ------------------------------------------------------------------

    def load(self, pdf_path: Path) -> DocumentContent:
        """
        Converte um único PDF e retorna DocumentContent enriquecido.
        Lança exceção se a conversão falhar (capturada no pipeline).
        """
        logger.debug(f"Carregando: {pdf_path.name}")

        if self._ocr_mode == "always" and self._converter_ocr is not None:
            return self._convert_result(
                self._converter_no_ocr.convert(str(pdf_path)),
                pdf_path, ocr_used=False, force_ocr=True,
            )

        result_no_ocr = self._converter_no_ocr.convert(str(pdf_path))
        content = self._enrich(result_no_ocr, ocr_used=False)

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
            result_ocr = self._converter_ocr.convert(str(pdf_path))
            content = self._enrich(result_ocr, ocr_used=True)

        return content

    # ------------------------------------------------------------------
    # API pública — batch (paralelismo via convert_all)
    # ------------------------------------------------------------------

    def load_batch(self, pdf_paths: list[Path]) -> Iterator[tuple[Path, DocumentContent]]:
        """
        Converte múltiplos PDFs em batch usando convert_all() com paralelismo nativo.

        Estratégia de dois passes (preserva a lógica auto-OCR):
          1. Converte todos sem OCR em paralelo.
          2. Identifica quais têm texto insuficiente e reconverte com OCR (também em paralelo).
          3. Yielda (pdf_path, DocumentContent) conforme cada resultado fica pronto.

        O grau de paralelismo é controlado via DOCLING_BATCH_CONCURRENCY (padrão: 2).
        """
        if not pdf_paths:
            return

        _apply_batch_concurrency()

        # --- Passe 1: sem OCR (todos) ---
        no_ocr_map: dict[str, tuple[Path, object]] = {}
        for conv_result in self._converter_no_ocr.convert_all(
            [str(p) for p in pdf_paths], raises_on_error=False
        ):
            try:
                file_name = Path(conv_result.input.file).name
                original_path = next(p for p in pdf_paths if p.name == file_name)
                no_ocr_map[file_name] = (original_path, conv_result)
            except Exception as exc:
                logger.warning(f"Resultado de batch não mapeado: {exc}")

        # --- Identifica quais precisam de OCR ---
        needs_ocr_paths: list[Path] = []
        if self._ocr_mode != "never" and self._converter_ocr is not None:
            for file_name, (pdf_path, conv_result) in no_ocr_map.items():
                if conv_result.status != ConversionStatus.SUCCESS:
                    continue
                try:
                    text_len = sum(
                        len(getattr(item, "text", "") or "")
                        for item in conv_result.document.texts
                    )
                    page_count = max(len(conv_result.document.pages), 1)
                    if text_len < _MIN_CHARS_PER_PAGE * page_count:
                        needs_ocr_paths.append(pdf_path)
                except Exception:
                    pass

        # --- Passe 2: com OCR (somente os que precisam) ---
        ocr_map: dict[str, object] = {}
        if needs_ocr_paths:
            logger.info(
                f"Batch OCR: {len(needs_ocr_paths)}/{len(pdf_paths)} arquivo(s) "
                f"precisam de OCR — reprocessando..."
            )
            for conv_result in self._converter_ocr.convert_all(
                [str(p) for p in needs_ocr_paths], raises_on_error=False
            ):
                try:
                    file_name = Path(conv_result.input.file).name
                    ocr_map[file_name] = conv_result
                except Exception:
                    pass

        # --- Yielda resultados na ordem original ---
        for pdf_path in pdf_paths:
            file_name = pdf_path.name
            try:
                if file_name in ocr_map:
                    conv_result = ocr_map[file_name]
                    content = self._enrich(conv_result, ocr_used=True)
                elif file_name in no_ocr_map:
                    _, conv_result = no_ocr_map[file_name]
                    content = self._enrich(conv_result, ocr_used=False)
                else:
                    raise RuntimeError(f"Nenhum resultado de conversão para {file_name}")
                yield pdf_path, content
            except Exception as exc:
                logger.error(f"Erro ao enriquecer {file_name}: {exc}")
                raise

    # ------------------------------------------------------------------
    # Construção do DocumentContent a partir de ConversionResult
    # ------------------------------------------------------------------

    def _convert_result(self, result, pdf_path: Path, ocr_used: bool, force_ocr: bool = False) -> DocumentContent:
        """
        Converte ConversionResult (single-file, modo always) respeitando force_ocr.
        """
        if force_ocr and self._converter_ocr is not None:
            result = self._converter_ocr.convert(str(pdf_path))
            ocr_used = True
        return self._enrich(result, ocr_used=ocr_used)

    def _enrich(self, conv_result, ocr_used: bool) -> DocumentContent:
        """
        Transforma um ConversionResult do Docling em DocumentContent enriquecido.
        Extrai texto plano, markdown, tabelas E os novos campos estruturais.
        """
        document = conv_result.document

        try:
            markdown = document.export_to_markdown()
        except Exception:
            markdown = ""

        full_text = self._extract_text(document, fallback=markdown)
        tables = self._extract_tables(document)
        page_count = self._page_count(document)

        section_map = self._build_section_map(document)
        kv_pairs = self._extract_kv_pairs(document)
        page_texts = self._build_page_texts(document)
        spatial_index = self._build_spatial_index(document)

        source_path = ""
        try:
            source_path = str(conv_result.input.file)
        except Exception:
            pass

        logger.info(
            f"PDF carregado: {Path(source_path).name} | "
            f"{page_count} pág(s) | {len(tables)} tabela(s) | {len(full_text)} chars | "
            f"ocr={'sim' if ocr_used else 'não'} | "
            f"seções={len(section_map)} | kv={len(kv_pairs)} | "
            f"spatial={len(spatial_index)}"
        )

        return DocumentContent(
            full_text=full_text,
            markdown=markdown,
            tables=tables,
            page_count=page_count,
            source_path=source_path,
            ocr_used=ocr_used,
            section_map=section_map,
            kv_pairs=kv_pairs,
            page_texts=page_texts,
            spatial_index=spatial_index,
        )

    # ------------------------------------------------------------------
    # Extratores de campos individuais (estáticos, sem estado)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(document, fallback: str) -> str:
        """
        Extrai texto plano concatenando os itens textuais do documento.
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

    @staticmethod
    def _build_section_map(document) -> Dict[str, str]:
        """
        Constrói {NOME_SEÇÃO → texto_do_bloco} usando SectionHeaderItem nativo do Docling.

        Itera todos os itens em ordem de leitura (incluindo furniture — cabeçalhos/rodapés)
        e agrupa o texto entre seções consecutivas. Quando o mesmo nome de seção aparece
        mais de uma vez (ex: "EMITENTE" em frente/verso), concatena os textos.

        Retorna dict vazio se o Docling não detectar seções ou se os tipos ricos
        não estiverem disponíveis (import defensivo).
        """
        if not _RICH_TYPES:
            return {}

        section_map: Dict[str, str] = {}
        current_section = "__root__"
        current_parts: list[str] = []

        def _flush(section: str, parts: list[str]) -> None:
            if not parts:
                return
            block = "\n".join(parts)
            existing = section_map.get(section)
            section_map[section] = (existing + "\n" + block).strip() if existing else block

        try:
            items_iter = document.iterate_items(
                included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
            )
            for item, _level in items_iter:
                if isinstance(item, SectionHeaderItem):
                    _flush(current_section, current_parts)
                    current_section = item.text.strip().upper()
                    current_parts = []
                else:
                    text = getattr(item, "text", None)
                    if text and isinstance(text, str) and text.strip():
                        current_parts.append(text.strip())
        except Exception as exc:
            logger.debug(f"_build_section_map: {exc}")

        _flush(current_section, current_parts)
        return section_map

    @staticmethod
    def _extract_kv_pairs(document) -> List[Tuple[str, str]]:
        """
        Extrai pares (chave, valor) detectados nativamente pelo modelo de layout
        do Docling via document.key_value_items.

        Defensivo: tenta múltiplos formatos de acesso (GraphData body/cells e
        atributos diretos key/value) para cobrir variações de versão do docling-core.
        """
        pairs: List[Tuple[str, str]] = []
        try:
            for kv_item in document.key_value_items:
                try:
                    key_text: str | None = None
                    val_text: str | None = None

                    # Formato GraphData: body.cells[0] = chave, body.cells[1] = valor
                    body = getattr(kv_item, "body", None)
                    if body is not None:
                        cells = getattr(body, "cells", None) or []
                        if len(cells) >= 2:
                            key_text = getattr(cells[0], "text", None)
                            val_text = getattr(cells[1], "text", None)

                    # Formato direto: atributos key e value como objetos com .text
                    if not key_text:
                        key_obj = getattr(kv_item, "key", None)
                        key_text = getattr(key_obj, "text", None) if key_obj else None
                    if not val_text:
                        val_obj = getattr(kv_item, "value", None)
                        val_text = getattr(val_obj, "text", None) if val_obj else None

                    if key_text and val_text:
                        pairs.append((str(key_text).strip(), str(val_text).strip()))
                except Exception:
                    pass
        except (AttributeError, TypeError):
            pass
        return pairs

    @staticmethod
    def _build_page_texts(document) -> Dict[int, str]:
        """
        Segrega o texto por página usando a informação de provenance de cada item.
        Permite extração scoped: buscar access_key só na pág. 1, código de barras
        só na última página, métricas de energia na pág. 2, etc.
        """
        page_parts: Dict[int, list[str]] = {}
        try:
            for item in document.texts:
                text = getattr(item, "text", None)
                if not text or not isinstance(text, str) or not text.strip():
                    continue
                prov_list = getattr(item, "prov", [])
                page_no = getattr(prov_list[0], "page_no", 1) if prov_list else 1
                page_parts.setdefault(page_no, []).append(text.strip())
        except (AttributeError, TypeError) as exc:
            logger.debug(f"_build_page_texts: {exc}")

        return {pg: "\n".join(parts) for pg, parts in sorted(page_parts.items())}

    @staticmethod
    def _build_spatial_index(document) -> List[SpatialElement]:
        """
        Constrói índice espacial de todos os elementos textuais com bounding boxes.

        Usado pelo DoclingExtractor para busca por proximidade visual:
        encontrar o valor monetário imediatamente à direita de "TOTAL A PAGAR",
        ou a data abaixo de "VENCIMENTO" — sem depender de texto vizinho na string.

        Substitui hacks posicionais como o "MTE\\d{7}" anchor no parser determinístico.
        """
        elements: List[SpatialElement] = []
        try:
            for item in document.texts:
                text = getattr(item, "text", None)
                if not text or not isinstance(text, str) or not text.strip():
                    continue
                label = type(item).__name__.replace("Item", "").lower()
                prov_list = getattr(item, "prov", [])
                for prov in prov_list:
                    bbox = getattr(prov, "bbox", None)
                    if bbox is None:
                        continue
                    page_no = getattr(prov, "page_no", 1)
                    elements.append(SpatialElement(
                        text=text.strip(),
                        label=label,
                        page_no=page_no,
                        bbox_l=float(getattr(bbox, "l", 0.0)),
                        bbox_t=float(getattr(bbox, "t", 0.0)),
                        bbox_r=float(getattr(bbox, "r", 0.0)),
                        bbox_b=float(getattr(bbox, "b", 0.0)),
                    ))
        except (AttributeError, TypeError) as exc:
            logger.debug(f"_build_spatial_index: {exc}")
        return elements


# ---------------------------------------------------------------------------
# Configuração de paralelismo de batch
# ---------------------------------------------------------------------------

def _apply_batch_concurrency() -> None:
    """
    Aplica DOCLING_BATCH_CONCURRENCY às settings globais do Docling.
    Chamado antes de convert_all() no modo batch.
    """
    concurrency = int(os.getenv("DOCLING_BATCH_CONCURRENCY", "2"))
    try:
        from docling.datamodel.settings import settings as docling_settings
        docling_settings.perf.doc_batch_concurrency = concurrency
        docling_settings.perf.doc_batch_size = concurrency
        logger.debug(f"Docling batch_concurrency={concurrency}")
    except Exception as exc:
        logger.debug(f"Não foi possível configurar batch_concurrency: {exc}")

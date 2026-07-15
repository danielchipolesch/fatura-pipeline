"""
Orquestrador do pipeline de extração de faturas.

Fluxo por arquivo (3 camadas de extração):
  1. DoclingLoader carrega o PDF → DocumentContent enriquecido
  2. classify_layout() detecta o tipo de fatura
  3. DoclingExtractor (PRIMÁRIO) — usa estrutura semântica nativa do Docling
       (section_map, kv_pairs, spatial_index, page_texts)
  4. DeterministicParser (FALLBACK) — preenche o que o DoclingExtractor não encontrou
       via regex e heurísticas sobre o texto plano
  5. Confidence score é calculado
  6. Se score < CONFIDENCE_THRESHOLD e houver campos faltantes:
       LLMFallback.enrich() preenche o que falta (síncrono)
  7. Invoice validado pelo Pydantic é retornado

Modo batch (process_batch):
  Agrupa múltiplos PDFs e usa convert_all() com paralelismo nativo do Docling
  para a etapa de conversão. A extração (DoclingExtractor → DeterministicParser →
  LLM) permanece sequencial por arquivo para não sobrecarregar a CPU.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from loguru import logger

from src.extractors.docling_extractor import DoclingExtractor
from src.extractors.layout import classify_layout
from src.extractors.tipo_fatura import infer_tipo_fatura_operacional
from src.models.invoice import Invoice, ParsingMethod
from src.parsers.deterministic import (
    DeterministicParser,
    calculate_confidence,
    get_missing_required_fields,
)
from src.parsers.docling_loader import DoclingLoader
from src.parsers.llm_fallback import LLMFallback
from src.parsers.scanned_preprocessor import ScannedPreprocessor, is_available as _paddle_available

_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


class FaturaPipeline:
    """
    Pipeline reutilizável: crie uma instância e chame process() ou process_batch()
    para cada arquivo/lote. Todos os componentes são inicializados uma única vez.
    """

    def __init__(self) -> None:
        self._loader = DoclingLoader()
        self._docling_extractor = DoclingExtractor()
        self._deterministic = DeterministicParser()
        self._llm = LLMFallback()
        self._scanned_preprocessor = ScannedPreprocessor()

        logger.info(
            f"Pipeline inicializado — "
            f"threshold={_CONFIDENCE_THRESHOLD:.2f} | "
            f"llm_fallback={'habilitado' if self._llm.enabled else 'desabilitado'} | "
            f"paddleocr={'habilitado' if _paddle_available() else 'indisponível'}"
        )

    # ------------------------------------------------------------------
    # API pública — arquivo único
    # ------------------------------------------------------------------

    def process(self, pdf_path: Path) -> Invoice:
        """
        Processa um único PDF e retorna o Invoice padronizado.
        Pode lançar exceção se o PDF for ilegível — trate no chamador.
        """
        logger.info(f"[INÍCIO] {pdf_path.name}")

        content = self._loader.load(pdf_path)
        invoice = self._extract(content, pdf_path)

        logger.info(
            f"[FIM] {pdf_path.name} — "
            f"method={invoice.parsing_method.value} | "
            f"confidence={invoice.confidence_score:.2f}"
        )
        return invoice

    # ------------------------------------------------------------------
    # API pública — batch
    # ------------------------------------------------------------------

    def process_batch(self, pdf_paths: list[Path]) -> Iterator[tuple[Path, Invoice]]:
        """
        Processa múltiplos PDFs em batch.

        A conversão PDF→DoclingDocument ocorre em paralelo (convert_all com
        doc_batch_concurrency configurável via DOCLING_BATCH_CONCURRENCY).
        A extração de campos permanece sequencial por arquivo.

        Yielda (pdf_path, invoice) conforme cada arquivo é processado.
        Lança exceção por arquivo — o chamador decide se continua ou para.
        """
        if not pdf_paths:
            return

        logger.info(f"[BATCH] {len(pdf_paths)} arquivo(s) para converter")

        for pdf_path, content in self._loader.load_batch(pdf_paths):
            logger.info(f"[INÍCIO] {pdf_path.name}")
            try:
                invoice = self._extract(content, pdf_path)
                logger.info(
                    f"[FIM] {pdf_path.name} — "
                    f"method={invoice.parsing_method.value} | "
                    f"confidence={invoice.confidence_score:.2f}"
                )
                yield pdf_path, invoice
            except Exception as exc:
                logger.error(f"Erro na extração de {pdf_path.name}: {exc}")
                raise

    # ------------------------------------------------------------------
    # Extração (comum a process() e process_batch())
    # ------------------------------------------------------------------

    def _extract(self, content, pdf_path: Path) -> Invoice:
        """
        Executa as 3 camadas de extração sobre um DocumentContent já carregado.
        """
        # 1. Classificar layout
        layout = classify_layout(content.full_text)
        logger.info(f"Layout detectado: {layout.value}")

        # 2. DoclingExtractor — camada primária (estrutura semântica nativa)
        prefilled = self._docling_extractor.extract(content, layout)

        # 2.5. ScannedPreprocessor — para documentos escaneados, substitui o
        #      full_text e page_texts do Docling por texto extraído via
        #      OpenCV + PaddleOCR, de qualidade superior para text plano.
        #      Roda ANTES do DeterministicParser para que o regex opere sobre
        #      texto limpo. section_map/kv_pairs/spatial_index do Docling são
        #      preservados pois vêm da estrutura semântica (não do texto plano).
        if content.ocr_used and _paddle_available():
            try:
                paddle_text, paddle_pages = self._scanned_preprocessor.extract(pdf_path)
                if paddle_text:
                    content.full_text = paddle_text
                    content.page_texts = paddle_pages
                    logger.info(
                        f"ScannedPreprocessor: full_text atualizado para {pdf_path.name} "
                        f"({len(paddle_text)} chars)"
                    )
            except Exception as _paddle_exc:
                logger.warning(
                    f"ScannedPreprocessor: erro não-fatal em {pdf_path.name}: {_paddle_exc} — "
                    f"continuando com texto do Docling OCR"
                )

        # 3. DeterministicParser — fallback (regex sobre texto plano)
        invoice = self._deterministic.parse(content, layout, prefilled=prefilled)

        # Método reflete qual camada foi a fonte principal dos campos extraídos.
        # Se o DoclingExtractor contribuiu com pelo menos um campo, a leitura
        # usou a estrutura semântica nativa do documento (KV pairs, seções,
        # bounding boxes) — fonte mais confiável que regex sobre texto plano.
        if prefilled:
            invoice.parsing_method = ParsingMethod.SEMANTIC
        # else: permanece DETERMINISTIC (gravado em DeterministicParser.parse())

        # 4. Inferência do tipo operacional de fatura de energia
        invoice.tipo_fatura_operacional = infer_tipo_fatura_operacional(
            text=content.full_text,
            layout=layout,
            supplier_cnpj=invoice.supplier.cnpj if invoice.supplier else None,
            line_items=invoice.line_items,
        )

        # Pós-processamento: boleto com chave de acesso NF-e/NF3E embarcada
        # (44 dígitos, modelo 55/66) permite inferir o tipo pelo CNPJ do emitente.
        if invoice.tipo_fatura_operacional is None and invoice.bank_slip_barcode:
            bc = invoice.bank_slip_barcode
            if len(bc) == 44 and bc.isdigit() and bc[20:22] in ("55", "66"):
                from src.extractors.known_entities import lookup_supplier as _ls
                from src.models.invoice import TipoFaturaOperacional as _TFO
                _info = _ls(bc[6:20])
                if _info:
                    _role = _info.get("role")
                    if _role == "distribuidora":
                        invoice.tipo_fatura_operacional = _TFO.DISTRIBUIDORA_MLE
                    elif _role == "comercializadora":
                        invoice.tipo_fatura_operacional = _TFO.COMERCIALIZADORA_MLE

        # Documentos escaneados (OCR): o parser determinístico operou sobre texto
        # OCR degradado — resultados são não-confiáveis e devem ser enriquecidos
        # pelo LLM. Força confidence=0 para garantir que o fallback sempre rode.
        if content.ocr_used:
            invoice.confidence_score = 0.0
            invoice.confidence_score_initial = 0.0

        logger.info(
            f"Extração — "
            f"method={invoice.parsing_method.value} | "
            f"tipo={invoice.tipo_fatura_operacional.value if invoice.tipo_fatura_operacional else 'N/A'} | "
            f"docling_fields={len(prefilled)} | "
            f"ocr={content.ocr_used} | "
            f"confidence={invoice.confidence_score:.2f} | "
            f"número={invoice.invoice_number} | "
            f"total={invoice.total}"
        )

        # 5. LLM fallback (somente se necessário)
        if invoice.confidence_score < _CONFIDENCE_THRESHOLD:
            missing = get_missing_required_fields(invoice)

            # Documentos OCR: campos podem estar preenchidos com texto degradado.
            # Força reextração de todos os campos essenciais pelo LLM.
            if content.ocr_used and self._llm.enabled:
                _ocr_reextract = [
                    "invoice_number — número da fatura/nota fiscal",
                    "supplier_name — nome/razão social do emitente",
                    "supplier_cnpj — CNPJ do emitente",
                    "customer_name — nome/razão social do destinatário",
                    "customer_cnpj — CNPJ do destinatário",
                    "total — valor total da fatura",
                    "issue_date — data de emissão (formato YYYY-MM-DD)",
                    "due_date — data de vencimento (formato YYYY-MM-DD)",
                    "consumer_unit — código da unidade consumidora (instalação/UC)",
                    "line_items — componentes/itens da fatura (TUSD, TE, demanda, impostos, etc.)",
                ]
                missing = list({*missing, *_ocr_reextract})

            if missing and self._llm.enabled:
                invoice = self._llm.enrich(invoice, missing, content=content)
                invoice.confidence_score = calculate_confidence(invoice)
                logger.info(
                    f"Após LLM fallback — "
                    f"número={invoice.invoice_number} | "
                    f"total={invoice.total} | "
                    f"confidence={invoice.confidence_score:.2f}"
                )
            elif missing:
                logger.warning(
                    f"Confidence baixo ({invoice.confidence_score:.2f}) e LLM fallback "
                    f"desabilitado — campos ausentes: "
                    f"{[f.split(' — ')[0] for f in missing]}"
                )

        return invoice

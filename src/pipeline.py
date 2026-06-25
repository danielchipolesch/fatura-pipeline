"""
Orquestrador do pipeline de extração de faturas.

Fluxo por arquivo:
  1. DoclingLoader carrega o PDF → DocumentContent
  2. classify_layout() detecta o tipo de fatura
  3. DeterministicParser extrai todos os campos
  4. Confidence score é calculado
  5. Se score < CONFIDENCE_THRESHOLD e houver campos faltantes:
       LLMFallback.enrich() preenche o que falta (síncrono)
  6. Invoice validado pelo Pydantic é retornado
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from src.extractors.layout import classify_layout
from src.models.invoice import Invoice, ParsingMethod
from src.parsers.deterministic import (
    DeterministicParser,
    calculate_confidence,
    get_missing_required_fields,
)
from src.parsers.docling_loader import DoclingLoader
from src.parsers.llm_fallback import LLMFallback

_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


class FaturaPipeline:
    """
    Pipeline reutilizável: crie uma instância e chame process() para cada arquivo.
    O DoclingLoader e o LLMFallback são inicializados uma única vez.
    """

    def __init__(self) -> None:
        self._loader = DoclingLoader()
        self._deterministic = DeterministicParser()
        self._llm = LLMFallback()

        logger.info(
            f"Pipeline inicializado — "
            f"threshold={_CONFIDENCE_THRESHOLD:.2f} | "
            f"llm_fallback={'habilitado' if self._llm.enabled else 'desabilitado'}"
        )

    def process(self, pdf_path: Path) -> Invoice:
        """
        Processa um único PDF e retorna o Invoice padronizado.
        Pode lançar exceção se o PDF for ilegível — trate no chamador.
        """
        logger.info(f"[INÍCIO] {pdf_path.name}")

        # 1. Carregar PDF com docling
        content = self._loader.load(pdf_path)

        # 2. Classificar layout
        layout = classify_layout(content.full_text)
        logger.info(f"Layout detectado: {layout.value}")

        # 3. Extração determinística
        invoice = self._deterministic.parse(content, layout)

        logger.info(
            f"Extração determinística — "
            f"confidence={invoice.confidence_score:.2f} | "
            f"número={invoice.invoice_number} | "
            f"total={invoice.total}"
        )

        # 4. LLM fallback (somente se necessário)
        if invoice.confidence_score < _CONFIDENCE_THRESHOLD:
            missing = get_missing_required_fields(invoice)
            if missing and self._llm.enabled:
                invoice = self._llm.enrich(invoice, missing)
                # Recalcula o score: o valor calculado antes do enrich() fica
                # obsoleto assim que o LLM preenche novos campos — sem isso, o
                # JSON final mostraria o confidence_score "congelado" de antes
                # do fallback, mesmo já tendo dados adicionais preenchidos.
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

        logger.info(
            f"[FIM] {pdf_path.name} — "
            f"method={invoice.parsing_method.value} | "
            f"confidence={invoice.confidence_score:.2f}"
        )
        return invoice

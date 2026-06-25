"""
Classificador de layout de faturas.
Analisa o texto extraído pelo docling e determina o tipo de documento.
A classificação é puramente determinística (baseada em palavras-chave e padrões).
"""
from __future__ import annotations

import re
from typing import Sequence

from src.models.invoice import InvoiceLayout

# ---------------------------------------------------------------------------
# Padrões por tipo de documento (todos case-insensitive via flag)
# ---------------------------------------------------------------------------

_NFE_PATTERNS: Sequence[str] = [
    r"NOTA\s+FISCAL\s+ELETR[ÔO]NICA",
    r"\bDANFE\b",
    r"NF\-?e\b",
    r"\bCFOP\b",
    r"CHAVE\s+(?:DE\s+)?ACESSO",
    r"PROTOCOLO\s+DE\s+AUTORIZA[ÇC][ÃA]O",
    r"NATUREZA\s+DA\s+OPERA[ÇC][ÃA]O",
]

_NFSE_PATTERNS: Sequence[str] = [
    r"NOTA\s+FISCAL\s+(?:ELETR[ÔO]NICA\s+)?DE\s+SERVI[ÇC]OS",
    r"\bNFS\-?e\b",
    r"PRESTADOR\s+DE\s+SERVI[ÇC]OS",
    r"TOMADOR\s+DE\s+SERVI[ÇC]OS",
    r"IMPOSTO\s+SOBRE\s+SERVI[ÇC]OS",
    r"\bISSQN\b",
    r"C[OÓ]DIGO\s+DO\s+SERVI[ÇC]O",
]

_BOLETO_PATTERNS: Sequence[str] = [
    r"BOLETO\s+BANC[AÁ]RIO",
    r"\bBLOQUETO\b",
    r"LINHA\s+DIGIT[AÁ]VEL",
    r"NOSSO\s+N[ÚU]MERO",
    r"BENEFICI[AÁ]RIO",
    r"C[OÓ]DIGO\s+DO\s+BANCO",
    r"\bSACADO\b",
    r"\bCEDENTE\b",
    r"DATA\s+(?:DE\s+)?VENCIMENTO",
    r"AGÊNCIA\s*/\s*C[OÓ]DIGO",
]

_UTILITY_PATTERNS: Sequence[str] = [
    r"CONTA\s+DE\s+(?:ENERGIA|[AÁ]GUA|G[AÁ]S|LUZ|ESGOTO)",
    r"LEITURA\s+ANTERIOR",
    r"CONSUMO\s+(?:EM\s+)?(?:KWH|M3|M³)",
    r"TARIFA\s+DE\s+(?:ENERGIA|DISTRIBUI[ÇC][ÃA]O)",
    r"BANDEIRA\s+TARIFÁRIA",
    r"FATURA\s+DE\s+(?:ENERGIA|[AÁ]GUA|G[AÁ]S)",
    # Concessionárias conhecidas
    r"\b(?:CEMIG|COPEL|CPFL|ENEL|LIGHT|COELBA|CELPE|CELESC|ELEKTRO|ENERGISA|SABESP|SANEPAR|CEDAE|COPASA)\b",
]


def _score(text: str, patterns: Sequence[str]) -> int:
    """Retorna o número de padrões que casam no texto."""
    count = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return count


_CFOP_RE = re.compile(r"\bCFOP\b", re.IGNORECASE)


def classify_layout(text: str) -> InvoiceLayout:
    """
    Classifica o tipo de fatura com base em palavras-chave no texto.

    Prioridade de desambiguação:
      NF-e > NFS-e > Boleto > Concessionária > Genérico

    Retorna o layout com maior score, desde que ≥ 1 padrão tenha casado.
    """
    scores = {
        InvoiceLayout.NFE: _score(text, _NFE_PATTERNS),
        InvoiceLayout.NFSE: _score(text, _NFSE_PATTERNS),
        InvoiceLayout.BOLETO: _score(text, _BOLETO_PATTERNS),
        InvoiceLayout.UTILITY: _score(text, _UTILITY_PATTERNS),
    }

    # CFOP é um código tributário que praticamente só aparece em NF-e/NFS-e —
    # faturas de concessionária no formato DANFE (ex: Enel, CEMIG) frequentemente
    # também citam o nome da empresa e termos como "conta de energia", o que
    # inflaria o score de UTILITY/BOLETO acima do de NFE sem essa regra. CFOP
    # presente e nenhum sinal de NFS-e é um indicador praticamente definitivo
    # de que o documento é uma NF-e, mesmo que outros scores sejam maiores.
    if _CFOP_RE.search(text) and scores[InvoiceLayout.NFSE] == 0:
        return InvoiceLayout.NFE

    best_layout = max(scores, key=lambda k: scores[k])
    if scores[best_layout] == 0:
        return InvoiceLayout.GENERIC

    # Desempate: NF-e tem precedência sobre NFS-e quando scores iguais
    if scores[InvoiceLayout.NFE] == scores[InvoiceLayout.NFSE] and scores[InvoiceLayout.NFE] > 0:
        return InvoiceLayout.NFE

    return best_layout

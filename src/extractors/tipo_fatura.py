"""
Inferência do tipo operacional de fatura de energia elétrica.

O tipo operacional NÃO está escrito explicitamente no documento — é derivado
por um conjunto de regras que analisam sinais textuais, layout e fornecedor.
Isso importa para o dashboard porque uma UC em Mercado Livre gera DUAS faturas
por competência (distribuidora + comercializadora). Somar ambas como "despesa"
duplicaria o custo de energia — o dashboard usa `tipo_fatura_operacional` para
consolidar corretamente por UC/competência.

Ordem de avaliação (do mais específico para o mais genérico):
  1. COMERCIALIZADORA_MLE — sinais mais fortes: MWh como unidade, CNPJ de
     comercializadora conhecida, descritivos de energia ACL/contratada/incentivada
     sem TUSD/demanda.
  2. DISTRIBUIDORA_MLE — sinais inequívocos de MLE com layout de distribuidora:
     abatimento "Energia Terc. Comercializada" ou "Energia ACL", "Componente Fio",
     colunas TUSD-Ponta + TUSD-Fora Ponta simultâneas.
  3. CATIVO — padrão quando é fatura de concessionária (layout UTILITY) sem sinais MLE.
  4. None — fatura não identificada como de energia (NF-e genérica, boleto, etc.).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.models.invoice import InvoiceLayout, LineItem

from src.extractors.known_entities import lookup_supplier
from src.models.invoice import InvoiceLayout, TipoFaturaOperacional


# ---------------------------------------------------------------------------
# Padrões compilados (compilados uma vez ao importar o módulo)
# ---------------------------------------------------------------------------

# COMERCIALIZADORA_MLE
_RE_MWH = re.compile(r"\bMWH\b", re.IGNORECASE)
_RE_ENERGIA_CONTRATADA = re.compile(
    r"ENERGIA\s+(CONTRATADA|INCENTIVADA|ACL|LIVRE)\b", re.IGNORECASE
)

# DISTRIBUIDORA_MLE
_RE_ENERGIA_TERC = re.compile(
    r"ENERGIA\s+TERC\.?\s+COMERCIALIZADA\b", re.IGNORECASE
)
_RE_COMPONENTE_FIO = re.compile(
    r"COMPONENTE\s+FIO\s*[AB]?\b", re.IGNORECASE
)
_RE_ENERGIA_ACL = re.compile(r"ENERGIA\s+ACL\b", re.IGNORECASE)
_RE_TUSD_PONTA = re.compile(r"TUSD[\s\-]+PONTA\b", re.IGNORECASE)
_RE_TUSD_FORA = re.compile(r"TUSD[\s\-]+(?:FORA|F\.?\s*PONTA)\b", re.IGNORECASE)


def infer_tipo_fatura_operacional(
    text: str,
    layout: "InvoiceLayout",
    supplier_cnpj: Optional[str],
    line_items: List["LineItem"],
) -> Optional[TipoFaturaOperacional]:
    """
    Infere o tipo operacional da fatura por regras determinísticas.

    Parâmetros:
      text          — texto completo extraído da fatura (full_text do DocumentContent)
      layout        — layout classificado (UTILITY, NFE, etc.)
      supplier_cnpj — CNPJ do emitente extraído/normalizado (pode ser None)
      line_items    — itens de linha da fatura (LineItem[])

    Retorna:
      TipoFaturaOperacional ou None (quando não é fatura de energia identificável)
    """
    text_upper = text.upper()

    # ------------------------------------------------------------------
    # 1. COMERCIALIZADORA_MLE
    #
    # Verificamos primeiro porque é o caso mais específico: uma NF-e de
    # comercializadora privada de energia tem características únicas que
    # não aparecem em faturas de distribuidoras. A detecção prematura
    # aqui evita falsos positivos de DISTRIBUIDORA_MLE.
    # ------------------------------------------------------------------
    if _is_comercializadora_mle(text_upper, layout, supplier_cnpj, line_items):
        return TipoFaturaOperacional.COMERCIALIZADORA_MLE

    # ------------------------------------------------------------------
    # 2. DISTRIBUIDORA_MLE
    #
    # Fatura de distribuidora, mas o cliente está no mercado livre:
    # cobra TUSD/uso da rede; a energia em si vem da comercializadora
    # e aparece como abatimento/dedução.
    # ------------------------------------------------------------------
    if _is_distribuidora_mle(text_upper, layout):
        return TipoFaturaOperacional.DISTRIBUIDORA_MLE

    # ------------------------------------------------------------------
    # 3. CATIVO
    #
    # Padrão para qualquer fatura de concessionária sem sinais MLE.
    # Inclui Grupo B (residencial/comercial pequeno) e Grupo A sem ACL.
    # ------------------------------------------------------------------
    if layout == InvoiceLayout.UTILITY:
        return TipoFaturaOperacional.CATIVO

    # ------------------------------------------------------------------
    # 4. None
    #
    # Não é uma fatura de energia reconhecível (NF-e de fornecedor
    # genérico, boleto avulso, etc.). O dashboard deve ignorar este
    # campo ou tratá-lo como não aplicável.
    # ------------------------------------------------------------------
    return None


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _is_comercializadora_mle(
    text_upper: str,
    layout: "InvoiceLayout",
    supplier_cnpj: Optional[str],
    line_items: List["LineItem"],
) -> bool:
    """
    Retorna True se a fatura é de uma comercializadora no Mercado Livre.

    Regras (qualquer uma é suficiente, por ordem de confiabilidade):
      a) CNPJ do emitente cadastrado com role="comercializadora"
      b) Energia faturada em MWh — unidade exclusiva de comercializadoras
         (distribuidoras usam kWh)
      c) Descritivo de item contém "ENERGIA CONTRATADA", "ENERGIA INCENTIVADA"
         ou "ENERGIA ACL" / "ENERGIA LIVRE" — termos do ACL
    """
    # a) Fornecedor cadastrado como comercializadora
    if supplier_cnpj:
        info = lookup_supplier(supplier_cnpj)
        if info and info.get("role") == "comercializadora":
            return True

    # b) MWh como unidade de faturamento (em qualquer item de linha)
    for item in line_items:
        unit = (item.unit or "").upper()
        if "MWH" in unit:
            return True
        desc = (item.description or "").upper()
        if _RE_MWH.search(desc):
            return True

    # c) MWh ou descritivos ACL no texto corrido
    if _RE_MWH.search(text_upper):
        # Confirmar que é fatura de energia (evitar MWh mencionado
        # incidentalmente em boleto ou NF-e não relacionada)
        if _RE_ENERGIA_CONTRATADA.search(text_upper):
            return True
        # MWh sozinho já é sinal forte em qualquer layout NFE/UTILITY
        if layout in (InvoiceLayout.NFE, InvoiceLayout.UTILITY):
            return True

    return False


def _is_distribuidora_mle(
    text_upper: str,
    layout: "InvoiceLayout",
) -> bool:
    """
    Retorna True se a fatura é de distribuidora com cliente em Mercado Livre.

    Regras (qualquer uma é suficiente, por ordem de especificidade):
      a) "ENERGIA TERC. COMERCIALIZADA" — termo CEMIG/Enel para energia
         fornecida por terceiro (comercializadora); aparece como abatimento.
      b) "COMPONENTE FIO A" ou "COMPONENTE FIO B" — encargo de uso de rede
         exclusivo de faturas ACL do sistema CEMIG/ANEEL.
      c) "ENERGIA ACL" em layout UTILITY — dedução da energia ACL da base
         de cobrança de TUSD.
      d) TUSD-Ponta E TUSD-Fora Ponta presentes simultaneamente — estrutura
         tarifária Grupo A típica de distribuidoras MLE (Horossazonal Verde/Azul).
    """
    # a) Abatimento "Energia Terc. Comercializada"
    if _RE_ENERGIA_TERC.search(text_upper):
        return True

    # b) Componente Fio A/B
    if _RE_COMPONENTE_FIO.search(text_upper):
        return True

    # c) "Energia ACL" em fatura de concessionária
    if layout == InvoiceLayout.UTILITY and _RE_ENERGIA_ACL.search(text_upper):
        return True

    # d) Estrutura TUSD-Ponta + TUSD-Fora Ponta (ambas presentes)
    if _RE_TUSD_PONTA.search(text_upper) and _RE_TUSD_FORA.search(text_upper):
        return True

    return False

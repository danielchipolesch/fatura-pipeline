"""
Extração semântica de impostos a partir da estrutura Docling.

Três estratégias em ordem de confiança decrescente:
  1. kv_pairs  — pares chave-valor com labels como "BASE DE CÁLCULO DO ICMS"
  2. tables    — DataFrames com linha ou coluna de tributos (ICMS, PIS, etc.)
  3. section   — texto de seção identificada semanticamente ("TRIBUTAÇÃO",
                 "BASE DE CÁLCULO", etc.) — escoped, não é regex global

Compartilhado entre DoclingExtractor (primário) e DeterministicParser
(que pode usar os resultados via pre.get("taxes")).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.extractors.fields import parse_currency
from src.models.invoice import Tax

# ---------------------------------------------------------------------------
# Mapa canônico de nomes de impostos (substring → nome canônico)
# ---------------------------------------------------------------------------

_TAX_CANONICAL: List[Tuple[str, str]] = [
    ("icms", "ICMS"),
    ("ipi", "IPI"),
    ("issqn", "ISS"),
    ("iss", "ISS"),
    ("pis/pasep", "PIS"),
    ("pis", "PIS"),
    ("cofins", "COFINS"),
    ("csll", "CSLL"),
    ("irpj", "IRPJ"),
    ("irrf", "IRRF"),
    ("inss", "INSS"),
]

_BASE_WORDS = {"base", "bc", "base de calculo", "base calc", "base do", "base icms"}
_RATE_WORDS = {"aliquota", "aliq", "taxa", "percentual", "%"}
_AMOUNT_WORDS = {"valor do", "valor da", "valor total", "montante", "valor icms", "vlr icms"}

# Seções que tipicamente contêm dados de impostos
SECTIONS_TAX = [
    "CÁLCULO DO IMPOSTO", "CALCULO DO IMPOSTO",   # seção padrão do DANFE NF-e
    "CÁLCULO DO ICMS", "CALCULO DO ICMS",
    "BASE DE CÁLCULO", "BASE DE CALCULO",
    "TRIBUTAÇÃO", "TRIBUTACAO",
    "TRIBUTOS", "IMPOSTOS",
    "DADOS FISCAIS", "DADOS TRIBUTÁRIOS",
    "DISCRIMINAÇÃO DOS SERVIÇOS",
    "DISCRIMINACAO DOS SERVICOS",
    "ICMS",
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normaliza: minúsculas, remove acentos simples, colapsa espaços."""
    s = s.lower()
    s = re.sub(r"[áàãâä]", "a", s)
    s = re.sub(r"[éèêë]", "e", s)
    s = re.sub(r"[íìîï]", "i", s)
    s = re.sub(r"[óòõôö]", "o", s)
    s = re.sub(r"[úùûü]", "u", s)
    s = re.sub(r"[ç]", "c", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_tax_name(text: str) -> Optional[str]:
    n = _norm(text)
    for key, canonical in _TAX_CANONICAL:
        if key in n:
            return canonical
    return None


def _classify_field(key_norm: str) -> str:
    """Retorna 'base', 'rate', 'amount' ou 'unknown'."""
    if any(w in key_norm for w in _BASE_WORDS):
        return "base"
    if any(w in key_norm for w in _RATE_WORDS):
        return "rate"
    if any(w in key_norm for w in _AMOUNT_WORDS):
        return "amount"
    return "unknown"


def _safe_parse(text: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.,\-]", "", text)
    if not cleaned:
        return None
    return parse_currency(cleaned)


# ---------------------------------------------------------------------------
# 1. Extração via kv_pairs
# ---------------------------------------------------------------------------

def extract_taxes_from_kv_pairs(
    kv_pairs: List[Tuple[str, str]],
) -> List[Tax]:
    """
    Varre os pares chave-valor exportados pelo Docling.
    Agrupa por nome canônico do imposto e classifica cada par como
    base/alíquota/valor. Exemplos de kv que captura:

      "BASE DE CÁLCULO DO ICMS"  → "19.040,00"   (base)
      "ALÍQUOTA DO ICMS"         → "18,00"        (rate)
      "VALOR DO ICMS"            → "3.427,23"     (amount)
      "ICMS"                     → "3.427,23"     (amount — chave simples)
    """
    buckets: Dict[str, Dict] = {}

    for key, value in kv_pairs:
        tax_name = _match_tax_name(key)
        if not tax_name:
            continue
        key_n = _norm(key)
        field = _classify_field(key_n)
        val = _safe_parse(value)
        if val is None or val < 0:
            continue

        bucket = buckets.setdefault(tax_name, {})
        if field == "base" and "base" not in bucket:
            bucket["base"] = val
        elif field == "rate" and "rate" not in bucket:
            bucket["rate"] = val
        elif field == "amount" and "amount" not in bucket:
            bucket["amount"] = val
        elif field == "unknown" and "amount" not in bucket and val > 0:
            # Chave sem qualificador (ex: "ICMS" → valor) → assume amount
            bucket["amount"] = val

    return _buckets_to_taxes(buckets)


# ---------------------------------------------------------------------------
# 2. Extração via tabelas (DataFrames do Docling)
# ---------------------------------------------------------------------------

def extract_taxes_from_tables(tables: List[pd.DataFrame]) -> List[Tax]:
    """
    Detecta tabelas de impostos em dois formatos:

    Formato A — linhas com nome do tributo (col 0 = nome):
      ICMS | 19.040,00 | 18,00% | 3.427,23

    Formato B — colunas com base/alíquota/valor (ICMS implícito ou no título):
      BASE DE CÁLCULO | ALÍQUOTA | VALOR DO ICMS
      19.040,00       | 18,00%   | 3.427,23
    """
    taxes: List[Tax] = []

    for df in tables:
        if df.empty:
            continue
        cols = [str(c) for c in df.columns]

        # ---- Formato A: primeira coluna tem nomes de impostos ----
        first_col = cols[0]
        tax_rows = _try_row_based(df, first_col, cols)
        if tax_rows:
            taxes.extend(tax_rows)
            continue

        # ---- Formato B: colunas são BASE / ALÍQUOTA / VALOR ----
        col_taxes = _try_column_based(df, cols)
        if col_taxes:
            taxes.extend(col_taxes)

    return _dedupe(taxes)


def _try_row_based(
    df: pd.DataFrame, name_col: str, all_cols: List[str]
) -> List[Tax]:
    """Extrai impostos quando cada linha representa um tributo."""
    buckets: Dict[str, Dict] = {}

    # Determina qual coluna é base/rate/amount pelo cabeçalho
    field_map: Dict[str, str] = {}
    for col in all_cols:
        if col == name_col:
            continue
        cn = _norm(col)
        if any(w in cn for w in _BASE_WORDS):
            field_map[col] = "base"
        elif any(w in cn for w in _RATE_WORDS):
            field_map[col] = "rate"
        elif any(w in cn for w in _AMOUNT_WORDS) or "valor" in cn or "total" in cn:
            field_map[col] = "amount"

    found_any_tax = False
    for _, row in df.iterrows():
        cell0 = str(row.get(name_col, "")).strip()
        tax_name = _match_tax_name(cell0)
        if not tax_name:
            continue
        found_any_tax = True
        bucket = buckets.setdefault(tax_name, {})

        if field_map:
            for col, field in field_map.items():
                raw = str(row.get(col, "")).strip()
                val = _safe_parse(raw)
                if val is not None and val >= 0 and field not in bucket:
                    bucket[field] = val
        else:
            # Sem cabeçalhos reconhecíveis: pega os valores numéricos em ordem
            nums = []
            for col in all_cols:
                if col == name_col:
                    continue
                raw = str(row.get(col, "")).strip()
                v = _safe_parse(raw)
                if v is not None and v >= 0:
                    nums.append(v)
            if len(nums) == 1:
                bucket.setdefault("amount", nums[0])
            elif len(nums) == 2:
                # assume base, amount (sem alíquota)
                bucket.setdefault("base", nums[0])
                bucket.setdefault("amount", nums[1])
            elif len(nums) >= 3:
                bucket.setdefault("base", nums[0])
                bucket.setdefault("rate", nums[1])
                bucket.setdefault("amount", nums[2])

    return _buckets_to_taxes(buckets) if found_any_tax else []


def _try_column_based(df: pd.DataFrame, cols: List[str]) -> List[Tax]:
    """
    Extrai impostos quando as colunas são BASE / ALÍQUOTA / VALOR.
    Tenta identificar o nome do imposto pela coluna "valor" ou pelo contexto.
    """
    base_col = rate_col = amount_col = name_col = None

    for col in cols:
        cn = _norm(col)
        tax_in_col = _match_tax_name(cn)
        if any(w in cn for w in _BASE_WORDS):
            base_col = col
        elif any(w in cn for w in _RATE_WORDS):
            rate_col = col
        elif tax_in_col and ("valor" in cn or "total" in cn or cn == tax_in_col.lower()):
            amount_col = col
            name_col = tax_in_col
        elif any(w in cn for w in _AMOUNT_WORDS):
            amount_col = col
            if tax_in_col:
                name_col = tax_in_col

    # Pelo menos dois dos três campos reconhecidos para continuar
    recognized = sum(x is not None for x in (base_col, rate_col, amount_col))
    if recognized < 2:
        return []

    # Nome: tenta extrair do nome da coluna de valor; fallback = ICMS (imposto
    # mais comum em faturas de energia elétrica)
    tax_name = name_col or "ICMS"

    buckets: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        vals = {}
        if base_col:
            v = _safe_parse(str(row.get(base_col, "")))
            if v is not None and v > 0:
                vals["base"] = v
        if rate_col:
            v = _safe_parse(str(row.get(rate_col, "")))
            if v is not None and v > 0:
                vals["rate"] = v
        if amount_col:
            v = _safe_parse(str(row.get(amount_col, "")))
            if v is not None and v > 0:
                vals["amount"] = v

        if not vals:
            continue

        bucket = buckets.setdefault(tax_name, {})
        for field, val in vals.items():
            if field not in bucket:
                bucket[field] = val

    return _buckets_to_taxes(buckets)


# ---------------------------------------------------------------------------
# 3. Extração via texto de seção (fallback semântico)
# ---------------------------------------------------------------------------

def extract_taxes_from_section_text(text: str) -> List[Tax]:
    """
    Extrai impostos do texto de uma seção identificada semanticamente pelo
    Docling (ex: "BASE DE CÁLCULO", "TRIBUTAÇÃO"). Opera com regex escoped —
    mais seguro que busca global porque o texto já foi selecionado pelo modelo
    de layout do Docling.

    Detecta padrões:
      • BASE: número > 1000 logo após label "base"/"bc"
      • ALÍQUOTA: número < 100 logo após label "aliq"/"%" ou seguido de "%"
      • VALOR DO ICMS: número após label de valor + "icms"
    """
    buckets: Dict[str, Dict] = {}
    text_n = _norm(text)

    # Detecta qual imposto aparece no texto
    tax_names_present = []
    for key, canonical in _TAX_CANONICAL:
        if key in text_n:
            tax_names_present.append(canonical)
    if not tax_names_present:
        return []

    # Para cada imposto encontrado, tenta extrair base/alíquota/valor
    for tax_name in set(tax_names_present):
        tax_key = tax_name.lower()

        # Base de cálculo: número após "base" ou "bc" em sequência
        base_m = re.search(
            r"base[^\d]{0,30}([\d]{1,3}(?:[.,]\d{3})*[.,]\d{2})",
            text, re.IGNORECASE,
        )
        # Alíquota: número < 100 seguido de "%" ou após "alíquota"
        rate_m = re.search(
            r"al[ií]quota[^\d]{0,20}([\d]{1,2}[.,]\d{1,4})|(\d{1,2}[.,]\d{1,4})\s*%",
            text, re.IGNORECASE,
        )
        # Valor do imposto: número após padrão "valor do <imposto>" ou "<imposto>" isolado
        amount_m = re.search(
            rf"valor\s+d[ao]\s+{tax_key}[^\d]{{0,20}}([\d]{{1,3}}(?:[.,]\d{{3}})*[.,]\d{{2}})"
            rf"|{tax_key}[^\d]{{0,10}}([\d]{{1,3}}(?:[.,]\d{{3}})*[.,]\d{{2}})",
            text, re.IGNORECASE,
        )

        bucket: Dict = {}
        if base_m:
            v = _safe_parse(base_m.group(1))
            if v and v > 0:
                bucket["base"] = v
        if rate_m:
            v = _safe_parse(rate_m.group(1) or rate_m.group(2))
            if v and 0 < v < 100:
                bucket["rate"] = v
        if amount_m:
            v = _safe_parse(amount_m.group(1) or amount_m.group(2))
            if v and v > 0:
                bucket["amount"] = v

        if bucket:
            buckets[tax_name] = bucket

    return _buckets_to_taxes(buckets)


# ---------------------------------------------------------------------------
# Helpers finais
# ---------------------------------------------------------------------------

def _buckets_to_taxes(buckets: Dict[str, Dict]) -> List[Tax]:
    taxes = []
    for name, data in buckets.items():
        if not data:
            continue
        taxes.append(Tax(
            name=name,
            base=data.get("base"),
            rate=data.get("rate"),
            amount=data.get("amount"),
        ))
    return taxes


def _dedupe(taxes: List[Tax]) -> List[Tax]:
    """Remove duplicatas, mantendo a entrada com mais campos preenchidos."""
    seen: Dict[str, Tax] = {}
    for t in taxes:
        existing = seen.get(t.name)
        if existing is None:
            seen[t.name] = t
        else:
            # Mantém o que tem mais campos não-nulos
            new_count = sum(1 for v in (t.base, t.rate, t.amount) if v is not None)
            old_count = sum(1 for v in (existing.base, existing.rate, existing.amount) if v is not None)
            if new_count > old_count:
                seen[t.name] = t
    return list(seen.values())

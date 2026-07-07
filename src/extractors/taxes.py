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

def _extract_by_calculation(text: str) -> List[Tax]:
    """
    Encontra triplas (base, alíquota, valor) matematicamente consistentes
    (valor ≈ base × alíquota / 100) e associa ao imposto pelo label mais
    próximo no texto.

    Resolve layouts DANFE NF3E onde os valores aparecem ANTES do label —
    ex: RGE SUL 141846706.pdf tem "223.361,36 / 17,00 / 37.971,43 [...] ICMS".
    Suporta triplas não-consecutivas (janela de até 6 tokens entre base e valor)
    para capturar PIS e COFINS em layouts como "base ... rate_pis rate_cofins
    ... total_pis total_cofins".
    """
    _MONEY_RE = re.compile(r"\b(\d[\d.,]*[.,]\d\d)\b")
    tokens = [(m.start(), _safe_parse(m.group(1))) for m in _MONEY_RE.finditer(text)]
    tokens = [(pos, v) for pos, v in tokens if v is not None]

    found: Dict[str, Tax] = {}
    _WINDOW = 6  # máx. tokens entre base e alíquota (e entre alíquota e valor)

    for i, (pos_b, base_v) in enumerate(tokens):
        if base_v <= 100:
            continue  # base deve ser > 100 (não é alíquota)

        for j in range(i + 1, min(i + 1 + _WINDOW, len(tokens))):
            pos_r, rate_v = tokens[j]
            if not (0 < rate_v < 100):
                continue

            expected = base_v * rate_v / 100
            if expected <= 0:
                continue

            for k in range(j + 1, min(j + 1 + _WINDOW, len(tokens))):
                pos_a, amount_v = tokens[k]
                if expected == 0:
                    continue
                error = abs(amount_v - expected) / expected
                if error > 0.02:  # tolerância 2 % (arredondamentos)
                    continue

                # Tripla válida — identifica imposto pelo label próximo no texto
                ctx_start = max(0, pos_b - 150)
                ctx_end = min(len(text), pos_a + 300)
                context = text[ctx_start:ctx_end].upper()

                for key, canonical in _TAX_CANONICAL:
                    if canonical not in found and key.upper() in context:
                        found[canonical] = Tax(
                            name=canonical,
                            base=base_v,
                            rate=rate_v,
                            amount=amount_v,
                        )
                        break  # imposto identificado; próxima tripla

    return list(found.values())


def extract_taxes_from_section_text(text: str) -> List[Tax]:
    """
    Extrai impostos do texto de uma seção identificada semanticamente pelo
    Docling (ex: "BASE DE CÁLCULO", "TRIBUTAÇÃO"). Opera com regex escoped —
    mais seguro que busca global porque o texto já foi selecionado pelo modelo
    de layout do Docling.

    Estratégia A — label → valor: regex busca label seguido de número.
      Melhorias vs abordagem simples:
        • Detecção de nomes usa \b (evita "iss" em "em**iss**ao", "pis" em "piso")
        • Alíquota: prefere "aliquota + número" antes de "número%" — evita capturar
          percentual de reajuste tarifário que aparece antes da tabela de ICMS
        • Valor: padrão com dois números (quando o primeiro ≤ 100 é alíquota, usa o
          segundo como valor) — resolve "ICMS Aliquota 24,000 2942,64" → 2942.64
    Estratégia B — cálculo matemático: encontra triplas (base × aliq/100 ≈ valor)
      mesmo quando os valores aparecem ANTES do label (ex: DANFE NF3E RGE SUL).
    A estratégia B substitui a A quando encontra resultado mais completo.
    """
    buckets: Dict[str, Dict] = {}

    # Padrão de número monetário: aceita 2-4 casas decimais, sem dígito seguinte.
    # "37.971,43" ✓  "24,000" ✓ (alíquota com 3 zeros)  "0,28537" ✗ (5 decimais)
    _MONEY = r"\d[\d.,]*[.,]\d{2,4}(?!\d)"

    # Detecta quais impostos aparecem no texto usando \b para evitar falsos
    # positivos por substring (ex: "iss" em "emissão", "pis" em "piso").
    tax_names_present: List[str] = []
    seen_canonical: set = set()
    for key, canonical in _TAX_CANONICAL:
        if canonical in seen_canonical:
            continue
        escaped = re.escape(key)  # "pis/pasep" → "pis\/pasep"
        if re.search(rf"\b{escaped}\b", text, re.IGNORECASE):
            tax_names_present.append(canonical)
            seen_canonical.add(canonical)
    if not tax_names_present:
        return []

    # Estratégia A: label → valor (regex com limites de palavra)
    for tax_name in tax_names_present:
        tax_key = tax_name.lower()
        _tk = re.escape(tax_key)

        # Base: número após "base" (com limite de palavra)
        base_m = re.search(
            rf"\bbase\b[^\d]{{0,50}}({_MONEY})",
            text, re.IGNORECASE,
        )

        # Alíquota: prefire "alíquota + número" antes de "número %" para evitar
        # capturar percentuais de contexto (ex: "reajuste tarifário de 13,46%").
        rate_m = (
            re.search(r"al[ií]quota[^\d]{0,20}(\d{1,2}[.,]\d{1,4})", text, re.IGNORECASE)
            or re.search(r"(\d{1,2}[.,]\d{1,4})\s*%", text, re.IGNORECASE)
        )

        # Valor do imposto — três padrões em ordem de preferência:
        #   1. "valor do ICMS NN,NN"
        #   2. dois números após label: se o primeiro ≤ 100 (alíquota?), usa o segundo
        #      (resolve "ICMS Aliquota 24,000 2942,64" → 2942,64)
        #   3. um número após label (fallback simples)
        _amount_val: Optional[float] = None
        vd_m = re.search(
            rf"valor\s+d[ao]\s+\b{_tk}\b[^\d]{{0,25}}({_MONEY})",
            text, re.IGNORECASE,
        )
        if vd_m:
            _amount_val = _safe_parse(vd_m.group(1))
        else:
            tk2_m = re.search(
                rf"\b{_tk}\b[^\d]{{0,20}}({_MONEY})[^\d]{{0,30}}({_MONEY})",
                text, re.IGNORECASE,
            )
            if tk2_m:
                v1 = _safe_parse(tk2_m.group(1))
                v2 = _safe_parse(tk2_m.group(2))
                if v1 is not None and v1 <= 100 and v2 is not None and v2 > 0:
                    _amount_val = v2  # primeiro é alíquota, segundo é o valor
                else:
                    _amount_val = v1
            else:
                tk1_m = re.search(
                    rf"\b{_tk}\b[^\d]{{0,20}}({_MONEY})",
                    text, re.IGNORECASE,
                )
                if tk1_m:
                    _amount_val = _safe_parse(tk1_m.group(1))

        bucket: Dict = {}
        if base_m:
            v = _safe_parse(base_m.group(1))
            if v and v > 0:
                bucket["base"] = v
        if rate_m:
            v = _safe_parse(rate_m.group(1))
            if v and 0 < v < 100:
                bucket["rate"] = v
        if _amount_val is not None and _amount_val >= 0:
            bucket["amount"] = _amount_val

        if bucket:
            buckets[tax_name] = bucket

    # Estratégia B: cálculo matemático (substitui A quando mais completo)
    for t in _extract_by_calculation(text):
        existing = buckets.get(t.name, {})
        calc_count = sum(1 for v in (t.base, t.rate, t.amount) if v is not None)
        exist_count = sum(1 for v in existing.values() if v is not None)
        if calc_count > exist_count:
            buckets[t.name] = {"base": t.base, "rate": t.rate, "amount": t.amount}

    return _buckets_to_taxes(buckets)


# ---------------------------------------------------------------------------
# Helpers finais
# ---------------------------------------------------------------------------

def _buckets_to_taxes(buckets: Dict[str, Dict]) -> List[Tax]:
    taxes = []
    for name, data in buckets.items():
        if not data:
            continue
        # Descarta entradas com apenas alíquota e sem base/valor — provavelmente ruído
        # (ex: "iss" detectado em "emissão" e rate matched por "%" de ajuste tarifário)
        if data.get("amount") is None and data.get("base") is None:
            continue
        taxes.append(Tax(
            name=name,
            base=data.get("base"),
            rate=data.get("rate"),
            amount=data.get("amount"),
        ))
    return taxes


def build_taxes(buckets: Dict[str, Dict]) -> List[Tax]:
    """Constrói lista de Tax a partir de {nome: {base, rate, amount}}."""
    return _buckets_to_taxes(buckets)


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

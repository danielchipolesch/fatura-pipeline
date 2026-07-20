"""
Extração de itens de linha (line_items) a partir de tabelas Docling (DataFrames).

Compartilhada entre DoclingExtractor (camada primária) e DeterministicParser
(fallback), garantindo lógica idêntica independentemente de qual camada extrai.
"""
from __future__ import annotations

import re
from typing import List

import pandas as pd

from src.extractors.fields import parse_currency
from src.models.invoice import LineItem

_desc_aliases  = {"descri", "produto", "servi", "item", "discrimina", "histor", "especif"}
_qty_aliases   = {"qtd", "quant", "qty", "amount"}
_price_aliases = {"unit", "pre", "price", "valor unit"}
_total_aliases = {"total", "valor total", "vlr total", "subtotal"}
_code_aliases  = {"cod", "código", "codigo", "ref", "ncm", "cfop"}

# Mapeamento de unidades por descrição — usado quando a coluna UN não é extraída pelo Docling.
# As unidades são fixadas por norma ANEEL: energia ativa em kWh, demanda em kW.
_UNIT_BY_DESC_FRAGMENT: list[tuple[str, str]] = [
    ("energia ativa",   "kWh"),
    ("energia reativa", "kVARh"),
    ("demanda reativa", "kVAR"),
    ("demanda ativa",   "kW"),
    ("demanda",         "kW"),
]

# Linhas que são cabeçalhos residuais ou metadados de histórico de leitura —
# não representam itens de cobrança.
_NOISE_ROW = re.compile(
    r"^(TOTAL|SUBTOTAL|HISTÓRICO|HISTÓRIA|MÊS|ANO|UF|\d{1,2}/\d{2,4}|"
    r"OUT|SET|AGO|JUL|JUN|MAI|ABR|MAR|FEV|JAN|DEZ|NOV|$)",
    re.IGNORECASE,
)

# Padrão de "código de produto curto" — quando a coluna de descrição contém
# apenas códigos (ex: ZENF01, 100, ENER01) em vez de texto descritivo.
_IS_CODE = re.compile(r"^[A-Z0-9]{2,12}$")


def _col_match(col_name: str, aliases: set) -> bool:
    normalized = str(col_name).lower().strip()
    return any(a in normalized for a in aliases)


def _pick_desc_col(cols: list) -> str:
    """
    Escolhe a coluna de descrição entre as colunas disponíveis.

    Heurística:
    1. Coluna cujo nome contém um dos _desc_aliases
    2. Coluna cujo nome contém "nome"
    3. Se a primeira coluna tem nome que parece código (code alias), tenta a segunda
    4. Fallback: primeira coluna
    """
    # 1. Alias direto
    for c in cols:
        if _col_match(c, _desc_aliases):
            return c
    # 2. "nome"
    for c in cols:
        if "nome" in str(c).lower():
            return c
    # 3. Se col[0] é código, tenta col[1]
    if len(cols) >= 2 and _col_match(cols[0], _code_aliases):
        return cols[1]
    # 4. Fallback
    return cols[0]


def extract_line_items_from_tables(tables: List[pd.DataFrame]) -> List[LineItem]:
    """
    Percorre os DataFrames extraídos pelo Docling em busca de tabelas de itens.
    Identifica a tabela de produtos/serviços pelo padrão das colunas.

    Heurísticas especiais para NF-e:
      - Fallback 1: valor numérico no nome da coluna (inversão header/dado do Docling)
      - Fallback 2: "V.TOTAL/V.UNITARIO <qty>" embutido no nome de outra coluna
      - Fallback 3: coluna de código (ex: "ZENF01") ≠ coluna de descrição
    """
    items: List[LineItem] = []

    for df in tables:
        cols = list(df.columns)

        has_desc  = any(_col_match(c, _desc_aliases) for c in cols)
        has_value = any(_col_match(c, _price_aliases | _total_aliases) for c in cols)

        if not (has_desc or (has_value and len(cols) >= 3)):
            continue

        desc_col  = _pick_desc_col(cols)
        qty_col   = next((c for c in cols if _col_match(c, _qty_aliases)), None)
        price_col = next((c for c in cols if _col_match(c, _price_aliases)), None)
        total_col = next((c for c in cols if _col_match(c, _total_aliases)), None)
        un_col    = next(
            (c for c in cols if str(c).strip().upper() in ("UN", "UNID", "UNIDADE", "UNIT", "UND")),
            None,
        )
        # Coluna de código do produto (separada da descrição)
        code_col  = next((c for c in cols if _col_match(c, _code_aliases) and c != desc_col), None)

        # Detecta deslocamento de coluna DANFE: o Docling às vezes remove o cabeçalho
        # da coluna UN (unidade), mantendo a coluna com valor mas sem header. Isso empurra
        # os dados uma coluna para a esquerda — a coluna em branco entre CFOP e QTD tem
        # a quantidade real, e a coluna QTD passa a ter o valor unitário.
        _danfe_shift_col = None
        _cfop_idx = next((i for i, c in enumerate(cols) if "cfop" in str(c).lower()), None)
        if _cfop_idx is not None and qty_col is not None:
            _qty_idx = cols.index(qty_col)
            for _ci in range(_cfop_idx + 1, _qty_idx):
                _cn = str(cols[_ci]).strip()
                if not _cn or _cn.lower() in ("nan", "none", ""):
                    _danfe_shift_col = cols[_ci]
                    break

        for _, row in df.iterrows():
            desc_val = str(row.get(desc_col, "")).strip()
            if not desc_val or desc_val.lower() in ("nan", "none", ""):
                continue
            if _NOISE_ROW.match(desc_val):
                continue
            if re.match(r"^[\d.,\s]+$", desc_val):
                continue

            # Se a descrição parece um código curto (ex: "ZENF01"), tenta usar
            # a próxima coluna disponível que não seja numérica como descrição real.
            if _IS_CODE.match(desc_val):
                # Guarda o valor como code, tenta next text column as description
                code_candidate = desc_val
                alt_desc = None
                for c in cols:
                    if c == desc_col:
                        continue
                    v = str(row.get(c, "")).strip()
                    if v and v.lower() not in ("nan", "none", "") and not re.match(r"^[\d.,\s]+$", v) and not _IS_CODE.match(v):
                        alt_desc = v
                        break
                if alt_desc:
                    desc_val = alt_desc
                    # Não sobrescreve code_col se já temos um; usa o candidato
                    if not code_col:
                        code_col = desc_col

            qty = None
            _danfe_shifted = False
            # DANFE shift: coluna em branco entre CFOP e QTD tem a quantidade real
            if _danfe_shift_col is not None:
                _bv = str(row.get(_danfe_shift_col, "")).strip()
                if _bv and _bv.lower() not in ("nan", "none"):
                    _bc = parse_currency(_bv)
                    if _bc is not None and _bc > 1:
                        qty = _bc
                        _danfe_shifted = True

            if qty is None and qty_col and str(row.get(qty_col, "")).strip():
                try:
                    qty = parse_currency(str(row[qty_col]))
                except Exception:
                    pass

            # Fallback 1: inversão header/dado do Docling — valor no nome da coluna
            if qty is None and qty_col and _col_match(str(qty_col), _qty_aliases):
                candidate = parse_currency(str(qty_col))
                if candidate is not None and 0 < candidate < 1_000_000:
                    qty = candidate

            # Fallback 2: "V.TOTAL/V.UNITARIO <qty>" embutido no nome de outra coluna
            if qty is None:
                for col in cols:
                    m_emb = re.search(
                        r"\bV\.?\s*(?:TOTAL|UNIT(?:ARIO)?)\s+([\d]{1,6}[.,][\d]+)\s*$",
                        str(col), re.IGNORECASE,
                    )
                    if m_emb:
                        candidate = parse_currency(m_emb.group(1))
                        if candidate is not None and 0 < candidate < 1_000_000:
                            qty = candidate
                            break

            unit_val_str = str(row.get(un_col, "")).strip() if un_col else ""
            unit = unit_val_str if unit_val_str and unit_val_str.lower() not in ("nan", "none") else None

            # Inferência de unidade por descrição quando a coluna UN não foi extraída.
            if unit is None:
                desc_lower = desc_val.lower()
                for _frag, _inferred in _UNIT_BY_DESC_FRAGMENT:
                    if _frag in desc_lower:
                        unit = _inferred
                        break

            unit_price = None
            if _danfe_shifted:
                # No deslocamento DANFE, a coluna QTD tem o valor unitário real
                if qty_col and str(row.get(qty_col, "")).strip():
                    unit_price = parse_currency(str(row[qty_col]))
            elif price_col and str(row.get(price_col, "")).strip():
                unit_price = parse_currency(str(row[price_col]))

            total_val = None
            if total_col and str(row.get(total_col, "")).strip():
                total_val = parse_currency(str(row[total_col]))

            code_val = None
            if code_col:
                v = str(row.get(code_col, "")).strip()
                if v and v.lower() not in ("nan", "none", ""):
                    code_val = v

            items.append(LineItem(
                code=code_val,
                description=desc_val,
                unit=unit,
                quantity=qty,
                unit_price=unit_price,
                total=total_val,
            ))

    return items

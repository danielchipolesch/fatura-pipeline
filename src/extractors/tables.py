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

_desc_aliases  = {"descri", "produto", "servi", "item", "discrimina", "histor"}
_qty_aliases   = {"qtd", "quant", "qty", "amount"}
_price_aliases = {"unit", "pre", "price", "valor unit"}
_total_aliases = {"total", "valor total", "vlr total", "subtotal"}

_NOISE_ROW = re.compile(
    r"^(TOTAL|SUBTOTAL|HISTÓRICO|HISTÓRIA|MÊS|ANO|UF|HP|HFP|HR|\d{1,2}/\d{2,4}|"
    r"OUT|SET|AGO|JUL|JUN|MAI|ABR|MAR|FEV|JAN|DEZ|NOV|$)",
    re.IGNORECASE,
)


def _col_match(col_name: str, aliases: set) -> bool:
    normalized = str(col_name).lower().strip()
    return any(a in normalized for a in aliases)


def extract_line_items_from_tables(tables: List[pd.DataFrame]) -> List[LineItem]:
    """
    Percorre os DataFrames extraídos pelo Docling em busca de tabelas de itens.
    Identifica a tabela de produtos/serviços pelo padrão das colunas.

    Heurísticas especiais para NF-e:
      - Fallback 1: valor numérico no nome da coluna (inversão header/dado do Docling)
      - Fallback 2: "V.TOTAL/V.UNITARIO <qty>" embutido no nome de outra coluna
    """
    items: List[LineItem] = []

    for df in tables:
        cols = list(df.columns)

        has_desc  = any(_col_match(c, _desc_aliases) for c in cols)
        has_value = any(_col_match(c, _price_aliases | _total_aliases) for c in cols)

        if not (has_desc or (has_value and len(cols) >= 3)):
            continue

        desc_col  = next((c for c in cols if _col_match(c, _desc_aliases)), cols[0])
        qty_col   = next((c for c in cols if _col_match(c, _qty_aliases)), None)
        price_col = next((c for c in cols if _col_match(c, _price_aliases)), None)
        total_col = next((c for c in cols if _col_match(c, _total_aliases)), None)
        un_col    = next(
            (c for c in cols if str(c).strip().upper() in ("UN", "UNID", "UNIDADE", "UNIT", "UND")),
            None,
        )

        for _, row in df.iterrows():
            desc_val = str(row.get(desc_col, "")).strip()
            if not desc_val or desc_val.lower() in ("nan", "none", ""):
                continue
            if _NOISE_ROW.match(desc_val):
                continue
            if re.match(r"^[\d.,\s]+$", desc_val):
                continue

            qty = None
            if qty_col and str(row.get(qty_col, "")).strip():
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

            unit_price = None
            if price_col and str(row.get(price_col, "")).strip():
                unit_price = parse_currency(str(row[price_col]))

            total_val = None
            if total_col and str(row.get(total_col, "")).strip():
                total_val = parse_currency(str(row[total_col]))

            items.append(LineItem(
                description=desc_val,
                unit=unit,
                quantity=qty,
                unit_price=unit_price,
                total=total_val,
            ))

    return items

"""
Parser determinístico de faturas.

Estratégia geral:
  1. Extrai campos usando padrões regex específicos para o layout detectado.
  2. Faz fallback para padrões genéricos quando o padrão específico não casa.
  3. Para tabelas de itens, usa os DataFrames extraídos pelo docling.
  4. Calcula score de confiança com base nos campos preenchidos.

Nenhuma LLM é usada aqui — tudo é baseado em regras.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from src.extractors.fields import (
    extract_access_key,
    extract_all_cnpjs,
    extract_all_currencies,
    extract_all_dates,
    extract_all_dates_ptbr_abbrev,
    extract_boleto_barcode,
    extract_boleto_line,
    extract_br_state,
    extract_cnpj,
    extract_cnpj_after_label,
    extract_client_number,
    extract_consumer_unit,
    extract_consumption_kwh,
    extract_cpf,
    extract_currency,
    extract_date,
    extract_date_before_currency,
    extract_demand_overage_value,
    extract_email,
    extract_energy_quantity_mwh,
    extract_labeled_value,
    extract_lines_after_label,
    extract_measured_demand_kw,
    extract_tarifa_azul_demand_nf3e,
    extract_number_after_hex_signature,
    extract_phone,
    extract_reactive_excess,
    format_cnpj_digits,
    has_hex_signature,
    parse_currency,
    parse_currency_or_flag,
    strip_label_prefix,
)
from src.extractors.known_entities import (
    lookup_customer_name,
    lookup_customer_name_by_text,
    lookup_supplier,
    lookup_supplier_by_name,
)
from src.models.invoice import (
    Address,
    EnergyMetrics,
    Invoice,
    InvoiceLayout,
    LineItem,
    Party,
    ParsingMethod,
    READ_ERROR_SENTINEL,
    Tax,
)
from src.parsers.docling_loader import DocumentContent

# ---------------------------------------------------------------------------
# Weights para score de confiança
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "invoice_number": 0.15,
    "issue_date": 0.15,
    "supplier_name": 0.15,
    "supplier_cnpj": 0.10,
    "customer_name": 0.08,
    "customer_cnpj": 0.07,
    "total": 0.20,
    "line_items": 0.05,
    "due_date": 0.05,
}


# ---------------------------------------------------------------------------
# Extratores de campos por layout
# ---------------------------------------------------------------------------

class _FieldExtractor:
    """Namespace estático de extratores genéricos e por layout."""

    # -- Número da fatura / nota --

    @staticmethod
    def invoice_number(text: str, layout: InvoiceLayout) -> Optional[str]:
        patterns_by_layout: dict[InvoiceLayout, list[str]] = {
            InvoiceLayout.NFE: [
                r"NOTA\s+FISCAL\s*:\s*(\d+)",
                r"N[°ºo\.]\s*(?:da\s+)?(?:Nota|NF)?\s*[:\s]+(\d+)",
                r"N[úu]mero\s*[:\s]+(\d+)",
                r"N[°ºo]\.\s*(\d+)",
            ],
            InvoiceLayout.NFSE: [
                r"N[úu]mero\s+(?:da\s+)?(?:Nota|NFS?)\s*[:\s]+(\d+)",
                r"NOTA\s+FISCAL\s+N[°ºo\.]\s*(\d+)",
                r"NOTA\s+FISCAL\s*:\s*(\d+)",
                r"N[úu]mero\s*[:\s]+(\d+)",
            ],
            InvoiceLayout.BOLETO: [
                r"(?:N[°ºo\.]\s+do\s+)?Documento\s*[:\s]+([A-Z0-9\-/]+)",
                r"Nosso\s+N[úu]mero\s*[:\s]+([A-Z0-9\-/]+)",
                r"N[úu]mero\s*[:\s]+([A-Z0-9\-/]+)",
            ],
            InvoiceLayout.UTILITY: [
                r"NOTA\s+FISCAL\s*:\s*(\d+)",
                r"NOTA\s+FISCAL\s+N[°ºo\.]\s*(\d+)",
                r"(?:N[°ºo\.]\s+da\s+)?Fatura\s*[:\s]+([A-Z0-9\-/]+)",
                r"Refer[êe]ncia\s*[:\s]+([A-Z0-9\-/]+)",
            ],
            InvoiceLayout.GENERIC: [
                r"NOTA\s+FISCAL\s*:\s*(\d+)",
                r"(?:Fatura|Invoice)\s*[#N°ºo\.]+\s*[:\s]*([A-Z0-9\-/]+)",
                r"N[°ºo\.]\s*(?:da\s+)?(?:Nota|NF)\s*[:\s]+([A-Z0-9\-/]+)",
                r"N[úu]mero\s*[:\s]+([A-Z0-9\-/]+)",
            ],
        }
        patterns = patterns_by_layout.get(layout, patterns_by_layout[InvoiceLayout.GENERIC])
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val and val not in ("0", ""):
                    return val
        return None

    # -- Datas --

    @staticmethod
    def issue_date(text: str, layout: InvoiceLayout) -> Optional[date]:
        labels = {
            InvoiceLayout.NFE: [r"Data\s+de\s+Emiss[ãa]o", r"Emiss[ãa]o"],
            InvoiceLayout.NFSE: [r"Data\s+de\s+Emiss[ãa]o", r"Emitida\s+em"],
            InvoiceLayout.BOLETO: [r"Data\s+(?:de\s+)?Emiss[ãa]o", r"Emiss[ãa]o"],
            InvoiceLayout.UTILITY: [r"Emiss[ãa]o", r"Data\s+da\s+Fatura"],
            InvoiceLayout.GENERIC: [r"Data\s+(?:de\s+)?Emiss[ãa]o", r"Emitido\s+em", r"Issue\s+Date"],
        }
        for label in labels.get(layout, labels[InvoiceLayout.GENERIC]):
            val = extract_labeled_value(text, label, r"[\d/\-\.]{8,10}")
            if val:
                d = extract_date(val)
                if d:
                    return d
        # fallback: primeira data do documento
        dates = extract_all_dates(text)
        return dates[0] if dates else None

    @staticmethod
    def due_date(text: str, layout: InvoiceLayout) -> Optional[date]:
        labels = [
            r"Vencimento",
            r"Data\s+(?:de\s+)?Vencimento",
            r"Pagar\s+at[eé]",
            r"Due\s+Date",
        ]
        for label in labels:
            val = extract_labeled_value(text, label, r"[\d/\-\.]{8,10}")
            if val:
                d = extract_date(val)
                if d:
                    return d
        dates = extract_all_dates(text)
        return dates[1] if len(dates) >= 2 else None

    # -- Partes (Emitente / Destinatário) --

    @staticmethod
    def _build_party(name_line: str, block_text: str) -> Party:
        """Constrói um objeto Party a partir do nome e bloco de texto circundante."""
        cnpj = extract_cnpj(block_text)
        cpf = None if cnpj else extract_cpf(block_text)
        ie_m = re.search(r"INSCRI[ÇC][ÃA]O\s+ESTADUAL\s*[:\s]+([0-9.\-/]{5,20})", block_text, re.IGNORECASE)
        email = extract_email(block_text)
        phone = extract_phone(block_text)

        # CEP
        cep_m = re.search(r"\b(\d{5})[\-](\d{3})\b", block_text)

        # Estado: somente siglas válidas de UF brasileiras
        state = extract_br_state(block_text)

        # Município: após label explícito, ou inline CIDADE/UF
        city_m = re.search(
            r"(?:MUN[IÍ]C[IÍ]PIO|CIDADE|MUNIC[ÍI]PIO)\s*[:\s]+([^\n,/]{3,40})",
            block_text, re.IGNORECASE,
        )
        city: Optional[str] = city_m.group(1).strip() if city_m else None
        if not city:
            inline_m = re.search(
                r"\b([A-ZÀ-Ú][A-ZÀ-Ú\s]{2,38}?)\s*/\s*([A-Z]{2})\b",
                block_text.upper(),
            )
            if inline_m:
                candidate_state = extract_br_state(inline_m.group(2))
                if candidate_state:
                    city = inline_m.group(1).strip()
                    if not state:
                        state = candidate_state

        address = Address(
            zip_code=f"{cep_m.group(1)}-{cep_m.group(2)}" if cep_m else None,
            state=state,
            city=city,
        )

        return Party(
            name=name_line.strip() if name_line else None,
            cnpj=cnpj,
            cpf=cpf,
            ie=ie_m.group(1).strip() if ie_m else None,
            address=address,
            email=email,
            phone=phone,
        )

    @classmethod
    def supplier(cls, text: str, layout: InvoiceLayout) -> Optional[Party]:
        # NF-e / DANFE: nome do emitente está na frase padrão "RECEBEMOS DE [NAME] OS PRODUTOS"
        if layout == InvoiceLayout.NFE:
            m = re.search(
                r"RECEBEMOS\s+DE\s+(.+?)\s+OS\s+PRODUTOS\s+CONSTANTES",
                text, re.IGNORECASE | re.DOTALL,
            )
            if m:
                name = re.sub(r"\s+", " ", m.group(1)).strip()
                # Endereço: usa a seção EMITENTE quando encontrada (evita cruzar
                # com o bloco DESTINATÁRIO que aparece próximo em DANFEs).
                # Fallback para os primeiros 1500 chars se a seção não for achada.
                emitente_m = re.search(r"EMIT[EI]NTE", text, re.IGNORECASE)
                if emitente_m:
                    block_text = text[emitente_m.start(): emitente_m.start() + 600]
                else:
                    block_text = text[:1500]
                return cls._build_party(name, block_text)

            # Fallback: primeiras linhas antes de "DANFE" (lado esquerdo do cabeçalho)
            danfe_pos = re.search(r"\bDANFE\b", text, re.IGNORECASE)
            if danfe_pos:
                header = text[:danfe_pos.start()]
                lines = [l.strip() for l in header.splitlines() if l.strip()]
                # Filtra linhas que são claramente labels ou de baixo valor
                skip = re.compile(
                    r"^(RECEBEMOS|DATA|IDENTIFICAÇÃO|NF-?e|N[°ºo]|SÉRIE|$)", re.IGNORECASE
                )
                candidates = [l for l in lines if not skip.match(l) and len(l) > 5]
                if candidates:
                    return cls._build_party(candidates[0], header)

        supplier_labels = {
            InvoiceLayout.NFSE: [r"PRESTADOR\s+DE\s+SERVI[ÇC]OS", r"Prestador"],
            InvoiceLayout.BOLETO: [r"BENEFICI[AÁ]RIO", r"CEDENTE"],
            InvoiceLayout.UTILITY: [r"CONCESSION[AÁ]RIA", r"DISTRIBUIDORA"],
            InvoiceLayout.GENERIC: [r"FORNECEDOR", r"EMIT[EI]NTE", r"Empresa", r"Vendedor"],
        }
        # Para NF-e sem a frase "RECEBEMOS DE", tenta label "EMITENTE" dominante na linha
        if layout == InvoiceLayout.NFE:
            lines = extract_lines_after_label(text, r"EMIT[EI]NTE", n_lines=4, dominant=True)
            if lines:
                block_start = re.search(r"EMIT[EI]NTE", text, re.IGNORECASE)
                block_text = text[block_start.start(): block_start.start() + 600] if block_start else "\n".join(lines)
                return cls._build_party(lines[0], block_text)

        for label in supplier_labels.get(layout, supplier_labels[InvoiceLayout.GENERIC]):
            lines = extract_lines_after_label(text, label, n_lines=6, dominant=True)
            if lines:
                block_start = re.search(label, text, re.IGNORECASE)
                block_text = text[block_start.start(): block_start.start() + 600] if block_start else "\n".join(lines)
                return cls._build_party(lines[0], block_text)

        # Último recurso: alguns layouts não trazem nenhum rótulo de seção para
        # o emitente (ex: certas faturas de energia sem "RECEBEMOS DE"/"EMITENTE"
        # nem CNPJ impresso no corpo do texto). Nesses casos, o nome da empresa
        # aparece apenas como texto literal — buscamos no cadastro conhecido.
        known = lookup_supplier_by_name(text)
        if known:
            return Party(name=known.get("name"), cnpj=known.get("cnpj"))
        return None

    _CUSTOMER_LABEL_PREFIXES = [
        r"NOME[/\\]?RAZ[ÃA]O\s+SOCIAL",
        r"NOME",
        r"RAZ[ÃA]O\s+SOCIAL",
        r"SACADO",
        r"PAGADOR",
    ]

    @classmethod
    def customer(cls, text: str, layout: InvoiceLayout) -> Optional[Party]:
        customer_labels = {
            InvoiceLayout.NFE: [r"DESTINAT[AÁ]RIO(?:/REMETENTE)?"],
            InvoiceLayout.NFSE: [r"TOMADOR\s+DE\s+SERVI[ÇC]OS", r"Tomador"],
            InvoiceLayout.BOLETO: [r"SACADO", r"PAGADOR", r"Devedor"],
            InvoiceLayout.UTILITY: [r"CONSUMIDOR", r"CLIENTE", r"UNIDADE\s+CONSUMIDORA"],
            InvoiceLayout.GENERIC: [r"CLIENTE", r"DESTINAT[AÁ]RIO", r"COMPRADOR"],
        }
        for label in customer_labels.get(layout, customer_labels[InvoiceLayout.GENERIC]):
            lines = extract_lines_after_label(text, label, n_lines=6, dominant=True)
            if not lines:
                continue
            block_start = re.search(label, text, re.IGNORECASE)
            block_text = text[block_start.start(): block_start.start() + 600] if block_start else "\n".join(lines)

            # Procura a linha com o nome real: descarta linhas que são apenas rótulos
            clean_name = ""
            for candidate in lines:
                stripped = strip_label_prefix(candidate, cls._CUSTOMER_LABEL_PREFIXES)
                if stripped and len(stripped) > 3:
                    clean_name = stripped
                    break

            if not clean_name and lines:
                clean_name = lines[0]  # último recurso: usa a primeira linha sem strip

            return cls._build_party(clean_name, block_text)

        # Último recurso: layout sem rótulo de seção para o destinatário, ou
        # CNPJ mascarado no PDF por privacidade (ex: "00.***.***/****-04") —
        # nesses casos buscamos o nome do cliente como texto literal conhecido.
        known_name = lookup_customer_name_by_text(text)
        if known_name:
            return Party(name=known_name)
        return None

    # -- Totais --

    _TOTAL_LABELS_BY_LAYOUT = {
        InvoiceLayout.NFE: [
            r"VALOR\s+TOTAL\s+(?:DA\s+)?(?:NOTA|NF)",
            r"Total\s+(?:da\s+)?Nota",
            r"VALOR\s+DO\s+DOCUMENTO",
            r"\bTOTAL\b",
        ],
        InvoiceLayout.NFSE: [
            r"VALOR\s+TOTAL\s+(?:DOS\s+)?SERVI[ÇC]OS",
            r"Valor\s+L[íi]quido",
            r"Total\s+(?:a\s+)?Pagar",
        ],
        InvoiceLayout.BOLETO: [
            r"VALOR\s+DO\s+DOCUMENTO",
            r"Valor\s+Cobrado",
            r"Total\s+(?:a\s+)?Pagar",
        ],
        InvoiceLayout.UTILITY: [
            r"VALOR\s+DO\s+DOCUMENTO",
            r"TOTAL\s+(?:A\s+)?PAGAR",
            r"Valor\s+(?:Total\s+)?(?:da\s+)?Fatura",
            r"\bTOTAL\b",
        ],
        InvoiceLayout.GENERIC: [
            r"TOTAL\s+GERAL",
            r"VALOR\s+TOTAL",
            r"Total\s+(?:a\s+)?Pagar",
            r"Grand\s+Total",
        ],
    }

    @classmethod
    def total_via_labels(cls, text: str, layout: InvoiceLayout) -> float | str | None:
        """
        Busca o total somente via rótulos conhecidos — sem heurística de
        fallback. Se algum rótulo for encontrado mas nenhum valor capturado
        puder ser convertido em número válido, retorna READ_ERROR_SENTINEL
        em vez de None — sinaliza que HÁ um total na fatura, mas a extração
        falhou (diferente de nenhum rótulo ter sido encontrado).
        """
        labels = cls._TOTAL_LABELS_BY_LAYOUT.get(layout, cls._TOTAL_LABELS_BY_LAYOUT[InvoiceLayout.GENERIC])
        label_matched_unparseable = False
        for label in labels:
            val = extract_labeled_value(text, label, r"R?\$?\s*[\d.,\-]+")
            if val:
                v = parse_currency(val)
                if v is not None and v > 0:
                    return v
                label_matched_unparseable = True
        return READ_ERROR_SENTINEL if label_matched_unparseable else None

    @staticmethod
    def total(text: str, layout: InvoiceLayout) -> float | str | None:
        """
        Busca o total via rótulos e, na ausência destes, usa o maior valor
        monetário do documento como último recurso.

        Cuidado: esse fallback de "maior valor" pode escolher uma base de
        cálculo de imposto em vez do total real (bases de cálculo costumam
        ser numericamente maiores que o valor final a pagar). Por isso, no
        parser principal, o fallback posicional via âncora "MTE\\d{7}"
        (estável em faturas Enel) é tentado ANTES deste método ser usado
        como último recurso — ver DeterministicParser.parse().
        """
        labeled = _FieldExtractor.total_via_labels(text, layout)
        if labeled is not None:
            return labeled  # float ou READ_ERROR_SENTINEL

        values = extract_all_currencies(text)
        return max(values) if values else None

    @staticmethod
    def subtotal(text: str) -> float | str | None:
        label_patterns = [
            r"Subtotal",
            r"VALOR\s+(?:DOS\s+)?PRODUTOS",
            r"VALOR\s+(?:DOS\s+)?SERVI[ÇC]OS",
            r"Base\s+de\s+C[áa]lculo",
        ]
        label_matched_unparseable = False
        for label in label_patterns:
            val = extract_labeled_value(text, label, r"R?\$?\s*[\d.,\-]+")
            if val:
                v = parse_currency(val)
                if v is not None and v > 0:
                    return v
                label_matched_unparseable = True
        return READ_ERROR_SENTINEL if label_matched_unparseable else None

    # -- Impostos --

    @staticmethod
    def taxes(text: str, layout: InvoiceLayout) -> list[Tax]:
        """
        Extrai impostos do texto. Cada imposto pode ter mais de um padrão,
        tentados em ordem do mais específico para o mais genérico — documentos
        diferentes (ou seções diferentes da mesma fatura) usam estruturas bem
        distintas para o mesmo tributo:
          - "TRIBUTO\\nBASE\\nALÍQUOTA%\\nVALOR" — tabela de 3 números em
            sequência (comum em faturas Enel SP). Tentada PRIMEIRO porque,
            sendo mais específica, evita o risco de capturar a base de
            cálculo como se fosse o valor do imposto (ver abaixo).
          - "Valor do ICMS: R$ X" — estrutura padrão de DANFE, um só número.
          - "CSLL (1,00%): R$49.00" — estrutura de "impostos retidos" anotada
            separadamente do cálculo principal (formato do valor pode vir em
            notação americana, "49.00"; parse_currency já trata isso).

        Cuidado: SEM a tentativa da estrutura de tabela primeiro, o padrão
        simples capturaria erroneamente a BASE DE CÁLCULO em vez do VALOR do
        imposto nesses casos (ex.: "ICMS\\n36.324,77\\n18,00\\n6.538,43" —
        o padrão simples pegaria 36.324,77, mas o valor real do ICMS é
        6.538,43). A estrutura de tabela usa os 3 números corretamente como
        base/alíquota/valor.

        Quando um rótulo é encontrado mas o valor capturado não pode ser
        convertido em número, o campo correspondente recebe
        READ_ERROR_SENTINEL em vez de ser descartado silenciosamente —
        sinaliza que HÁ um dado na fatura, mas a extração falhou.
        """
        result: list[Tax] = []
        # name -> lista de (regex, número_de_grupos: 1=só valor, 3=base/aliquota/valor)
        tax_defs: list[tuple[str, list[tuple[str, int]]]] = [
            ("ICMS", [
                (r"\bICMS\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)", 3),
                (r"(?:Valor\s+do\s+)?ICMS\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("IPI", [
                (r"(?:Valor\s+do\s+)?IPI\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("ISS", [
                (r"(?:Valor\s+do\s+)?ISS(?:QN)?\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("PIS", [
                (r"PIS(?:/PASEP)?\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)", 3),
                (r"PIS(?:/PASEP)?\s*\([\d.,]+\s*%\)\s*:\s*R\$\s*([\d.,\-]+)", 1),
                (r"(?:Valor\s+do\s+)?PIS(?:/PASEP)?\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("COFINS", [
                (r"COFINS\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)\s*\n?\s*([\d.,\-]+)", 3),
                (r"COFINS\s*\([\d.,]+\s*%\)\s*:\s*R\$\s*([\d.,\-]+)", 1),
                (r"(?:Valor\s+da?\s+)?COFINS\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("CSLL", [
                (r"CSLL\s*\([\d.,]+\s*%\)\s*:\s*R\$\s*([\d.,\-]+)", 1),
                (r"(?:Valor\s+da?\s+)?CSLL\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
            ("IRPJ", [
                (r"IRPJ\s*\([\d.,]+\s*%\)\s*:\s*R\$\s*([\d.,\-]+)", 1),
                (r"(?:Valor\s+do\s+)?IRPJ\s*[:\s]+R?\$?\s*([\d.,\-]+)", 1),
            ]),
        ]
        for name, patterns in tax_defs:
            tax: Optional[Tax] = None
            for pattern, n_groups in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if not m:
                    continue
                if n_groups == 3:
                    tax = Tax(
                        name=name,
                        base=parse_currency_or_flag(m.group(1)),
                        rate=parse_currency_or_flag(m.group(2)),
                        amount=parse_currency_or_flag(m.group(3)),
                    )
                else:
                    tax = Tax(name=name, amount=parse_currency_or_flag(m.group(1)))
                break
            if tax is None:
                continue
            # Imposto com valor negativo é sinal de que o rótulo provavelmente
            # capturou a linha errada (impostos não são negativos por
            # natureza) — sinaliza para revisão em vez de reportar confuso.
            if isinstance(tax.amount, float) and tax.amount < 0:
                tax.amount = READ_ERROR_SENTINEL
            result.append(tax)
        return result


# ---------------------------------------------------------------------------
# Extração de itens de tabela
# ---------------------------------------------------------------------------

def _extract_line_items_from_tables(tables: list[pd.DataFrame]) -> list[LineItem]:
    """
    Percorre os DataFrames extraídos pelo docling em busca de tabelas de itens.
    Identifica a tabela de produtos/serviços pelo padrão das colunas.
    """
    items: list[LineItem] = []

    desc_aliases = {"descri", "produto", "servi", "item", "discrimina", "histor"}
    qty_aliases = {"qtd", "quant", "qty", "amount"}
    price_aliases = {"unit", "pre", "price", "valor unit"}
    total_aliases = {"total", "valor total", "vlr total", "subtotal"}

    def _col_match(col_name: str, aliases: set[str]) -> bool:
        normalized = str(col_name).lower().strip()
        return any(a in normalized for a in aliases)

    for df in tables:
        cols = list(df.columns)
        col_str = " ".join(str(c).lower() for c in cols)

        # Heurística: tabela de itens tem coluna de descrição + pelo menos mais 1 coluna de valor
        has_desc = any(_col_match(c, desc_aliases) for c in cols)
        has_value = any(_col_match(c, price_aliases | total_aliases) for c in cols)

        if not (has_desc or (has_value and len(cols) >= 3)):
            continue

        desc_col = next((c for c in cols if _col_match(c, desc_aliases)), cols[0])
        qty_col = next((c for c in cols if _col_match(c, qty_aliases)), None)
        price_col = next((c for c in cols if _col_match(c, price_aliases)), None)
        total_col = next((c for c in cols if _col_match(c, total_aliases)), None)

        # Padrões de linhas de ruído que não são itens reais
        _noise = re.compile(
            r"^(TOTAL|SUBTOTAL|HISTÓRICO|HISTÓRIA|MÊS|ANO|UF|HP|HFP|HR|\d{1,2}/\d{2,4}|"
            r"OUT|SET|AGO|JUL|JUN|MAI|ABR|MAR|FEV|JAN|DEZ|NOV|$)",
            re.IGNORECASE,
        )

        for _, row in df.iterrows():
            desc_val = str(row.get(desc_col, "")).strip()
            if not desc_val or desc_val.lower() in ("nan", "none", ""):
                continue
            if _noise.match(desc_val):
                continue
            # Itens puramente numéricos não são descrições
            if re.match(r"^[\d.,\s]+$", desc_val):
                continue

            qty = None
            if qty_col and str(row.get(qty_col, "")).strip():
                try:
                    qty = parse_currency(str(row[qty_col]))
                except Exception:
                    pass

            # Fallback 1: Docling às vezes inverte header/dado em tabelas NF-e
            # (ex: o valor numérico da quantidade fica no nome da coluna, não
            # na célula). Detecta pelo fato de qty_col ser "QUANT..." mas ter
            # valor não-numérico na célula → tenta parse do próprio nome.
            if qty is None and qty_col and _col_match(str(qty_col), qty_aliases):
                candidate = parse_currency(str(qty_col))
                # Aceita apenas se o valor for plausível para uma quantidade
                # (> 0 e < 1.000.000) e o nome der um float razoável
                if candidate is not None and 0 < candidate < 1_000_000:
                    qty = candidate

            # Fallback 2: "V.TOTAL/V.UNITARIO <qty>" no nome de outra coluna.
            # Docling às vezes desloca o valor para o nome da coluna de total/
            # unitário — ex: coluna "V.TOTAL 259,3100" com valor real=40.299,37.
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

            # Unidade de medida: coluna "UN"/"UNID"/"UNIDADE" etc.
            un_col = next(
                (c for c in cols if str(c).strip().upper() in ("UN", "UNID", "UNIDADE", "UNIT", "UND")),
                None,
            )
            unit_val_str = str(row.get(un_col, "")).strip() if un_col else ""
            unit = unit_val_str if unit_val_str and unit_val_str.lower() not in ("nan", "none") else None

            unit_price = None
            if price_col and str(row.get(price_col, "")).strip():
                unit_price = parse_currency(str(row[price_col]))

            total_val = None
            if total_col and str(row.get(total_col, "")).strip():
                total_val = parse_currency(str(row[total_col]))

            items.append(
                LineItem(
                    description=desc_val,
                    unit=unit,
                    quantity=qty,
                    unit_price=unit_price,
                    total=total_val,
                )
            )

    return items


def _extract_line_items_from_text(text: str, layout: InvoiceLayout) -> list[LineItem]:
    """
    Fallback: tenta extrair itens a partir de padrões de texto quando
    a extração via tabela não encontrou itens.
    """
    items: list[LineItem] = []

    if layout == InvoiceLayout.NFSE:
        # NFS-e frequentemente lista serviços em parágrafo
        m = re.search(
            r"Discrimina[çc][ãa]o\s+(?:dos\s+)?Servi[çc]os?\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            desc = re.sub(r"\s+", " ", m.group(1)).strip()
            total = _FieldExtractor.total(text, layout)
            items.append(LineItem(description=desc[:500], total=total))

    elif layout == InvoiceLayout.BOLETO:
        m = re.search(
            r"(?:Descri[çc][ãa]o|Hist[oó]rico|Refer[êe]ncia)\s*[:\-]?\s*(.+?)(?=\n|\Z)",
            text, re.IGNORECASE,
        )
        if m:
            items.append(LineItem(description=m.group(1).strip()[:300]))

    return items


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class DeterministicParser:

    def parse(
        self,
        content: DocumentContent,
        layout: InvoiceLayout,
        prefilled: dict | None = None,
    ) -> Invoice:
        """
        Extrai campos da fatura de forma determinística (regex).

        Quando `prefilled` é fornecido (dict vindo do DoclingExtractor), os campos
        já extraídos são usados diretamente e o regex correspondente é ignorado.
        Isso permite que o DoclingExtractor seja a camada primária e o regex seja
        apenas o fallback para o que o Docling não detectou.

        Retorna um Invoice com confidence_score calculado.
        """
        text = content.full_text
        ext = _FieldExtractor
        pre = prefilled or {}

        # -- Chave de acesso / código de barras (extraído antes do número para NF-e) --
        access_key = pre.get("access_key")
        barcode = None
        barcode_line = None
        if layout == InvoiceLayout.NFE and not access_key:
            access_key = extract_access_key(text)

        # Fallback: chave de acesso como título de seção do Docling
        # (ex: "3526 0110 7324 4000 0197 55 000 000027932..." como heading)
        if layout == InvoiceLayout.NFE and not access_key:
            for sec_key in content.section_map:
                digits_only = re.sub(r"\s", "", sec_key)
                if len(digits_only) >= 44 and digits_only[:44].isdigit():
                    access_key = digits_only[:44]
                    break

        # -- Campos base --
        # Para NF-e, o número da nota vem preferencialmente do DoclingExtractor
        # (que extrai o campo impresso no documento com zeros à esquerda preservados);
        # a chave de acesso é usada como fallback quando nenhum outro extrator o encontrou.
        if layout == InvoiceLayout.NFE and access_key and len(access_key) == 44:
            key_number = access_key[25:34]  # 9 dígitos com zeros à esquerda
            invoice_number = (
                pre.get("invoice_number")
                or ext.invoice_number(text, layout)
                or str(int(key_number))
            )
        else:
            invoice_number = pre.get("invoice_number") or ext.invoice_number(text, layout)

        issue_date = pre.get("issue_date") or ext.issue_date(text, layout)
        due_date = pre.get("due_date") or ext.due_date(text, layout)

        # Supplier: se DoclingExtractor já extraiu nome/cnpj, usa como base e
        # complementa com regex (endereço, e-mail, telefone, IE).
        if pre.get("supplier_name") or pre.get("supplier_cnpj"):
            supplier = Party(
                name=pre.get("supplier_name"),
                cnpj=pre.get("supplier_cnpj"),
            )
            regex_supplier = ext.supplier(text, layout)
            if regex_supplier:
                if not supplier.name and regex_supplier.name:
                    supplier.name = regex_supplier.name
                if not supplier.cnpj and regex_supplier.cnpj:
                    supplier.cnpj = regex_supplier.cnpj
                if regex_supplier.address and (
                    regex_supplier.address.city or regex_supplier.address.state
                ):
                    supplier.address = regex_supplier.address
                if regex_supplier.email:
                    supplier.email = regex_supplier.email
                if regex_supplier.phone:
                    supplier.phone = regex_supplier.phone
        else:
            supplier = ext.supplier(text, layout)

        # Customer: mesma lógica do supplier.
        if pre.get("customer_name") or pre.get("customer_cnpj"):
            customer = Party(
                name=pre.get("customer_name"),
                cnpj=pre.get("customer_cnpj"),
            )
            regex_customer = ext.customer(text, layout)
            if regex_customer:
                if not customer.name and regex_customer.name:
                    customer.name = regex_customer.name
                if not customer.cnpj and regex_customer.cnpj:
                    customer.cnpj = regex_customer.cnpj
                if regex_customer.address and (
                    regex_customer.address.city or regex_customer.address.state
                ):
                    customer.address = regex_customer.address
                if regex_customer.email:
                    customer.email = regex_customer.email
                if regex_customer.phone:
                    customer.phone = regex_customer.phone
        else:
            customer = ext.customer(text, layout)

        # DoclingExtractor via kv_pairs/section_map tem prioridade para cidade/UF —
        # substitui o valor regex que pode ter sofrido contaminação entre seções.
        if supplier:
            if pre.get("supplier_city"):
                supplier.address.city = pre["supplier_city"]
            if pre.get("supplier_state"):
                supplier.address.state = pre["supplier_state"]
        if customer:
            if pre.get("customer_city"):
                customer.address.city = pre["customer_city"]
            if pre.get("customer_state"):
                customer.address.state = pre["customer_state"]

        # Total: DoclingExtractor via kv_pairs/spatial tem prioridade; labels regex como fallback.
        total = pre.get("total") or ext.total_via_labels(text, layout)
        subtotal = ext.subtotal(text)
        taxes = ext.taxes(text, layout)

        # Fallback de data de vencimento: padrão "DD/MM/AAAA \n R$ valor" sem rótulo,
        # comum quando o rótulo (ex: "VENCIMENTO") é um elemento gráfico não
        # capturado como texto pelo docling.
        if layout == InvoiceLayout.NFE and not due_date:
            fallback_due_date, _ = extract_date_before_currency(text)
            if fallback_due_date and fallback_due_date != issue_date:
                due_date = fallback_due_date

        # Âncora "MTE\d{7}": código de instalação estável presente em diversos
        # sub-layouts de fatura de energia da Enel SP (com ou sem assinatura
        # hex, com datas numéricas ou abreviadas em português), sempre seguido
        # do valor total e da data de vencimento SEM rótulo explícito. Tentado
        # ANTES do fallback de "maior valor monetário" porque bases de cálculo
        # de imposto no corpo do documento costumam ser numericamente maiores
        # que o total real, o que faria aquele fallback escolher o valor errado.
        if layout == InvoiceLayout.NFE and (not total or not due_date):
            mte_m = re.search(r"MTE\d{7}", text)
            if mte_m:
                window_text = text[mte_m.end(): mte_m.end() + 150]
                if not total:
                    val_m = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})", window_text)
                    if val_m:
                        total = parse_currency(val_m.group(1))
                if not due_date:
                    numeric_in_window = extract_all_dates(window_text)
                    ptbr_in_window = extract_all_dates_ptbr_abbrev(window_text)
                    candidate = (numeric_in_window or ptbr_in_window or [None])[0]
                    if candidate:
                        due_date = candidate

        # Layout "assinatura hexadecimal" (algumas faturas de energia da Enel
        # SP): não traz chave de acesso convencional nem rótulos de campo —
        # o número da nota aparece apenas como valor solto, posicionalmente
        # estável em relação à própria assinatura hex.
        if has_hex_signature(text) and not invoice_number:
            invoice_number = extract_number_after_hex_signature(text)

        # Último recurso para o total: maior valor monetário do documento.
        # Arriscado (pode pegar base de cálculo de imposto), por isso só roda
        # se nem rótulo nem âncora MTE encontraram nada.
        if not total:
            values = extract_all_currencies(text)
            total = max(values) if values else None

        if layout == InvoiceLayout.BOLETO:
            barcode = extract_boleto_barcode(text)
            barcode_line = extract_boleto_line(text)

            # Boleto com chave de acesso NF-e/NF3E embarcada (44 dígitos, modelo 55/66):
            # extrai CNPJ do emitente (pos 6-19) e preenche o fornecedor via cadastro.
            if barcode and len(barcode) == 44 and re.match(r"^\d{44}$", barcode):
                model_from_key = barcode[20:22]
                if model_from_key in ("55", "66"):
                    nfe_cnpj = barcode[6:20]
                    known_nfe = lookup_supplier(nfe_cnpj)
                    if known_nfe:
                        if supplier is None:
                            supplier = Party()
                        if not supplier.name:
                            supplier.name = known_nfe.get("name")
                        if not supplier.cnpj:
                            supplier.cnpj = known_nfe.get("cnpj")
                        if not supplier.address.city:
                            supplier.address.city = known_nfe.get("city")
                            supplier.address.state = known_nfe.get("state")

        # -- Itens --
        line_items = _extract_line_items_from_tables(content.tables)
        if not line_items:
            line_items = _extract_line_items_from_text(text, layout)

        # -- Série (NF-e) --
        series = None
        if layout == InvoiceLayout.NFE:
            m = re.search(r"S[ée]rie\s*[:\s]+(\d+)", text, re.IGNORECASE)
            series = m.group(1) if m else None

        # Para NF-e, o CNPJ do emitente está codificado na chave de acesso (pos 6-19)
        # — é a fonte mais confiável, pois sobrevive a OCR e a glitches de fonte/encoding.
        # Esta parte depende de uma chave de acesso válida ter sido encontrada.
        emitter_cnpj: Optional[str] = None
        if layout == InvoiceLayout.NFE and access_key and len(access_key) == 44:
            emitter_cnpj_digits = access_key[6:20]
            emitter_cnpj = format_cnpj_digits(emitter_cnpj_digits)

            if supplier is None:
                supplier = Party(cnpj=emitter_cnpj)
            elif not supplier.cnpj:
                supplier.cnpj = emitter_cnpj

            # Cadastro de fornecedores conhecidos: preenche nome/endereço quando a
            # extração via regex falhar (ex: layout sem "RECEBEMOS DE"/"EMITENTE",
            # ou texto corrompido por OCR/encoding).
            known = lookup_supplier(emitter_cnpj_digits)
            if known:
                if not supplier.name:
                    supplier.name = known.get("name")
                if not supplier.address.city and not supplier.address.state:
                    supplier.address.city = known.get("city")
                    supplier.address.state = known.get("state")

        # CNPJ do destinatário via rótulo: independente da chave de acesso —
        # vários layouts (ex: faturas Enel SP sem chave de acesso convencional)
        # ainda imprimem "CNPJ/CPF"/"CPF/CNPJ" com o CNPJ legível no texto.
        if layout == InvoiceLayout.NFE:
            dest_cnpj = (
                extract_cnpj_after_label(text, r"CNPJ\s*/\s*CPF")
                or extract_cnpj_after_label(text, r"CPF\s*/\s*CNPJ")
            )
            if dest_cnpj:
                if customer is None:
                    customer = Party(cnpj=dest_cnpj)
                else:
                    customer.cnpj = dest_cnpj
                # Se o CNPJ encontrado coincide com o do emitente, provavelmente
                # extract_cnpj_after_label capturou o label do emitente — busca alternativa.
                if emitter_cnpj and customer.cnpj == emitter_cnpj:
                    all_cnpjs = extract_all_cnpjs(text)
                    alt = next((c for c in all_cnpjs if c != emitter_cnpj), None)
                    if alt:
                        customer.cnpj = alt
            elif emitter_cnpj and customer and customer.cnpj == emitter_cnpj:
                # Fallback: pega o segundo CNPJ distinto do emitente
                all_cnpjs = extract_all_cnpjs(text)
                alt = next((c for c in all_cnpjs if c != emitter_cnpj), None)
                if alt and customer:
                    customer.cnpj = alt

            # Cadastro de clientes conhecidos: preenche o nome pela raiz do CNPJ
            # (8 primeiros dígitos) quando a extração via regex não encontrar o
            # nome (ex: layout sem rótulo "DESTINATÁRIO").
            if customer and customer.cnpj and not customer.name:
                known_name = lookup_customer_name(customer.cnpj)
                if known_name:
                    customer.name = known_name

        # -- Métricas de energia para BI (apenas faturas de energia elétrica) --
        # Extração puramente direta: nenhum valor aqui é calculado a partir de
        # outro campo. Quando o dado não está impresso na fatura (ex: fator de
        # potência), o campo correspondente fica null — ausência confirmada,
        # não falha de extração. Ver EnergyMetrics para detalhes de cada campo.
        energy = EnergyMetrics(
            consumer_unit=extract_consumer_unit(text),
            client_number=extract_client_number(text),
            measured_demand_offpeak_kw=extract_measured_demand_kw(text),
            demand_overage_value=extract_demand_overage_value(text),
        )
        energy.consumption_peak_kwh, energy.consumption_offpeak_kwh = extract_consumption_kwh(text)
        reactive = extract_reactive_excess(text)
        energy.reactive_energy_excess_peak_kwh = reactive["peak_kwh"]
        energy.reactive_energy_excess_offpeak_kwh = reactive["offpeak_kwh"]
        energy.reactive_penalty_peak_value = reactive["peak_value"]
        energy.reactive_penalty_offpeak_value = reactive["offpeak_value"]

        # DANF3E / Tarifa Azul: demanda ponta e fora-ponta separadas
        peak_demand, fp_demand = extract_tarifa_azul_demand_nf3e(text)
        if peak_demand is not None:
            energy.measured_demand_peak_kw = peak_demand
        if fp_demand is not None:
            energy.measured_demand_offpeak_kw = fp_demand

        # NF-e de comercializadora: quantidade em MWh da linha de produto
        # (substitui consumption_offpeak_kwh como proxy de consumo total)
        if layout == InvoiceLayout.NFE and energy.consumption_offpeak_kwh is None:
            mwh = extract_energy_quantity_mwh(text)
            if mwh is not None:
                energy.consumption_offpeak_kwh = mwh

        # Fallback via line_items: item com unidade MWh e quantidade numérica
        # (cobre casos em que o valor está na tabela Docling mas não no texto plano)
        if layout == InvoiceLayout.NFE and energy.consumption_offpeak_kwh is None:
            for item in line_items:
                unit_norm = re.sub(r"[\s/]", "", (item.unit or "")).upper()
                if "MWH" in unit_norm and item.quantity is not None:
                    q = item.quantity
                    if isinstance(q, (int, float)) and 0 < q < 99999:
                        energy.consumption_offpeak_kwh = q
                        break

        invoice = Invoice(
            invoice_number=invoice_number,
            invoice_layout=layout,
            access_key=access_key,
            series=series,
            issue_date=issue_date,
            due_date=due_date,
            supplier=supplier or Party(),
            customer=customer or Party(),
            line_items=line_items,
            taxes=taxes,
            energy=energy,
            subtotal=subtotal,
            total=total,
            bank_slip_barcode=barcode,
            bank_slip_line=barcode_line,
            source_file=content.source_path,
            parsing_method=ParsingMethod.DETERMINISTIC,
            raw_text=text,
        )

        # confidence_score_initial registra o score do passo determinístico,
        # mesmo quando ele não muda mais depois (faturas resolvidas sem LLM
        # ficam com os dois campos idênticos — é o comportamento esperado).
        score = calculate_confidence(invoice)
        invoice.confidence_score_initial = score
        invoice.confidence_score = score

        logger.debug(
            f"Parser determinístico finalizado — "
            f"layout={layout} confidence={invoice.confidence_score:.2f}"
        )
        return invoice


# ---------------------------------------------------------------------------
# Score de confiança
# ---------------------------------------------------------------------------
# Pública (sem underscore) porque também é usada pelo pipeline para recalcular
# o score depois do LLM fallback — caso contrário o JSON final mostraria o
# score "congelado" de antes do enrich, mesmo com os campos já preenchidos.

def calculate_confidence(invoice: Invoice) -> float:
    score = 0.0

    if invoice.invoice_number:
        score += _WEIGHTS["invoice_number"]
    if invoice.issue_date:
        score += _WEIGHTS["issue_date"]
    if invoice.supplier and invoice.supplier.name:
        score += _WEIGHTS["supplier_name"]
    if invoice.supplier and (invoice.supplier.cnpj or invoice.supplier.cpf):
        score += _WEIGHTS["supplier_cnpj"]
    if invoice.customer and invoice.customer.name:
        score += _WEIGHTS["customer_name"]
    if invoice.customer and (invoice.customer.cnpj or invoice.customer.cpf):
        score += _WEIGHTS["customer_cnpj"]
    # invoice.total pode ser a sentinela READ_ERROR_SENTINEL (string) em vez
    # de número — nesse caso não conta para o score (não temos um valor
    # numérico válido), mas não deve quebrar a comparação.
    if isinstance(invoice.total, (int, float)) and invoice.total > 0:
        score += _WEIGHTS["total"]
    if invoice.line_items:
        score += _WEIGHTS["line_items"]
    if invoice.due_date:
        score += _WEIGHTS["due_date"]

    return round(min(score, 1.0), 4)


def get_missing_required_fields(invoice: Invoice) -> list[str]:
    """Retorna lista de campos obrigatórios ausentes, para passar ao LLM fallback."""
    missing: list[str] = []
    if not invoice.invoice_number:
        missing.append("invoice_number — número da fatura/nota fiscal")
    if not invoice.issue_date:
        missing.append("issue_date — data de emissão (formato YYYY-MM-DD)")
    if not invoice.supplier or not invoice.supplier.name:
        missing.append("supplier_name — nome/razão social do emitente ou fornecedor")
    if not invoice.supplier or not (invoice.supplier.cnpj or invoice.supplier.cpf):
        missing.append("supplier_cnpj — CNPJ ou CPF do emitente")
    if not isinstance(invoice.total, (int, float)) or invoice.total <= 0:
        missing.append("total — valor total da fatura (número decimal, ex: 1234.56)")
    return missing

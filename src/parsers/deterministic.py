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
    extract_energy_quantity_mwh,
    extract_cpf,
    extract_currency,
    extract_date,
    extract_date_before_currency,
    extract_labeled_value,
    extract_lines_after_label,
    extract_number_after_hex_signature,
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
    Invoice,
    InvoiceLayout,
    LineItem,
    Party,
    ParsingMethod,
    READ_ERROR_SENTINEL,
    Tax,
)
from src.extractors.tables import extract_line_items_from_tables as _extract_line_items_from_tables
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

        # Estado: somente siglas válidas de UF brasileiras
        state = extract_br_state(block_text)
        if not state:
            inline_m = re.search(
                r"\b[A-ZÀ-Ú][A-ZÀ-Ú\s]{2,38}?\s*/\s*([A-Z]{2})\b",
                block_text.upper(),
            )
            if inline_m:
                state = extract_br_state(inline_m.group(1)) or state

        return Party(
            name=name_line.strip() if name_line else None,
            cnpj=cnpj,
            cpf=cpf,
            ie=ie_m.group(1).strip() if ie_m else None,
            address=Address(state=state),
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
# Fallback textual: extração de itens de NF-e de energia sem tabela Docling
# ---------------------------------------------------------------------------

# Mapeamento label → (descrição canônica, unidade) para itens de energia.
# Ordenado do mais específico para o mais genérico: padrões mais específicos
# vêm primeiro para que claimed_lines impeça o genérico de re-capturar a linha.
# Unidade vazia ("") indica item sem qty/unit_price — só total (ex: CIP, tributos).
_ENERGY_ITEM_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # CEMIG NF-e / NF3E — Consumo HP/HFP
    (re.compile(r"Consumo\s+de\s+energia\s+el[eé]trica\s+HFP\b", re.IGNORECASE),
     "Consumo de energia elétrica HFP", "kWh"),
    (re.compile(r"Consumo\s+de\s+energia\s+el[eé]trica\s+HP\b", re.IGNORECASE),
     "Consumo de energia elétrica HP",  "kWh"),
    (re.compile(r"Energia\s+Ativa\s+HFP\b", re.IGNORECASE),
     "Energia Ativa HFP", "kWh"),
    (re.compile(r"Energia\s+Ativa\s+HP\b", re.IGNORECASE),
     "Energia Ativa HP", "kWh"),
    # Energia terceirizada — DISTRIBUIDORA_MLE
    (re.compile(r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HFP\b", re.IGNORECASE),
     "Energia Terc Comercializad HFP", "kWh"),
    (re.compile(r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HP\b", re.IGNORECASE),
     "Energia Terc Comercializad HP",  "kWh"),
    # TUSD e TE com HP/HFP
    (re.compile(r"TUSD\s+(?:HP|Hora\s+de\s+Ponta)\b", re.IGNORECASE),
     "TUSD HP", "kWh"),
    (re.compile(r"TUSD\s+(?:HFP|F\.?\s*Ponta|Hora\s+Fora\s+de\s+Ponta)\b", re.IGNORECASE),
     "TUSD HFP", "kWh"),
    (re.compile(r"\bTE\s+(?:HP|Hora\s+de\s+Ponta)\b", re.IGNORECASE),
     "TE HP", "kWh"),
    (re.compile(r"\bTE\s+(?:HFP|F\.?\s*Ponta|Hora\s+Fora\s+de\s+Ponta)\b", re.IGNORECASE),
     "TE HFP", "kWh"),
    # Demanda — mais específico primeiro para não conflitar com genérico
    (re.compile(r"Demanda\s+Ativa\b(?!\s+Ultrapassagem)", re.IGNORECASE),
     "Demanda Ativa", "kW"),
    (re.compile(r"Demanda\s+(?:Ativa\s+)?Ultrapassagem\b", re.IGNORECASE),
     "Demanda Ultrapassagem", "kW"),
    (re.compile(r"ULTRAPASSAGEM\s+DEMANDA\b", re.IGNORECASE),
     "Ultrapassagem Demanda", "kW"),
    (re.compile(r"Demanda\s+Reativa\s+Excedente\b", re.IGNORECASE),
     "Demanda Reativa Excedente", "kVAR"),
    # Consumo Reativo — antes do Consumo Ativo
    (re.compile(r"Consumo\s+Reativo\s+Exc\w*\.?\s+(?:Na\s+)?Ponta\b", re.IGNORECASE),
     "Consumo Reativo Exc. Ponta", "kVARh"),
    (re.compile(r"Consumo\s+Reativo\s+Exc\w*\.?\s+Fora\s+(?:da?\s+)?Ponta\b", re.IGNORECASE),
     "Consumo Reativo Exc. Fora de Ponta", "kVARh"),
    # Consumo Ativo — Enel SP com sufixo TE/TUSD
    (re.compile(r"CONSUMO\s+ATIVO\s+PONTA\s+(?:TE|TUSD)\b", re.IGNORECASE),
     "Consumo Ativo Ponta", "kWh"),
    (re.compile(r"CONSUMO\s+ATIVO\s+F\.?\s*PONTA\s+(?:TE|TUSD)\b", re.IGNORECASE),
     "Consumo Ativo Fora de Ponta", "kWh"),
    # Consumo Ativo — COPEL/CEEE/genérico (sem TUSD/TE; suporta "Na Ponta")
    (re.compile(r"Consumo\s+Ativo\s+(?:Na\s+)?Ponta\b", re.IGNORECASE),
     "Consumo Ativo Ponta", "kWh"),
    (re.compile(r"Consumo\s+Ativo\s+(?:Fora\s+(?:da?\s+)?Ponta|F\.?\s*Ponta)\b", re.IGNORECASE),
     "Consumo Ativo Fora de Ponta", "kWh"),
    # Demanda genérica (após os específicos — lookahead exclui Ativa/Reativa/Máxima)
    (re.compile(r"\bDEMANDA\b(?!\s+(?:ATIVA|REATIVA|CONTRAT|M[AÁ]XIMA))", re.IGNORECASE),
     "Demanda", "kW"),
    # UFER
    (re.compile(r"UFER\s+PONTA\b", re.IGNORECASE), "UFER Ponta", "kWh"),
    (re.compile(r"UFER\s+(?:F\.?\s*PONTA|HFP)\b", re.IGNORECASE), "UFER Fora de Ponta", "kWh"),
    # CIP e tributos sem qty/unit_price (unidade "" → apenas total)
    (re.compile(r"Contrib\.?\s+Ilum\.?\s+P[úu]bl", re.IGNORECASE),
     "Contrib. Iluminação Pública Municipal", ""),
    (re.compile(r"ICMS\s+Subven[çc][ãa]o\b", re.IGNORECASE),
     "ICMS Subvenção", ""),
    (re.compile(r"\bTributo\s+Federal\b", re.IGNORECASE),
     "Tributo Federal", ""),
    # Comercializadora — Energia Elétrica MWh (fallback de último recurso)
    (re.compile(r"ENERGIA\s+EL[ÉE]TRICA\b", re.IGNORECASE),
     "Energia Elétrica", "MWh"),
]

_NUM = r"-?[\d.,]+"   # número (possivelmente negativo)
_BANDEIRA_RE = re.compile(r"^BANDEIRA\s+(\w+)\s*$", re.IGNORECASE)
_BANDEIRA_INLINE_RE = re.compile(r"BANDEIRA\s+(\w+)", re.IGNORECASE)

# Seções de texto que NÃO contêm itens de faturamento
_SKIP_SECTION_HEADERS = frozenset(["gráficos", "tarifas aplicadas", "graficos"])


def _parse_compound_description_line(line_idx: int, lines: list[str]) -> list[LineItem]:
    """
    Parseia uma linha composta onde múltiplas descrições de item estão concatenadas
    (sem dígitos), seguida de linhas de valores.

    Suporta dois formatos de bloco de valores:
      Simples  : cada item tem qty, price, total em linhas individuais separadas.
      Agrupado : quantidades empilhadas em uma linha; totais agrupados em outra linha
                 com múltiplos valores; preços em linhas individuais.
    """
    line = lines[line_idx].strip()

    # Extrai itens em ordem textual (esquerda → direita), rastreando BANDEIRA
    ordered_items: list[tuple[str, str, str | None]] = []
    current_bandeira: str | None = None
    pos = 0
    while pos < len(line):
        # Verifica marcador de BANDEIRA na posição atual
        bm = re.match(r"BANDEIRA\s+(\w+)", line[pos:], re.IGNORECASE)
        if bm:
            current_bandeira = bm.group(1).upper()
            pos += bm.end()
            continue

        # Encontra o padrão mais próximo a partir de pos
        best_start = len(line) + 1
        best_m = None
        best_data: tuple[str, str] | None = None
        for pat, cdesc, cunit in _ENERGY_ITEM_PATTERNS:
            m = pat.search(line, pos)
            if m and m.start() < best_start:
                best_start = m.start()
                best_m = m
                best_data = (cdesc, cunit)

        if best_m is None:
            break

        # Atualiza bandeira com qualquer marcador entre pos e o match
        between = line[pos:best_m.start()]
        for bm2 in _BANDEIRA_INLINE_RE.finditer(between):
            current_bandeira = bm2.group(1).upper()

        cdesc, cunit = best_data
        ordered_items.append((cdesc, cunit, current_bandeira))
        pos = best_m.end()

    if len(ordered_items) < 3:
        return []

    # Separa itens com unidade dos itens "só total" (CIP, ICMS Sub, Tributo)
    unit_items = [(cd, cu, ab) for cd, cu, ab in ordered_items if cu != ""]
    total_only_items = [(cd, ab) for cd, cu, ab in ordered_items if cu == ""]

    # Coleta linhas de valores após a linha composta (para quando encontrar
    # cabeçalho de texto puro sem dígitos)
    value_lines: list[list[float]] = []
    for j in range(line_idx + 1, min(line_idx + 60, len(lines))):
        ln = lines[j].strip()
        if not ln:
            continue
        if re.match(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ]{4,}", ln) and not re.search(r"\d", ln):
            break
        nums: list[float] = []
        for tok in re.split(r"\s+", ln):
            n = parse_currency(tok)
            if n is not None:
                nums.append(n)
        if nums:
            value_lines.append(nums)

    if not value_lines:
        return []

    results: list[tuple[str, str, str | None, float | None, float | None, float | None]] = []
    extra_totals: list[float] = []

    vl_idx = 0
    item_idx = 0

    while item_idx < len(unit_items) and vl_idx < len(value_lines):
        vl = value_lines[vl_idx]
        vl_idx += 1

        if len(vl) == 1:
            # Linha de valor único → qty do item atual (modo simples)
            cd, cu, ab = unit_items[item_idx]
            qty = vl[0]
            price = total = None
            while vl_idx < len(value_lines):
                next_vl = value_lines[vl_idx]
                if len(next_vl) == 1:
                    if price is None:
                        price = next_vl[0]
                        vl_idx += 1
                    elif total is None:
                        total = next_vl[0]
                        vl_idx += 1
                        break
                    else:
                        break
                else:
                    break  # linha multi-valor → pertence ao próximo grupo
            results.append((cd, cu, ab, qty, price, total))
            item_idx += 1

        else:
            # Linha multi-valor → quantidades empilhadas de N itens consecutivos
            N_vals = len(vl)
            stacked = unit_items[item_idx: item_idx + N_vals]
            N = len(stacked)
            qtys = vl[:N]

            prices: list[float | None] = [None] * N
            totals: list[float | None] = [None] * N
            price_idx = 0
            total_idx = 0
            combined_totals = False

            while (total_idx < N or price_idx < N) and vl_idx < len(value_lines):
                next_vl = value_lines[vl_idx]

                if len(next_vl) == 1:
                    if combined_totals:
                        # Modo B: totais já agrupados → lê preços restantes
                        while price_idx < N and prices[price_idx] is not None:
                            price_idx += 1
                        if price_idx < N:
                            prices[price_idx] = next_vl[0]
                            price_idx += 1
                        vl_idx += 1
                    else:
                        # Modo A: alternado [price, total, price, total, …]
                        if price_idx == total_idx:
                            prices[price_idx] = next_vl[0]
                            price_idx += 1
                        else:
                            totals[total_idx] = next_vl[0]
                            total_idx += 1
                        vl_idx += 1
                else:
                    # Linha multi-valor durante o grupo → linha de totais agrupados
                    needed = N - total_idx
                    for k in range(min(needed, len(next_vl))):
                        totals[total_idx + k] = next_vl[k]
                    total_idx = N
                    if len(next_vl) > needed:
                        extra_totals = list(next_vl[needed:])
                    vl_idx += 1
                    combined_totals = True
                    if price_idx >= N:
                        break

            for k, (cd, cu, ab) in enumerate(stacked):
                qty = qtys[k] if k < len(qtys) else None
                if cu == "MWh" and isinstance(qty, float) and qty > 99_999:
                    qty = None
                results.append((cd, cu, ab, qty, prices[k], totals[k]))

            item_idx += N

    # Itens sem unidade: totais vêm da linha de totais agrupados ou da maior linha
    if total_only_items:
        if not extra_totals:
            for vl in value_lines:
                if len(vl) >= len(total_only_items):
                    extra_totals = list(vl[-len(total_only_items):])
                    break
        if extra_totals:
            for k, (cd, ab) in enumerate(total_only_items):
                if k < len(extra_totals):
                    results.append((cd, "", ab, None, None, abs(extra_totals[k])))

    # Converte para LineItem
    items: list[LineItem] = []
    for cd, cu, ab, qty, price, total in results:
        if qty is None and total is None:
            continue
        desc = f"{cd} (Bandeira {ab.title()})" if ab else cd
        items.append(LineItem(
            description=desc,
            unit=cu or None,
            quantity=abs(qty) if isinstance(qty, float) else qty,
            unit_price=price,
            total=total,
        ))
    return items


def _extract_energy_items_from_text(text: str) -> list[LineItem]:
    """
    Extrai itens de fatura de energia a partir do texto plano quando o
    Docling não capturou a tabela de produtos/serviços.

    Formatos suportados:
      Inline  : <RÓTULO[(UNID)]>  <QTY>  <UNIT_PRICE>  <TOTAL>  (tudo na mesma linha)
      Vertical: <RÓTULO> / <UNID> / <QTY> / <UNIT_PRICE> / <TOTAL>  (linhas separadas)
      Total-only: <RÓTULO>  <TOTAL>  (itens sem qty/unit_price: CIP, ICMS Subvenção)

    Bandeiras tarifárias (BANDEIRA VERMELHA / AMARELA) são rastreadas por linha
    para distinguir itens com o mesmo rótulo mas tarifa diferente.
    """
    items: list[LineItem] = []
    seen_descs: set[tuple] = set()   # (canonical_desc, bandeira) — deduplicação
    claimed_lines: set[int] = set()  # evita que padrão genérico recapture linha já extraída

    lines = text.splitlines()

    # Pré-computa o contexto de bandeira vigente em cada linha
    bandeira_per_line: list[str | None] = [None] * len(lines)
    _current_bandeira: str | None = None
    for idx, ln in enumerate(lines):
        m = _BANDEIRA_RE.match(ln.strip())
        if m:
            _current_bandeira = m.group(1).upper()
        bandeira_per_line[idx] = _current_bandeira

    # Marca linhas em seções não-faturamento (GRÁFICOS, TARIFAS APLICADAS)
    skip_lines: set[int] = set()
    in_skip_section = False
    for idx, ln in enumerate(lines):
        stripped_lower = ln.strip().lower()
        if stripped_lower in _SKIP_SECTION_HEADERS:
            in_skip_section = True
            skip_lines.add(idx)
            continue
        # Próximo cabeçalho de seção em caixa alta sem dígitos encerra a seção de skip
        if in_skip_section:
            if re.match(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ]{4,}", ln.strip()) and not re.search(r"\d", ln):
                if stripped_lower not in _SKIP_SECTION_HEADERS:
                    in_skip_section = False
                    # Não adiciona essa linha ao skip (pode ser início de nova seção válida)
                    continue
            skip_lines.add(idx)

    # Detecta e parseia linhas compostas ANTES do loop principal
    # (linhas sem valores numéricos válidos que contenham 3+ padrões de energia).
    # Usa parse_currency para distinguir valores reais de códigos/datas ("083217793-30/10/19").
    for idx, ln in enumerate(lines):
        if idx in skip_lines:
            continue
        # Linha composta não contém valores numéricos "parseáveis" (qty/price/total)
        if any(parse_currency(tok) is not None
               for tok in re.split(r"\s+", ln.strip()) if tok):
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        match_count = sum(1 for pat, _, _ in _ENERGY_ITEM_PATTERNS if pat.search(stripped))
        if match_count < 3:
            continue
        compound_items = _parse_compound_description_line(idx, lines)
        if compound_items:
            for ci in compound_items:
                if ci.quantity is not None or ci.total is not None:
                    items.append(ci)
                    desc_key = (ci.description, None)
                    seen_descs.add(desc_key)
            claimed_lines.add(idx)

    for pattern, canonical_desc, canonical_unit in _ENERGY_ITEM_PATTERNS:
        for i, line in enumerate(lines):
            if i in claimed_lines:
                continue
            if i in skip_lines:
                continue
            if not pattern.search(line.strip()):
                continue

            active_bandeira = bandeira_per_line[i]
            desc_key = (canonical_desc, active_bandeira)

            if desc_key in seen_descs:
                if active_bandeira is None:
                    break   # item sem bandeira — só existe uma ocorrência; para busca
                continue    # item com bandeira — pode aparecer de novo com bandeira diferente

            qty = unit_price = total = None
            unit = canonical_unit or None

            # Números inline na própria linha do rótulo
            inline_nums = re.findall(_NUM, line)
            inline_nums = [parse_currency(n) for n in inline_nums if parse_currency(n) is not None]

            # Lê até 6 linhas seguintes (formato vertical)
            for j in range(i + 1, min(i + 7, len(lines))):
                l = lines[j].strip()
                if not l:
                    break
                if re.match(r"^[A-ZÀ-Ú]{4,}(\s+[A-ZÀ-Ú]+){0,3}$", l):
                    if not re.match(r"^[Kk][Ww]|^[Mm][Ww]", l):
                        break
                if re.match(r"^[Kk][Ww][Hh]$|^[Mm][Ww][Hh]?$|^[Kk][Ww]$|^[Kk][Vv][Aa][Rr][Hh]?$",
                            l, re.IGNORECASE):
                    unit = l.upper()
                    continue
                n = parse_currency(l)
                if n is None:
                    continue
                if qty is None:
                    qty = n
                elif unit_price is None:
                    unit_price = n
                elif total is None:
                    total = n

            # Itens sem unidade (CIP, ICMS Subvenção, Tributo Federal):
            # o único número relevante é o valor total — usa o último número
            # "razoável" da linha (< 1 000 000) para evitar capturar códigos NF.
            if canonical_unit == "":
                reasonable = [n for n in inline_nums if isinstance(n, float) and 0 < n < 1_000_000]
                total = reasonable[-1] if reasonable else total
                qty = None
                unit_price = None
            else:
                # Fallback inline para itens com unidade
                if qty is None and len(inline_nums) >= 1:
                    qty = inline_nums[0]
                if unit_price is None and len(inline_nums) >= 2:
                    unit_price = inline_nums[1]
                if total is None and len(inline_nums) >= 3:
                    total = inline_nums[2]

            # Sanity: MWh qty > 99 999 é quase certo um código/IE capturado errado
            if canonical_unit == "MWh" and isinstance(qty, float) and qty > 99_999:
                qty = None

            # "Energia Elétrica" só é criada se a unidade MWh foi confirmada
            # (inline "(MWh)" ou linha seguinte) — evita falsos positivos em
            # cabeçalhos de tabela de contas de concessionária.
            if canonical_desc == "Energia Elétrica":
                mwh_confirmed = bool(re.search(r"\bMWh?\b", line, re.IGNORECASE)) or unit == "MWH"
                if not mwh_confirmed:
                    continue

            if qty is not None or total is not None:
                # Inclui a bandeira na descrição quando relevante
                desc = (
                    f"{canonical_desc} (Bandeira {active_bandeira.title()})"
                    if active_bandeira else canonical_desc
                )
                items.append(LineItem(
                    description=desc,
                    unit=unit,
                    quantity=abs(qty) if isinstance(qty, float) else qty,
                    unit_price=unit_price,
                    total=total,
                ))
                seen_descs.add(desc_key)
                claimed_lines.add(i)

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
        # complementa com regex (estado, IE).
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
                if regex_supplier.address and regex_supplier.address.state:
                    supplier.address = regex_supplier.address
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
                if regex_customer.address and regex_customer.address.state:
                    customer.address = regex_customer.address
        else:
            customer = ext.customer(text, layout)

        # DoclingExtractor via kv_pairs/section_map tem prioridade para UF.
        if supplier and pre.get("supplier_state"):
            supplier.address.state = pre["supplier_state"]
        if customer and pre.get("customer_state"):
            customer.address.state = pre["customer_state"]

        # Total: DoclingExtractor via kv_pairs/spatial tem prioridade; labels regex como fallback.
        total = pre.get("total") or ext.total_via_labels(text, layout)
        subtotal = ext.subtotal(text)
        # DoclingExtractor (semântico) tem prioridade; regex como fallback.
        taxes = pre.get("taxes") or ext.taxes(text, layout)

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
                        if not supplier.address.state:
                            supplier.address.state = known_nfe.get("state")

        # -- Itens --
        # Ordem de prioridade:
        #   1. DoclingExtractor (tabelas estruturadas — maior qualidade)
        #   2. Fallback textual genérico (layout NF-e/NFS-e/boleto)
        #   3. Fallback textual de energia (items HP/HFP/TUSD/TE/demanda)
        line_items = (
            pre.get("line_items")
            or _extract_line_items_from_tables(content.tables)
        )
        if not line_items:
            line_items = _extract_line_items_from_text(text, layout)

        # Fallback de energia: para faturas sem tabela capturável pelo Docling,
        # tenta extrair itens de energia do texto plano. Só ativa se line_items
        # ainda estiver vazio OU se não houver nenhum item com quantidade kWh/MWh.
        if not line_items:
            energy_items = _extract_energy_items_from_text(text)
            if energy_items:
                line_items = energy_items
        else:
            # line_items tem dados, mas pode ter vindo de NF-e de comercializadora
            # onde só o cabeçalho ficou (sem linha de produto). Testa se algum
            # item tem quantidade numérica; caso contrário, complementa com
            # fallback de energia.
            has_qty = any(isinstance(i.quantity, (int, float)) for i in line_items)
            if not has_qty:
                energy_items = _extract_energy_items_from_text(text)
                if energy_items:
                    line_items = energy_items

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
                if not supplier.address.state:
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

        # -- Identificadores de instalação de energia --
        consumer_unit = pre.get("consumer_unit") or extract_consumer_unit(text)
        client_number = extract_client_number(text)

        # NF-e de comercializadora: tenta preencher qty MWh via padrão dedicado
        # (funciona mesmo quando _extract_energy_items_from_text criou o item mas
        # não conseguiu capturar a quantidade porque ela estava em outra posição).
        if layout == InvoiceLayout.NFE:
            mwh = extract_energy_quantity_mwh(text)
            if mwh is not None:
                filled = False
                for _item in line_items:
                    if _item.unit in ("MWh", "MWH") and _item.quantity is None:
                        _item.quantity = mwh
                        filled = True
                        break
                if not filled and not line_items:
                    line_items = [LineItem(
                        description="Energia Elétrica",
                        unit="MWh",
                        quantity=mwh,
                    )]

        invoice = Invoice(
            invoice_number=invoice_number,
            invoice_layout=layout,
            access_key=access_key,
            series=series,
            issue_date=issue_date,
            due_date=due_date,
            supplier=supplier or Party(),
            customer=customer or Party(),
            consumer_unit=consumer_unit,
            client_number=client_number,
            line_items=line_items,
            taxes=taxes,
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

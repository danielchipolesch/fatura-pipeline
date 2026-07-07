"""
Extrator primário de campos de faturas usando a estrutura semântica do DoclingDocument.

Opera sobre DocumentContent enriquecido (section_map, kv_pairs, spatial_index,
page_texts) gerado pelo DoclingLoader. É a primeira camada do pipeline de extração —
o DeterministicParser (regex) só processa campos que este extrator não conseguiu
preencher.

Estratégia por campo (em ordem de confiança decrescente):
  1. kv_pairs    — pares chave-valor detectados pelo modelo de layout (maior confiança)
  2. section_map — texto do bloco da seção + regex scoped (menor ruído que busca global)
  3. spatial_index — busca por proximidade visual (elimina hacks posicionais)
  4. page_texts  — regex scoped por página (reduz falsos positivos cross-page)
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from loguru import logger

from src.extractors.fields import (
    extract_br_state,
    extract_cnpj,
    extract_consumer_unit,
    extract_date,
    parse_currency,
)
from src.extractors.tables import extract_line_items_from_tables
from src.extractors.taxes import (
    SECTIONS_TAX,
    build_taxes,
    extract_taxes_from_kv_pairs,
    extract_taxes_from_section_text,
    extract_taxes_from_tables,
)
from src.models.invoice import InvoiceLayout
from src.parsers.docling_loader import DocumentContent, SpatialElement

# ---------------------------------------------------------------------------
# Padrões de chave para matching em kv_pairs
# ---------------------------------------------------------------------------

_KV_INVOICE_NUMBER = [
    "número da nota", "n.o da nota", "nf", "nota fiscal", "número", "fatura", "nota",
]
_KV_ISSUE_DATE = [
    "data de emissão", "data emissão", "emissão", "emitida em",
]
_KV_DUE_DATE = [
    "data de vencimento", "data vencimento", "vencimento", "pagar até",
    "data limite", "due date",
]
_KV_TOTAL = [
    "valor total", "total a pagar", "valor do documento", "valor cobrado",
    "total fatura", "valor fatura", "total",
]
_KV_CNPJ = [
    "cnpj", "cnpj/cpf", "cpf/cnpj",
]
_KV_CITY = [
    "município", "municipio", "munic.", "cidade",
]
_KV_STATE = [
    "u.f.", "uf", "estado",
]
_KV_CONSUMER_UNIT = [
    "unidade consumidora", "unid. consumidora",
    "instalacao", "instalação", "n. instalacao", "n. da instalacao",
    "cod. instalacao", "cod. instalação", "codigo instalacao",
    "ponto de entrega", "pde",
    "uc",
]

# ---------------------------------------------------------------------------
# Nomes de seção para matching em section_map
# ---------------------------------------------------------------------------

_SECTIONS_SUPPLIER = [
    "EMITENTE",
    "EMIT.",
    "PRESTADOR DE SERVIÇOS",
    "PRESTADOR",
    "BENEFICIÁRIO",
    "CEDENTE",
    "CONCESSIONÁRIA",
    "DISTRIBUIDORA",
    "FORNECEDOR",
    # DANF3E (Nota Fiscal de Energia Elétrica): o Docling agrupa o bloco
    # do emitente sob o título do documento auxiliar em vez de uma seção
    # "EMITENTE" explícita.
    "DANF3E",
    "NF3E",
    "NOTA FISCAL DE ENERGIA",
]
_SECTIONS_CUSTOMER = [
    "DESTINATÁRIO",
    "DESTINATÁRIO/REMETENTE",
    "TOMADOR DE SERVIÇOS",
    "TOMADOR",
    "SACADO",
    "PAGADOR",
    "CONSUMIDOR",
    "CLIENTE",
    "UNIDADE CONSUMIDORA",
]
_SECTIONS_ADDITIONAL = [
    "INFORMAÇÕES ADICIONAIS",
    "INFORMAÇÕES ADICIONAIS DO FISCO",
    "INFORMAÇÕES COMPLEMENTARES",
    "DADOS ADICIONAIS",
    "OBSERVAÇÕES",
    "OBS",
    "COMPLEMENTO",
    "INFORMAÇÕES",
]

# Linha de ruído: só labels, só números, e-mail avulso, CEP, etc.
_NOISE_LINE_RE = re.compile(
    r"^(CNPJ|CPF|IE|IM|CEP|ENDEREÇO|END\.|E-MAIL|FONE|TEL|FAX|"
    r"INSCRI|MUNIC[IÍ]PIO|CIDADE|ESTADO|UF|BAIRRO|RUA|AV\.|LOGRADOURO|"
    r"RAZÃO\s+SOCIAL|NOME)[:\s]",
    re.IGNORECASE,
)

# Rótulos de seção que aparecem literalmente na linha (ex: "EMITENTE:" dentro do texto)
_SECTION_LABEL_INLINE_RE = re.compile(
    r"^(EMITENTE|DESTINAT[ÁA]RIO|TOMADOR|PRESTADOR|SACADO|CEDENTE"
    r"|BENEFICI[ÁA]RIO|CONSUMIDOR|CLIENTE)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DoclingExtractor
# ---------------------------------------------------------------------------

class DoclingExtractor:
    """
    Camada primária de extração: usa a estrutura semântica do Docling.
    Retorna um dict parcial com os campos extraídos com alta confiança.
    O DeterministicParser usa esse dict para pular regex nos campos já preenchidos.
    """

    def extract(self, content: DocumentContent, layout: InvoiceLayout) -> dict:
        """
        Extrai campos estruturais do documento.
        Retorna dict parcial — apenas campos onde a extração teve sucesso.
        """
        result: dict = {}

        self._from_kv_pairs(content, result)
        self._from_sections(content, result)
        self._from_spatial(content, layout, result)
        self._from_page_texts(content, layout, result)
        self._from_tables(content, result)
        self._from_taxes(content, result)

        n = len(result)
        logger.debug(
            f"DoclingExtractor: {n} campo(s) extraído(s): {list(result.keys())}"
        )
        return result

    # ------------------------------------------------------------------
    # 1. Extração via kv_pairs (pares nativos do Docling)
    # ------------------------------------------------------------------

    def _from_kv_pairs(self, content: DocumentContent, result: dict) -> None:
        if not content.kv_pairs:
            return

        # Acumula valores por tipo para distinguir supplier vs customer pela ordem
        cnpj_list:  list[str] = []
        city_list:  list[str] = []
        state_list: list[str] = []

        for key, value in content.kv_pairs:
            key_lower = key.lower().strip()
            value_stripped = value.strip()

            if "invoice_number" not in result and _kv_matches(key_lower, _KV_INVOICE_NUMBER):
                if value_stripped and value_stripped not in ("0", ""):
                    result["invoice_number"] = value_stripped

            if "issue_date" not in result and _kv_matches(key_lower, _KV_ISSUE_DATE):
                d = extract_date(value_stripped)
                if d:
                    result["issue_date"] = d

            if "due_date" not in result and _kv_matches(key_lower, _KV_DUE_DATE):
                d = extract_date(value_stripped)
                if d:
                    result["due_date"] = d

            if "total" not in result and _kv_matches(key_lower, _KV_TOTAL):
                v = parse_currency(value_stripped)
                if v and v > 0:
                    result["total"] = v

            if _kv_matches(key_lower, _KV_CNPJ):
                cnpj = extract_cnpj(value_stripped)
                if cnpj and cnpj not in cnpj_list:
                    cnpj_list.append(cnpj)

            if _kv_matches(key_lower, _KV_CITY) and value_stripped:
                v = value_stripped.upper()
                if v not in city_list:
                    city_list.append(v)

            if _kv_matches(key_lower, _KV_STATE) and len(value_stripped) == 2:
                v = value_stripped.upper()
                if v not in state_list:
                    state_list.append(v)

            if "consumer_unit" not in result and _kv_matches(key_lower, _KV_CONSUMER_UNIT):
                # Aceita valor que contenha ao menos 4 dígitos (código numérico de UC)
                import re as _re
                if value_stripped and len(_re.sub(r"[^\d]", "", value_stripped)) >= 4:
                    result["consumer_unit"] = value_stripped

        # Heurística de ordem: 1ª ocorrência = supplier, 2ª = customer
        if cnpj_list and "supplier_cnpj" not in result:
            result["supplier_cnpj"] = cnpj_list[0]
        if len(cnpj_list) >= 2 and "customer_cnpj" not in result:
            result["customer_cnpj"] = cnpj_list[1]

        if city_list and "supplier_city" not in result:
            result["supplier_city"] = city_list[0]
        if len(city_list) >= 2 and "customer_city" not in result:
            result["customer_city"] = city_list[1]

        if state_list and "supplier_state" not in result:
            result["supplier_state"] = state_list[0]
        if len(state_list) >= 2 and "customer_state" not in result:
            result["customer_state"] = state_list[1]

    # ------------------------------------------------------------------
    # 2. Extração via section_map (blocos de seção tipados)
    # ------------------------------------------------------------------

    def _from_sections(self, content: DocumentContent, result: dict) -> None:
        if not content.section_map:
            return

        supplier_section = _get_section(content.section_map, _SECTIONS_SUPPLIER)
        if supplier_section:
            sec_key, supplier_text = supplier_section
            if "supplier_cnpj" not in result:
                cnpj = extract_cnpj(supplier_text)
                if cnpj:
                    result["supplier_cnpj"] = cnpj

            if "supplier_name" not in result:
                # Prioridade 1: nome embutido no título da seção
                # ex: "IDENTIFICAÇÃO DO EMITENTE COMERC POWER TRADING"
                name = _name_from_section_key(sec_key)
                if name:
                    # Incorpora sufixo legal ("LTDA", "S.A.") da 1ª linha do
                    # corpo quando ele segue imediatamente o nome truncado
                    body_first = supplier_text.lstrip().split("\n")[0].strip()
                    suffix_m = re.match(
                        r"^(LTDA\.?|S\.?A\.?|EIRELI|ME|EPP)\b",
                        body_first, re.IGNORECASE,
                    )
                    if suffix_m:
                        name = name + " " + suffix_m.group(1).upper()
                else:
                    # Prioridade 2: primeira linha substantiva do corpo
                    name = _first_substantive_line(supplier_text)
                if name:
                    result["supplier_name"] = name

            city, state = _extract_city_state(supplier_text)
            if city and "supplier_city" not in result:
                result["supplier_city"] = city
            if state and "supplier_state" not in result:
                result["supplier_state"] = state

        customer_text = _get_section_text(content.section_map, _SECTIONS_CUSTOMER)
        if customer_text:
            if "customer_cnpj" not in result:
                cnpj = extract_cnpj(customer_text)
                if cnpj:
                    result["customer_cnpj"] = cnpj

            if "customer_name" not in result:
                name = _first_substantive_line(customer_text)
                if name:
                    result["customer_name"] = name

            city, state = _extract_city_state(customer_text)
            if city and "customer_city" not in result:
                result["customer_city"] = city
            if state and "customer_state" not in result:
                result["customer_state"] = state

        # Consumer unit — NF-e de comercializadora tipicamente registra o código
        # da instalação/ponto de entrega nas "INFORMAÇÕES ADICIONAIS".
        if "consumer_unit" not in result:
            add_text = _get_section_text(content.section_map, _SECTIONS_ADDITIONAL)
            if add_text:
                cu = extract_consumer_unit(add_text)
                if cu:
                    result["consumer_unit"] = cu

        # Impostos via seção semântica (fallback quando kv_pairs/tabelas não
        # encontraram os tributos — seção já identificada pelo modelo Docling).
        if "taxes" not in result:
            tax_text = _get_section_text(content.section_map, SECTIONS_TAX)
            if tax_text:
                taxes = extract_taxes_from_section_text(tax_text)
                if taxes:
                    result["taxes"] = taxes

    # ------------------------------------------------------------------
    # 3. Extração via spatial_index (proximidade visual — bounding boxes)
    # ------------------------------------------------------------------

    def _from_spatial(
        self, content: DocumentContent, layout: InvoiceLayout, result: dict
    ) -> None:
        if not content.spatial_index:
            return

        # Total via âncora espacial (substitui o hack "MTE\d{7}")
        if "total" not in result:
            total_labels = [
                r"TOTAL\s+A\s+PAGAR",
                r"VALOR\s+TOTAL",
                r"VALOR\s+DO\s+DOCUMENTO",
                r"TOTAL\s+FATURA",
                r"\bTOTAL\b",
            ]
            for label_pattern in total_labels:
                val_text = _spatial_right_or_below(
                    content.spatial_index, label_pattern,
                    value_pattern=r"[\d.,]+",
                )
                if val_text:
                    v = parse_currency(val_text)
                    if v and v > 0:
                        result["total"] = v
                        break

        # Data de vencimento via âncora espacial
        if "due_date" not in result:
            due_labels = [
                r"VENCIMENTO",
                r"DATA\s+(?:DE\s+)?VENCIMENTO",
                r"PAGAR\s+AT[EÉ]",
            ]
            for label_pattern in due_labels:
                val_text = _spatial_right_or_below(
                    content.spatial_index, label_pattern,
                    value_pattern=r"\d{2}[/\-.]\d{2}[/\-.]\d{4}",
                )
                if val_text:
                    d = extract_date(val_text)
                    if d:
                        result["due_date"] = d
                        break

        # Data de emissão via âncora espacial
        if "issue_date" not in result:
            issue_labels = [
                r"DATA\s+(?:DE\s+)?EMISS[ÃA]O",
                r"EMISS[ÃA]O",
                r"EMITIDA?\s+EM",
            ]
            for label_pattern in issue_labels:
                val_text = _spatial_right_or_below(
                    content.spatial_index, label_pattern,
                    value_pattern=r"\d{2}[/\-.]\d{2}[/\-.]\d{4}",
                )
                if val_text:
                    d = extract_date(val_text)
                    if d:
                        result["issue_date"] = d
                        break

        # Consumer unit via âncora espacial
        if "consumer_unit" not in result:
            uc_labels = [
                r"PONTO\s+DE\s+ENTREGA",
                r"UNID(?:ADE)?\.?\s*CONSUMIDORA",
                r"C[ÓO]D(?:IGO)?\.?\s*INSTALA[ÇC][ÃA]O",
                r"N[ºo°]\.?\s*(?:DA\s+)?INSTALA[ÇC][ÃA]O",
            ]
            for label_pattern in uc_labels:
                val_text = _spatial_right_or_below(
                    content.spatial_index, label_pattern,
                    value_pattern=r"[\d][\d.\-/]*",
                )
                if val_text and len(re.sub(r"[^\d]", "", val_text)) >= 4:
                    result["consumer_unit"] = val_text
                    break

        # Impostos via âncora espacial — funciona quando labels e valores estão
        # lado a lado no layout (DANFE NF3E, boleto com tabela de ICMS/PIS/COFINS).
        # Prioridade: "V. DO ICMS" (específico) antes de "ICMS" genérico para
        # não confundir com labels de coluna de detalhe (ex: "ICMS por componente").
        if "taxes" not in result:
            _TAX_SPATIAL = {
                "ICMS": [
                    r"V(?:ALOR)?\.?\s*DO\s+ICMS\b",
                    r"TOTAL\s+ICMS\b",
                    r"\bICMS\b(?!\s*(?:ST\b|BASE\b|RET\b|RETID\b|RETENC))",
                ],
                "PIS": [
                    r"V(?:ALOR)?\.?\s*(?:TOTAL\s+)?PIS\b",
                    r"\bPIS/PASEP\b",
                    r"\bPIS\b",
                ],
                "COFINS": [
                    r"V(?:ALOR)?\.?\s*(?:TOTAL\s+)?(?:D[AO]\s+)?COFINS\b",
                    r"\bCOFINS\b",
                ],
            }
            _MONEY_PAT = r"\d[\d.,]*[.,]\d\d"
            tax_spatial: dict = {}
            for tax_name, patterns in _TAX_SPATIAL.items():
                for pat in patterns:
                    val_text = _spatial_right_or_below(
                        content.spatial_index, pat,
                        value_pattern=_MONEY_PAT,
                    )
                    if val_text is not None:
                        v = parse_currency(val_text)
                        if v is not None and v >= 0:
                            tax_spatial[tax_name] = {"amount": v}
                            break
            if tax_spatial:
                result["taxes"] = build_taxes(tax_spatial)

    # ------------------------------------------------------------------
    # 4. Extração via page_texts (regex scoped por página)
    # ------------------------------------------------------------------

    def _from_page_texts(
        self, content: DocumentContent, layout: InvoiceLayout, result: dict
    ) -> None:
        if not content.page_texts:
            return

        page_1 = content.page_texts.get(1, "")

        # Chave de acesso NF-e: sempre na pág. 1 (reduz falso-positivos de
        # sequências de 44 dígitos que possam aparecer em outras páginas)
        if layout == InvoiceLayout.NFE and "access_key" not in result and page_1:
            from src.extractors.fields import extract_access_key
            ak = extract_access_key(page_1)
            if ak:
                result["access_key"] = ak

        # Número da nota via pág. 1 (para NFe/NFSe que têm o número no cabeçalho)
        if layout in (InvoiceLayout.NFE, InvoiceLayout.NFSE):
            if "invoice_number" not in result and page_1:
                m = re.search(
                    r"N[°ºo\.]\s*(?:da\s+)?(?:Nota|NF)?\s*[:\s]+(\d+)",
                    page_1, re.IGNORECASE,
                )
                if m:
                    val = m.group(1).strip()
                    if val and val not in ("0", ""):
                        result["invoice_number"] = val


    # ------------------------------------------------------------------
    # 5. Extração de itens via tabelas Docling
    # ------------------------------------------------------------------

    def _from_tables(self, content: DocumentContent, result: dict) -> None:
        if not content.tables or "line_items" in result:
            return
        items = extract_line_items_from_tables(content.tables)
        if items:
            result["line_items"] = items

    # ------------------------------------------------------------------
    # 6. Extração de impostos (kv_pairs → tabelas; seção já feita em _from_sections)
    # ------------------------------------------------------------------

    def _from_taxes(self, content: DocumentContent, result: dict) -> None:
        if "taxes" in result:
            return

        # Estratégia 1: kv_pairs — campos como "BASE DE CÁLCULO DO ICMS"
        if content.kv_pairs:
            taxes = extract_taxes_from_kv_pairs(content.kv_pairs)
            if taxes:
                result["taxes"] = taxes
                return

        # Estratégia 2: tabelas — linhas ou colunas com estrutura de tributos
        if content.tables:
            taxes = extract_taxes_from_tables(content.tables)
            if taxes:
                result["taxes"] = taxes
                return

        # Estratégia 3: varre TODAS as seções em busca de conteúdo tributário.
        # Necessário quando Docling nomeia seções com títulos não-fiscais
        # (ex: número de boleto como título de seção no boleto Light 376993.pdf,
        #  ou "NOTA FISCAL Nº..." na NF3E da RGE SUL 141846706.pdf).
        if content.section_map:
            _TAX_KW = ("ICMS", "PIS", "COFINS", "TRIBUTO")
            for sec_text in content.section_map.values():
                if len(sec_text) < 30:
                    continue
                sec_up = sec_text.upper()
                if any(k in sec_up for k in _TAX_KW):
                    taxes = extract_taxes_from_section_text(sec_text)
                    if taxes:
                        result["taxes"] = taxes
                        return


# ---------------------------------------------------------------------------
# Funções auxiliares (puras — sem estado)
# ---------------------------------------------------------------------------

def _kv_matches(key_lower: str, patterns: list[str]) -> bool:
    """Verifica se algum padrão está contido na chave (busca por substring)."""
    return any(p in key_lower for p in patterns)


def _get_section_text(section_map: dict, candidates: list[str]) -> str | None:
    """Retorna o texto da primeira seção encontrada (exata > parcial)."""
    result = _get_section(section_map, candidates)
    return result[1] if result else None


def _get_section(section_map: dict, candidates: list[str]) -> tuple[str, str] | None:
    """
    Retorna (sec_key_original, sec_text) da primeira seção correspondente.
    Tenta exata primeiro, depois parcial (seção cujo nome contém o candidato).
    """
    section_upper = {k.upper(): (k, v) for k, v in section_map.items()}
    for candidate in candidates:
        cand_upper = candidate.upper()
        if cand_upper in section_upper:
            return section_upper[cand_upper]
        for sec_key_up, (orig_key, sec_text) in section_upper.items():
            if cand_upper in sec_key_up:
                return orig_key, sec_text
    return None


def _name_from_section_key(sec_key: str) -> str | None:
    """
    Extrai o nome do emitente embutido no título da seção.

    Padrão: "IDENTIFICAÇÃO DO EMITENTE <NomeEmpresa>" ou
            "IDENTIFICAÇÃO DE EMITENTE <NomeEmpresa>"
    Retorna apenas a parte após o rótulo "EMITENTE", ou None se não houver.
    """
    m = re.search(
        r"(?:IDENTIFICAÇÃO\s+D[AEO]\s+)?EMIT(?:ENTE)?\b[.:\s]+(.+)",
        sec_key,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
        if name and len(name) > 2:
            return name
    return None


def _extract_city_state(text: str) -> tuple[str | None, str | None]:
    """
    Extrai cidade e UF de um bloco de texto de endereço (seção EMITENTE ou
    DESTINATÁRIO extraída pelo Docling).

    Estratégias em ordem de confiança:
      1. Rótulo explícito: "MUNICÍPIO: GUARATINGUETA" + "U.F.: SP"
      2. Padrão inline sem rótulo: "GUARATINGUETA/SP" ou "GUARATINGUETA - SP"
    """
    text_upper = text.upper()
    city: str | None = None
    state: str | None = None

    # 1a. Cidade com rótulo
    city_m = re.search(
        r"(?:MUN[IÍ]C[IÍ]PIO|MUNICIPIO|CIDADE)\s*[:\s]+([A-ZÀ-Ú][^\n,/]{2,38}?)(?:\s*(?:U\.?F|$|\n))",
        text_upper,
    )
    if city_m:
        city = city_m.group(1).strip()

    # 1b. UF com rótulo
    uf_m = re.search(r"\bU\.?F\.?\s*[:\-]?\s*([A-Z]{2})\b", text_upper)
    if uf_m:
        candidate = uf_m.group(1)
        if extract_br_state(candidate):
            state = candidate

    # 2. Padrão inline "CIDADE/UF" — cobre layouts sem rótulos explícitos
    if not city or not state:
        inline_m = re.search(
            r"\b([A-ZÀ-Ú][A-ZÀ-Ú\s]{2,38}?)\s*/\s*([A-Z]{2})\b",
            text_upper,
        )
        if inline_m:
            candidate_state = extract_br_state(inline_m.group(2))
            if candidate_state:
                if not city:
                    city = inline_m.group(1).strip()
                if not state:
                    state = candidate_state

    return city, state


def _first_substantive_line(text: str) -> str | None:
    """
    Retorna a primeira linha não-vazia que represente um nome de empresa/pessoa.
    Filtra:
      - Linhas que são rótulos (CNPJ, CEP, Endereço, etc.)
      - Linhas que são só rótulo de seção (ex: "EMITENTE:")
      - Linhas puramente numéricas / datas / valores monetários
      - Linhas muito curtas (< 4 chars)
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        if _NOISE_LINE_RE.match(line):
            continue
        if _SECTION_LABEL_INLINE_RE.match(line):
            continue
        if re.match(r"^[\d.,/\-\s]+$", line):
            continue
        if re.match(r"^\d{2}/\d{2}/\d{4}$", line):
            continue
        return line
    return None


def _spatial_right_or_below(
    spatial_index: list[SpatialElement],
    label_pattern: str,
    page_no: int | None = None,
    value_pattern: str | None = None,
) -> str | None:
    """
    Busca o elemento de texto mais próximo à direita ou imediatamente abaixo
    de um elemento que casa com label_pattern.

    Retorna o texto do elemento candidato mais próximo que satisfaz
    value_pattern (se fornecido). Retorna None se nada for encontrado.

    Lógica de proximidade:
      - À direita: mesmo band vertical (centros dentro de ½ altura do rótulo),
        bbox_l do candidato >= bbox_r do rótulo (com tolerância de 2 pts).
      - Abaixo: bbox_t do candidato dentro de 2 alturas do rótulo abaixo de
        bbox_b, horizontalmente alinhado com o rótulo (±10 pts).
      - Candidatos "à direita" têm prioridade sobre os "abaixo" (bias = +1000).
    """
    label_re = re.compile(label_pattern, re.IGNORECASE)
    val_re = re.compile(value_pattern, re.IGNORECASE) if value_pattern else None

    for elem in spatial_index:
        if page_no is not None and elem.page_no != page_no:
            continue
        if not label_re.search(elem.text):
            continue

        mid_y = (elem.bbox_t + elem.bbox_b) / 2
        height = max(elem.bbox_b - elem.bbox_t, 1.0)

        candidates: list[tuple[float, SpatialElement]] = []
        for other in spatial_index:
            if other is elem or other.page_no != elem.page_no:
                continue
            if val_re and not val_re.search(other.text):
                continue

            other_mid_y = (other.bbox_t + other.bbox_b) / 2

            # À direita: mesma linha (centros verticais próximos)
            if (
                abs(other_mid_y - mid_y) <= height * 0.7
                and other.bbox_l >= elem.bbox_r - 2
            ):
                dist = other.bbox_l - elem.bbox_r
                candidates.append((dist, other))

            # Abaixo: próxima linha, alinhado horizontalmente com o rótulo
            elif (
                other.bbox_t >= elem.bbox_b - 2
                and other.bbox_t <= elem.bbox_b + height * 2
                and other.bbox_l >= elem.bbox_l - 10
                and other.bbox_r <= elem.bbox_r + 10
            ):
                dist = (other.bbox_t - elem.bbox_b) + 1000
                candidates.append((dist, other))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1].text

    return None

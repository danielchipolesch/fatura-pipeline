"""
Padrões regex e funções de extração de campos individuais de faturas brasileiras.
Toda extração aqui é puramente determinística — sem LLM.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from src.models.invoice import READ_ERROR_SENTINEL

# ---------------------------------------------------------------------------
# Padrões compilados
# ---------------------------------------------------------------------------

_CNPJ_RE = re.compile(r"\b(\d{2})[.\-]?(\d{3})[.\-]?(\d{3})[/\-]?(\d{4})[\-]?(\d{2})\b")
_CPF_RE = re.compile(r"\b(\d{3})[.\-]?(\d{3})[.\-]?(\d{3})[\-]?(\d{2})\b")
_CEP_RE = re.compile(r"\b(\d{5})[\-]?(\d{3})\b")
_ACCESS_KEY_RE = re.compile(r"\b(\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4})\b|\b(\d{44})\b")
_BARCODE_NUMERIC_RE = re.compile(r"\b(\d{44})\b")
_BARCODE_FULL_RE = re.compile(r"\b(\d{47,48})\b")
_ACCESS_KEY_LABEL_RE = re.compile(r"CHAVE\s+DE\s+ACESSO", re.IGNORECASE)
_CNPJ_PUNCT_RE = r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}"
_DATE_BEFORE_CURRENCY_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s*\n?\s*R\$\s*([\d.,]+)")
# Variante "solta": exige apenas que não haja OUTRO dígito colado antes/depois,
# mas permite que letras fiquem adjacentes (comum quando se remove espaço/quebra
# de linha de texto OCR e o número de dígitos vira vizinho de uma palavra —
# nesse caso \b não forma fronteira entre dígito e letra, ambos são \w).
_DIGITS_44_LOOSE_RE = re.compile(r"(?<!\d)(\d{44})(?!\d)")

# Linha digitável: formato com pontos e espaços, 47-48 dígitos no total
_BOLETO_LINE_RE = re.compile(r"\d{5}\.\d{5}\s+\d{5}\.\d{6}\s+\d{5}\.\d{6}\s+\d\s+\d{14}")

# Assinatura hexadecimal usada por alguns layouts de fatura de energia em vez
# da chave de acesso tradicional (ex: "1B69.7265.44BE.92BD.1DF1.0F3C.1485.AE2F")
_HEX_SIGNATURE_RE = re.compile(
    r"\b[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\."
    r"[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\b"
)
# Após a assinatura hex, segue um token curto e variável (código de 8 dígitos
# da instalação, uma letra de série, etc.) e então o número da nota com 9
# dígitos (zero-padded) — a posição do número é estável, o token entre eles
# varia conforme a sub-variante do layout.
_HEX_SIGNATURE_THEN_NUMBER_RE = re.compile(
    _HEX_SIGNATURE_RE.pattern + r"\s*\n?[^\n]{0,15}\n?\s*(\d{9})\b"
)

_DATE_DMY_RE = re.compile(r"\b(\d{2})[/\-.](\d{2})[/\-.](\d{4})\b")
_DATE_YMD_RE = re.compile(r"\b(\d{4})[\-/](\d{2})[\-/](\d{2})\b")

# Data abreviada em português, comum em layouts de fatura sem rótulo explícito
# (ex: "12 JAN 2022", "23 FEV 2023") — formato típico de DANFE de energia.
_MONTH_ABBR_PT = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}
_DATE_PTBR_ABBREV_RE = re.compile(
    r"\b(\d{1,2})\s+(JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)\s+(\d{4})\b",
    re.IGNORECASE,
)

_CURRENCY_RE = re.compile(
    r"R\$\s*([\d.,]+)"
    r"|(?<![A-Za-z\d])(\d{1,3}(?:\.\d{3})*,\d{2})(?!\d)"
)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Telefone brasileiro: (DDD) XXXX-XXXX ou (DDD) XXXXX-XXXX, com parênteses obrigatórios no DDD
_PHONE_RE = re.compile(r"\(\d{2}\)\s*\d{4,5}[\-\s]\d{4}")

_BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_cnpj(g: tuple[str, ...]) -> str:
    d = "".join(g)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def _fmt_cpf(g: tuple[str, ...]) -> str:
    d = "".join(g)
    return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"


def _fmt_cep(g: tuple[str, ...]) -> str:
    return f"{g[0]}-{g[1]}"


def format_cnpj_digits(digits: str) -> str:
    """Formata uma string de 14 dígitos no padrão XX.XXX.XXX/XXXX-XX."""
    d = re.sub(r"[^\d]", "", digits)
    if len(d) != 14:
        return digits
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_cnpj(text: str) -> Optional[str]:
    """Extrai o primeiro CNPJ válido do texto."""
    for m in _CNPJ_RE.finditer(text):
        digits = "".join(m.groups())
        if len(digits) == 14:
            return _fmt_cnpj(m.groups())
    return None


def extract_all_cnpjs(text: str) -> list[str]:
    """Extrai todos os CNPJs encontrados (deduplicado, ordem de aparição)."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _CNPJ_RE.finditer(text):
        digits = "".join(m.groups())
        if len(digits) == 14 and digits not in seen:
            seen.add(digits)
            result.append(_fmt_cnpj(m.groups()))
    return result


def extract_cpf(text: str) -> Optional[str]:
    for m in _CPF_RE.finditer(text):
        digits = "".join(m.groups())
        if len(digits) == 11:
            return _fmt_cpf(m.groups())
    return None


def extract_cep(text: str) -> Optional[str]:
    m = _CEP_RE.search(text)
    return _fmt_cep(m.groups()) if m else None


def extract_access_key(text: str) -> Optional[str]:
    """
    Extrai a chave de acesso da NF-e (44 dígitos).

    Tolerante a quebras de linha entre os grupos de dígitos — comum em texto
    extraído via OCR, onde cada grupo de 4 dígitos cai em uma linha própria.
    Usa lookaround restrito a dígitos (não \\b) porque, após remover espaços/
    quebras, o número pode ficar colado a uma palavra (ex: "...4499Protocolo"),
    e nesse caso \\b não marcaria fronteira (dígito e letra são ambos \\w).

    A busca tenta cada ocorrência do rótulo "CHAVE DE ACESSO" no texto (pode
    haver mais de uma, ex: um link de consulta antes do rótulo real), restrita
    a uma janela de 300 caracteres após cada uma, para evitar concatenar
    números não relacionados de outras partes do documento.
    """
    for label_m in _ACCESS_KEY_LABEL_RE.finditer(text):
        window = text[label_m.end(): label_m.end() + 300]
        clean = re.sub(r"\s+", "", window)
        m = _DIGITS_44_LOOSE_RE.search(clean)
        if m:
            return m.group(1)
    # Fallback: tenta no texto inteiro (rótulo pode não ter sido capturado como texto)
    clean_full = re.sub(r"\s+", "", text)
    m = _DIGITS_44_LOOSE_RE.search(clean_full)
    return m.group(1) if m else None


def extract_boleto_barcode(text: str) -> Optional[str]:
    """Extrai código de barras do boleto (44 dígitos após limpar)."""
    clean = re.sub(r"\s+", "", text)
    m = _DIGITS_44_LOOSE_RE.search(clean)
    return m.group(1) if m else None


def extract_cnpj_after_label(text: str, label_pattern: str) -> Optional[str]:
    """
    Extrai um CNPJ (14 dígitos, com ou sem pontuação) que aparece imediatamente
    após um rótulo específico (ex: 'CNPJ/CPF' ou 'CPF/CNPJ').
    """
    m = re.search(
        rf"(?:{label_pattern})\s*[:\s]*({_CNPJ_PUNCT_RE})",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    if len(digits) != 14:
        return None
    return format_cnpj_digits(digits)


def extract_date_before_currency(text: str) -> tuple[Optional[date], Optional[float]]:
    """
    Busca o padrão 'DD/MM/AAAA' seguido de 'R$ valor', sem depender de rótulo.
    Útil quando o rótulo (ex: 'VENCIMENTO') é renderizado como elemento gráfico
    e não é capturado como texto pelo docling — comum em alguns layouts de fatura.
    Retorna (data, valor) ou (None, None) se não encontrado.
    """
    m = _DATE_BEFORE_CURRENCY_RE.search(text)
    if not m:
        return None, None
    dt = None
    try:
        d, mo, y = m.group(1).split("/")
        dt = date(int(y), int(mo), int(d))
    except ValueError:
        pass
    return dt, parse_currency(m.group(2))


def extract_date_ptbr_abbrev(text: str) -> Optional[date]:
    """
    Extrai a primeira data no formato 'DD MÊS AAAA' com mês abreviado em
    português (ex: '12 JAN 2022'). Comum em layouts de fatura de energia que
    não imprimem rótulos de data, apenas o valor solto no texto.
    """
    m = _DATE_PTBR_ABBREV_RE.search(text)
    if not m:
        return None
    month = _MONTH_ABBR_PT.get(m.group(2).upper())
    if not month:
        return None
    try:
        return date(int(m.group(3)), month, int(m.group(1)))
    except ValueError:
        return None


def extract_all_dates_ptbr_abbrev(text: str) -> list[date]:
    """Extrai todas as datas no formato 'DD MÊS AAAA' (abreviado, PT-BR), em ordem de aparição."""
    result: list[date] = []
    seen: set[date] = set()
    for m in _DATE_PTBR_ABBREV_RE.finditer(text):
        month = _MONTH_ABBR_PT.get(m.group(2).upper())
        if not month:
            continue
        try:
            d = date(int(m.group(3)), month, int(m.group(1)))
        except ValueError:
            continue
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def has_hex_signature(text: str) -> bool:
    """Detecta o layout de fatura que usa assinatura hexadecimal em vez de chave de acesso."""
    return bool(_HEX_SIGNATURE_RE.search(text))


def extract_number_after_hex_signature(text: str) -> Optional[str]:
    """
    Extrai o número da nota fiscal em layouts que usam assinatura hexadecimal
    sem rótulo (ex: certas faturas de energia da Enel). O número aparece como
    o segundo código numérico (9 dígitos) após a assinatura hex.
    """
    m = _HEX_SIGNATURE_THEN_NUMBER_RE.search(text)
    if not m:
        return None
    return str(int(m.group(1)))  # remove zeros à esquerda


def extract_boleto_line(text: str) -> Optional[str]:
    """Extrai linha digitável do boleto."""
    m = _BOLETO_LINE_RE.search(text)
    return m.group(0).strip() if m else None


def extract_date(text: str) -> Optional[date]:
    """Tenta extrair a primeira data válida no formato DD/MM/AAAA ou AAAA-MM-DD."""
    m = _DATE_DMY_RE.search(text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    m = _DATE_YMD_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    return None


def extract_all_dates(text: str) -> list[date]:
    """Extrai todas as datas únicas encontradas no texto, em ordem de aparição."""
    seen: set[date] = set()
    result: list[date] = []

    for m in _DATE_DMY_RE.finditer(text):
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if d not in seen:
                seen.add(d)
                result.append(d)
        except ValueError:
            pass

    for m in _DATE_YMD_RE.finditer(text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if d not in seen:
                seen.add(d)
                result.append(d)
        except ValueError:
            pass

    return result


def parse_currency(value_str: str) -> Optional[float]:
    """
    Converte string numérica brasileira (ou americana) em float
    (ex: '1.234,56' → 1234.56; '49.00' → 49.0).

    Apesar do nome (mantido por compatibilidade — usado em todo o código para
    valores em R$), também é usado para quantidades técnicas com precisão
    diferente de 2 casas decimais (ex: kWh com 3 casas: '1.294,839'). Por
    isso aceita qualquer número de dígitos após o separador decimal, não
    apenas 2 — uma fatura de energia frequentemente mistura valores
    monetários (2 casas) e quantidades de consumo/demanda (3+ casas) nas
    mesmas tabelas de itens.

    Detecta automaticamente formato BR (1.234,56) e americano/US (1,234.56
    ou 49.00) — algumas faturas usam um formato em uma seção e outro em
    outra (ex: anotações de retenção de impostos em formato americano numa
    DANFE cujo restante do documento usa formato BR).

    Trata sinal de menos tanto no início ("-1.234,56") quanto no fim
    ("1.234,56-"), comum em linhas de dedução/retenção em faturas de energia.
    """
    s = value_str.strip().replace("R$", "").replace(" ", "")

    negative = False
    if s.startswith("-"):
        negative = True
        s = s[1:]
    elif s.endswith("-"):
        negative = True
        s = s[:-1]

    value: Optional[float] = None

    # Formato BR: pontos como milhar, vírgula como decimal (qualquer nº de
    # casas)  →  1.234,56  ou  1.294,839
    if re.match(r"^\d{1,3}(\.\d{3})*,\d+$", s):
        value = float(s.replace(".", "").replace(",", "."))

    # Formato US: vírgulas como milhar, ponto como decimal  →  1,234.56 ou 49.00
    elif re.match(r"^\d{1,3}(,\d{3})*\.\d+$", s):
        value = float(s.replace(",", ""))

    # Somente vírgula (sem milhar): 1234,56 ou 1234,839
    elif "," in s and "." not in s:
        try:
            value = float(s.replace(",", "."))
        except ValueError:
            value = None

    # Somente ponto ou inteiro puro: 1234.56 / 1234
    else:
        try:
            value = float(s)
        except ValueError:
            value = None

    if value is None:
        return None
    return -value if negative else value


def parse_currency_or_flag(raw: Optional[str]) -> float | str | None:
    """
    Converte string numérica em float, distinguindo ausência de erro de leitura:
      - raw is None (nenhum trecho casou com o padrão de extração) → None
        (ausência genuína — o dado não está na fatura)
      - raw não é None, mas parse_currency() não conseguiu convertê-lo em
        número válido → READ_ERROR_SENTINEL (HÁ um valor impresso na fatura
        para esse campo, mas o parser não conseguiu interpretá-lo)
      - raw é convertível → o float correspondente

    Use esta função (em vez de parse_currency) nos pontos de extração onde
    já se confirmou que um rótulo/âncora foi encontrado no texto — ali a
    diferença entre "ausente" e "ilegível" é significativa.
    """
    if raw is None:
        return None
    value = parse_currency(raw)
    return value if value is not None else READ_ERROR_SENTINEL


def extract_currency(text: str) -> Optional[float]:
    """Extrai o primeiro valor monetário do texto."""
    m = _CURRENCY_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        return parse_currency(raw)
    return None


def extract_all_currencies(text: str) -> list[float]:
    """Extrai todos os valores monetários únicos do texto."""
    seen: set[float] = set()
    result: list[float] = []
    for m in _CURRENCY_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        v = parse_currency(raw)
        if v is not None and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    return m.group(0).strip() if m else None


def extract_br_state(text: str) -> Optional[str]:
    """
    Extrai sigla de estado brasileiro válida do texto.

    Prioridade:
      1. Rótulo explícito  "UF: SP" / "ESTADO: MG" — mais confiável, evita
         capturar siglas de outros contextos (ex: abreviações de empresas,
         nomes de colunas, unidades de medida).
      2. Padrão CIDADE/UF  "GUARATINGUETA/SP" — comum em endereços NF-e.
      3. Primeira sigla solta válida — fallback quando não há rótulo nem
         padrão de barra (comportamento original).
    """
    # 1. Rótulo explícito
    m = re.search(r"\bU\.?F\.?\s*[:\-]?\s*([A-Z]{2})\b", text)
    if m and m.group(1) in _BR_STATES:
        return m.group(1)
    m = re.search(r"\bESTADO\s*[:\-]?\s*([A-Z]{2})\b", text, re.IGNORECASE)
    if m and m.group(1) in _BR_STATES:
        return m.group(1)

    # 2. Padrão CIDADE/UF (ou CIDADE - UF) em endereços
    m = re.search(r"[A-ZÀ-Ú]{3,}[^/\n]{0,30}[/ -]([A-Z]{2})\b", text)
    if m and m.group(1) in _BR_STATES:
        return m.group(1)

    # 3. Primeira sigla solta válida
    for m in re.finditer(r"\b([A-Z]{2})\b", text):
        if m.group(1) in _BR_STATES:
            return m.group(1)
    return None


def extract_labeled_value(text: str, label_pattern: str, value_pattern: str) -> Optional[str]:
    """
    Extrai o valor que aparece após um rótulo (label_pattern) no texto.
    Útil para campos no formato  'Rótulo: Valor'  ou  'Rótulo\\nValor'.
    """
    combined = re.compile(
        rf"(?:{label_pattern})\s*[:\-]?\s*({value_pattern})",
        re.IGNORECASE | re.MULTILINE,
    )
    m = combined.search(text)
    return m.group(1).strip() if m else None


def normalize_text(text: str) -> str:
    """Remove espaços múltiplos e normaliza quebras de linha."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_lines_after_label(
    text: str,
    label_pattern: str,
    n_lines: int = 3,
    dominant: bool = False,
) -> list[str]:
    """
    Retorna até n_lines linhas que aparecem logo após a linha que casa com label_pattern.
    Se dominant=True, a linha só é considerada se o label ocupa >50% do conteúdo
    (evita falsos positivos em linhas como "0 - EMITENTE     2 - TERCEIROS ...").
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        m = re.search(label_pattern, stripped_line, re.IGNORECASE)
        if not m:
            continue
        if dominant:
            ratio = len(m.group(0)) / max(len(stripped_line), 1)
            if ratio < 0.5:
                continue
        candidates = []
        for j in range(i + 1, min(i + 1 + n_lines, len(lines))):
            stripped = lines[j].strip()
            if stripped:
                candidates.append(stripped)
        return candidates
    return []


def strip_label_prefix(name: str, label_patterns: list[str]) -> str:
    """Remove prefixos de rótulo de um nome capturado (ex: 'NOME/RAZÃO SOCIAL Foo' → 'Foo')."""
    for p in label_patterns:
        name = re.sub(rf"^{p}\s*", "", name, flags=re.IGNORECASE).strip()
    return name


# ---------------------------------------------------------------------------
# Métricas de energia elétrica (BI) — extração direta, nunca calculada
# ---------------------------------------------------------------------------

def extract_consumer_unit(text: str) -> Optional[str]:
    """
    Extrai o código da Unidade Consumidora (UC) — identifica o ponto de
    conexão/instalação física. NÃO deve ser confundido com o "Número do
    Cliente" (identifica a conta/contrato junto à concessionária — ver
    extract_client_number). Tenta, em ordem de confiabilidade:
      1. "UC <código>" (ex: Amazonas Energia)
      2. "Nº DO CLIENTE ... Nº DA INSTALAÇÃO <cliente> <instalação>" (CEMIG)
         — captura especificamente o valor da INSTALAÇÃO, não do cliente
      3. "INSTALAÇÃO <código>" ou "UNID. CONSUMIDORA <código>" isolados
    """
    m = re.search(r"\bUC\s+([\d][\d.\-]*)", text)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"N[ºo°]\s*D[OA]\s*CLIENTE\s*\n?\s*N[ºo°]?\s*\n?\s*DA\s*\n?\s*"
        r"INSTALA[ÇC][ÃA]O\s*\n?\s*\d+\s*\n?\s*(\d+)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)

    m = re.search(
        r"(?:N[ºo°]\s*(?:DA\s*)?)?INSTALA[ÇC][ÃA]O\s*[:\s]*\n?\s*(\d{4,})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)

    m = re.search(
        r"UNID(?:ADE)?\.?\s*CONSUMIDORA\s*[:\s]*\n?\s*(\d{4,})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return None


def extract_client_number(text: str) -> Optional[str]:
    """
    Extrai o "Número do Cliente" — identifica a conta/contrato do cliente
    junto à concessionária. É um identificador DIFERENTE da Unidade
    Consumidora (UC/Instalação — ver extract_consumer_unit): um mesmo
    cliente pode ter várias UCs associadas à mesma conta.
    """
    m = re.search(r"N[ºo°]\s+do\s+cliente\s*:?\s*\n?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # CEMIG: "Nº DO CLIENTE ... Nº DA INSTALAÇÃO <cliente> <instalação>"
    # — aqui queremos especificamente o primeiro valor (cliente).
    m = re.search(
        r"N[ºo°]\s*D[OA]\s*CLIENTE\s*\n?\s*N[ºo°]?\s*\n?\s*DA\s*\n?\s*"
        r"INSTALA[ÇC][ÃA]O\s*\n?\s*(\d+)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return None


def extract_consumption_kwh(text: str) -> tuple[float | str | None, float | str | None]:
    """
    Extrai consumo ativo medido (kWh) — ponta e fora de ponta.

    Cobre dois formatos de fatura distintos:
      a) Enel SP / formato ANEEL antigo: "CONSUMO ATIVO PONTA TE/TUSD \\n KWH \\n <qty>"
      b) CEMIG NF-e / formato moderno:   "Consumo de energia elétrica HP \\n kWh \\n <qty>"

    Retorna (consumo_ponta_kwh, consumo_fora_ponta_kwh); cada um pode ser
    None (rótulo ausente), float (extraído) ou READ_ERROR_SENTINEL (rótulo
    encontrado, valor ilegível).

    Nota: quando o mesmo consumo aparece em mais de uma linha de detalhe
    tarifário (ex: TE + TUSD = dois itens "CONSUMO ATIVO PONTA"), ambas têm
    a mesma quantidade — usamos o primeiro match para evitar dupla contagem.
    """
    # --- Ponta (HP) ---
    # Padrão a) Enel SP: CONSUMO ATIVO PONTA TE/TUSD
    _PEAK_PATTERNS = [
        r"CONSUMO\s+ATIVO\s+PONTA\s+(?:TE|TUSD)\s*\n?\s*KWH\s*\n?\s*([\d.,]+)",
        # Padrão b) CEMIG NF-e: "Consumo de energia elétrica HP" (linha-por-linha ou inline)
        r"Consumo\s+de\s+energia\s+el[eé]trica\s+HP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*([\d.,]+)",
        r"Consumo\s+de\s+energia\s+el[eé]trica\s+HP\b\s+[Kk][Ww][Hh]\s+([\d.,]+)",
    ]
    peak = None
    for p in _PEAK_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            peak = parse_currency_or_flag(m.group(1))
            break

    # --- Fora de Ponta (HFP) ---
    _OFFPEAK_PATTERNS = [
        r"CONSUMO\s+ATIVO\s+F\.?\s*PONTA\s+(?:TE|TUSD)\s*\n?\s*KWH\s*\n?\s*([\d.,]+)",
        # Padrão b) CEMIG NF-e: "Consumo de energia elétrica HFP"
        r"Consumo\s+de\s+energia\s+el[eé]trica\s+HFP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*([\d.,]+)",
        r"Consumo\s+de\s+energia\s+el[eé]trica\s+HFP\b\s+[Kk][Ww][Hh]\s+([\d.,]+)",
    ]
    offpeak = None
    for p in _OFFPEAK_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            offpeak = parse_currency_or_flag(m.group(1))
            break

    return peak, offpeak


def extract_third_party_energy_kwh(text: str) -> tuple[float | str | None, float | str | None]:
    """
    Extrai energia adquirida de comercializadora (ACL), separada por período
    tarifário — presente somente em faturas DISTRIBUIDORA_MLE onde a energia
    da comercializadora aparece como crédito/abatimento.

    Rótulos buscados: "Energia Terc Comercializad HP" e "... HFP"
    (a quantidade em kWh é sempre positiva; o valor R$ pode ser negativo
    porque representa um abatimento no débito total da fatura).

    Retorna (ponta_kwh, fora_ponta_kwh).
    """
    _PEAK_TERC_PATTERNS = [
        r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*(-?[\d.,]+)",
        r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HP\b\s+[Kk][Ww][Hh]\s+(-?[\d.,]+)",
        r"ENERGIA\s+TERC[^\n]*HP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*(-?[\d.,]+)",
    ]
    peak = None
    for p in _PEAK_TERC_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = parse_currency_or_flag(m.group(1))
            peak = abs(raw) if isinstance(raw, float) else raw
            break

    _OFFPEAK_TERC_PATTERNS = [
        r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HFP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*(-?[\d.,]+)",
        r"Energia\s+Terc[a-z]*\s+Comercializad[a-z]*\s+HFP\b\s+[Kk][Ww][Hh]\s+(-?[\d.,]+)",
        r"ENERGIA\s+TERC[^\n]*HFP\b[^\n]*\n?\s*[Kk][Ww][Hh]\s*[\n\r]*\s*(-?[\d.,]+)",
    ]
    offpeak = None
    for p in _OFFPEAK_TERC_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = parse_currency_or_flag(m.group(1))
            offpeak = abs(raw) if isinstance(raw, float) else raw
            break

    return peak, offpeak


def extract_measured_demand_kw(text: str) -> float | str | None:
    """
    Extrai a demanda medida (kW) a partir do item de fatura "DEMANDA" —
    exige que a linha seja exatamente "DEMANDA" (não "ULTRAPASSAGEM DEMANDA"
    nem outras variantes), para evitar capturar a quantidade de ultrapassagem
    em vez da demanda medida.

    Em tarifa Horosazonal Verde existe um único valor de demanda (sem
    distinção ponta/fora-ponta); ver EnergyMetrics para detalhes.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper() != "DEMANDA":
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            if lines[j].strip().upper() == "KW" and j + 1 < len(lines):
                return parse_currency_or_flag(lines[j + 1].strip())
    return None


def extract_demand_overage_value(text: str) -> float | str | None:
    """
    Extrai o valor (R$) do item de fatura "ULTRAPASSAGEM DEMANDA" — a
    penalidade por exceder a demanda contratada. Estrutura do item:
    "ULTRAPASSAGEM DEMANDA / KW / <quant> / <preço unit> / <valor R$> / ...".
    """
    m = re.search(
        r"ULTRAPASSAGEM\s+DEMANDA\s*\n?\s*KW\s*\n?\s*[\d.,]+\s*\n?\s*[\d.,]+\s*\n?\s*([\d.,]+)",
        text, re.IGNORECASE,
    )
    return parse_currency_or_flag(m.group(1)) if m else None


def extract_reactive_excess(text: str) -> dict[str, float | str | None]:
    """
    Extrai energia reativa excedente (UFER — Unidade para Faturamento de
    Energia Reativa) e o respectivo valor (R$) da multa por baixo fator de
    potência, separados por ponta/fora-ponta. Estrutura do item:
    "UFER PONTA TE / KWH / <quant kWh> / <preço unit> / <valor R$> / ...".

    Retorna um dict com as chaves: peak_kwh, offpeak_kwh, peak_value, offpeak_value.
    """
    result: dict[str, float | str | None] = {
        "peak_kwh": None, "offpeak_kwh": None,
        "peak_value": None, "offpeak_value": None,
    }

    m = re.search(
        r"UFER\s+PONTA\s+TE\s*\n?\s*KWH\s*\n?\s*([\d.,]+)\s*\n?\s*[\d.,]+\s*\n?\s*([\d.,]+)",
        text, re.IGNORECASE,
    )
    if m:
        result["peak_kwh"] = parse_currency_or_flag(m.group(1))
        result["peak_value"] = parse_currency_or_flag(m.group(2))

    m = re.search(
        r"UFER\s+FORA\s+PONTA\s+TE\s*\n?\s*KWH\s*\n?\s*([\d.,]+)\s*\n?\s*[\d.,]+\s*\n?\s*([\d.,]+)",
        text, re.IGNORECASE,
    )
    if m:
        result["offpeak_kwh"] = parse_currency_or_flag(m.group(1))
        result["offpeak_value"] = parse_currency_or_flag(m.group(2))

    return result


def extract_tarifa_azul_demand_nf3e(text: str) -> tuple[float | str | None, float | str | None]:
    """
    Extrai demanda medida ponta e fora-ponta (kW) para DANF3E/NF3E em
    Tarifa Azul.

    No DANF3E, o Docling lê a tabela de itens coluna por coluna, gerando
    texto com "Demanda P. kW" e "Demanda FP. kW" em linhas consecutivas.
    Os valores de quantidade (kW inteiros) aparecem após 4 colunas numéricas
    intermediárias (2 alíquotas ICMS + 2 preços unitários).
    """
    peak = offpeak = None

    # Padrão 1: dois rótulos de demanda seguidos, depois 4 cols e os valores
    m = re.search(
        r"DEMANDA\s+P(?:ONTA)?\s*\.?\s*KW\s*\n"
        r"DEMANDA\s+F(?:P|ORA\s*PONTA)?\s*\.?\s*KW\s*\n"
        r"(?:[^\n]+\n){2,8}?"
        r"\s*([\d]+(?:[.,]\d+)?)\s*\n"
        r"\s*([\d]+(?:[.,]\d+)?)",
        text, re.IGNORECASE,
    )
    if m:
        v1 = parse_currency(m.group(1))
        v2 = parse_currency(m.group(2))
        # Os dois primeiros valores numéricos após os 4 cols são as quantidades kW
        # Heurística: o valor de demanda kW é tipicamente < 10.000 e sem centavos
        if v1 is not None and v1 < 10000:
            peak = v1
        if v2 is not None and v2 < 10000:
            offpeak = v2

    # Padrão 2: rótulos individuais com valor próximo (UTILITY ou OCR livre)
    # "DEMANDA PONTA kW ... 501"
    if peak is None:
        m = re.search(
            r"DEMANDA\s+P(?:ONTA)?\s*\.?\s*KW\b[^\n]*\n"
            r"(?:[^\n]*\n){0,6}?"
            r"\s*([\d]{1,6})\s*\n",
            text, re.IGNORECASE,
        )
        if m:
            v = parse_currency(m.group(1))
            if v is not None and v < 10000:
                peak = v

    if offpeak is None:
        m = re.search(
            r"DEMANDA\s+F(?:P|ORA\s*PONTA)?\s*\.?\s*KW\b[^\n]*\n"
            r"(?:[^\n]*\n){0,6}?"
            r"\s*([\d]{1,6})\s*\n",
            text, re.IGNORECASE,
        )
        if m:
            v = parse_currency(m.group(1))
            if v is not None and v < 10000:
                offpeak = v

    return peak, offpeak


def extract_energy_quantity_mwh(text: str) -> float | None:
    """
    Extrai a quantidade de energia (MWh) de NF-e de comercializadora.

    No DANFE de comercializadora, a coluna QUANT. registra o volume em MWh
    (ex: 259,3100). Docling às vezes inverte header/dado, colocando o valor
    da quantidade no nome da coluna do DataFrame. Esta função busca o valor
    de quantidade MWh tanto no texto livre quanto no padrão do cabeçalho
    da tabela ("QUANT. V.TOTAL <valor>").
    """
    # Padrão 1: "QUANT. [V.TOTAL] <decimal>" — cabeçalho de tabela NF-e
    m = re.search(
        r"\bQUANT(?:\.|\s+QUANTIDADE)?\s*(?:V\.?\s*(?:UNIT|TOTAL)\s+)?([\d]+[.,][\d]+)\b",
        text, re.IGNORECASE,
    )
    if m:
        v = parse_currency(m.group(1))
        if v is not None and 0.001 < v < 99999:
            return v

    # Padrão 2: linha com unidade MWh próxima a um número
    m = re.search(
        r"([\d.,]+)\s*\n?\s*MW\s*h?\b",
        text, re.IGNORECASE,
    )
    if m:
        v = parse_currency(m.group(1))
        if v is not None and 0.001 < v < 99999:
            return v

    return None

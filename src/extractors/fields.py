"""
Padrões regex e funções de extração de campos individuais de faturas brasileiras.
Toda extração aqui é puramente determinística — sem LLM.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

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
    """Converte string monetária brasileira em float (ex: '1.234,56' → 1234.56)."""
    s = value_str.strip().replace("R$", "").replace(" ", "")

    # Formato BR: pontos como milhar, vírgula como decimal  →  1.234,56
    if re.match(r"^\d{1,3}(\.\d{3})*,\d{2}$", s):
        return float(s.replace(".", "").replace(",", "."))

    # Formato US: vírgulas como milhar, ponto como decimal  →  1,234.56
    if re.match(r"^\d{1,3}(,\d{3})*\.\d{2}$", s):
        return float(s.replace(",", ""))

    # Somente vírgula (sem milhar): 1234,56
    if "," in s and "." not in s:
        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass

    # Somente ponto: 1234.56
    try:
        return float(s)
    except ValueError:
        return None


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
    """Extrai sigla de estado brasileiro válida do texto."""
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

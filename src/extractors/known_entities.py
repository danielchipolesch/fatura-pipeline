"""
Cadastro de entidades conhecidas (fornecedores e clientes recorrentes).

Complementa a extração por regex de duas formas:
  1. Por CNPJ: quando o CNPJ do emitente (derivado de forma confiável a
     partir da chave de acesso da NF-e) ou a raiz do CNPJ do destinatário
     é conhecida, usamos os dados cadastrados aqui em vez de depender de
     texto potencialmente corrompido por OCR, mascarado por privacidade,
     ou de rótulos ausentes no layout específico do PDF.
  2. Por nome literal: alguns layouts não imprimem o CNPJ do emitente em
     lugar nenhum do documento (e também não trazem chave de acesso
     convencional). Nesses casos, o nome da empresa é a única âncora
     disponível — buscamos o texto literal e atribuímos os dados completos
     do cadastro.

Mantenha esta lista atualizada conforme novos fornecedores/clientes
recorrentes aparecerem nas faturas processadas — cada entrada nova reduz
a necessidade de acionar o fallback via LLM.
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict


class SupplierInfo(TypedDict, total=False):
    name: str
    cnpj: str
    city: str
    state: str
    # Papel do fornecedor no setor elétrico — usado pela inferência de
    # tipo_fatura_operacional para distinguir COMERCIALIZADORA_MLE de
    # DISTRIBUIDORA_MLE sem depender de padrões textuais quando o CNPJ
    # do emitente está disponível (extraído da chave de acesso NF-e).
    # Valores: "distribuidora" | "comercializadora"
    role: str


# Chave: CNPJ do emitente, 14 dígitos sem formatação (extraído da chave de
# acesso NF-e, posições 6-19 — é a fonte mais confiável pois sobrevive a OCR).
KNOWN_SUPPLIERS: dict[str, SupplierInfo] = {
    "06981176000158": {
        "name": "CEMIG GERAÇÃO E TRANSMISSÃO S.A.",
        "cnpj": "06.981.176/0001-58",
        "city": "BELO HORIZONTE",
        "state": "MG",
        "role": "comercializadora",  # CEMIG GT atua como comercializadora no ACL
    },
    "06981180000116": {
        "name": "CEMIG DISTRIBUIÇÃO S.A.",
        "cnpj": "06.981.180/0001-16",
        "city": "BELO HORIZONTE",
        "state": "MG",
        "role": "distribuidora",
    },
    "07047251000170": {
        "name": "Companhia Energética do Ceará - Enel Ceará",
        "cnpj": "07.047.251/0001-70",
        "city": "FORTALEZA",
        "state": "CE",
        "role": "distribuidora",
    },
    "61695227000193": {
        "name": "ENEL DISTRIBUIÇÃO SÃO PAULO",
        "cnpj": "61.695.227/0001-93",
        "city": "SÃO PAULO",
        "state": "SP",
        "role": "distribuidora",
    },
    "02341467000120": {
        "name": "AMAZONAS ENERGIA",
        "cnpj": "02.341.467/0001-20",
        "city": "MANAUS",
        "state": "AM",
        "role": "distribuidora",
    },
    "10732440000197": {
        "name": "COMERC POWER TRADING LTDA",
        "cnpj": "10.732.440/0001-97",
        "city": "SÃO PAULO",
        "state": "SP",
        "role": "comercializadora",
    },
    "60444437000146": {
        "name": "LIGHT SERVIÇOS DE ELETRICIDADE S.A.",
        "cnpj": "60.444.437/0001-46",
        "city": "RIO DE JANEIRO",
        "state": "RJ",
        "role": "distribuidora",
    },
    "02016440000162": {
        "name": "RGE SUL DISTRIBUIDORA DE ENERGIA S.A.",
        "cnpj": "02.016.440/0001-62",
        "city": "SÃO LEOPOLDO",
        "state": "RS",
        "role": "distribuidora",
    },
}

# Fallback por nome literal: usado quando o layout não imprime o CNPJ do
# emitente em nenhum lugar do documento (nem sequer codificado em chave de
# acesso). Cada entrada mapeia um padrão regex do nome para o CNPJ completo
# (que deve existir em KNOWN_SUPPLIERS) — permite reaproveitar os mesmos
# dados cadastrais por CNPJ.
KNOWN_SUPPLIER_NAME_PATTERNS: list[tuple[str, str]] = [
    (r"ENEL\s+DISTRIBUI[ÇC][ÃA]O\s+S[ÃA]O\s+PAULO", "61695227000193"),
    (r"AMAZONAS\s+ENERGIA", "02341467000120"),
    (r"CEMIG\s+DISTRIBUI[ÇC][ÃA]O", "06981180000116"),
    (r"CEMIG\s+GERA[ÇC][ÃA]O\s+E\s+TRANSMISS[ÃA]O", "06981176000158"),
    (r"COMPANHIA\s+ENERG[ÉE]TICA\s+D.?\s*O?\s*CEAR[ÁA]", "07047251000170"),
    (r"COMERC\s+POWER\s+TRADING", "10732440000197"),
    (r"LIGHT\s+SERVI[ÇC]OS\s+DE\s+ELETRICIDADE", "60444437000146"),
    (r"RGE\s+SUL\s+DISTRIBUIDORA", "02016440000162"),
]

# Chave: raiz do CNPJ do destinatário (8 primeiros dígitos — identifica a
# entidade legal independentemente da unidade/filial/instalação específica).
KNOWN_CUSTOMER_ROOTS: dict[str, str] = {
    "00394429": "COMANDO DA AERONAUTICA",
}

# Fallback por nome literal para o cliente: cobre casos em que o CNPJ do
# destinatário está mascarado no PDF (ex: "00.***.***/****-04", comum em
# faturas de concessionárias por privacidade) e portanto não há dígitos
# suficientes para buscar pela raiz do CNPJ.
KNOWN_CUSTOMER_NAME_PATTERNS: list[str] = [
    r"COMANDO\s+DA\s+AERONAUTICA",
]


def lookup_supplier(cnpj: str) -> Optional[SupplierInfo]:
    """Busca dados do fornecedor por CNPJ completo (14 dígitos, com ou sem pontuação)."""
    digits = re.sub(r"[^\d]", "", cnpj)
    return KNOWN_SUPPLIERS.get(digits)


def lookup_supplier_by_name(text: str) -> Optional[SupplierInfo]:
    """
    Busca dados do fornecedor pelo nome literal presente no texto — usado
    quando o documento não traz CNPJ do emitente em lugar nenhum (nem
    codificado em chave de acesso), comum em alguns layouts de fatura de
    energia que omitem o CNPJ da concessionária no corpo do texto.
    """
    for pattern, cnpj in KNOWN_SUPPLIER_NAME_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return KNOWN_SUPPLIERS.get(cnpj)
    return None


def lookup_customer_name(cnpj: str) -> Optional[str]:
    """Busca nome do cliente pela raiz do CNPJ (8 primeiros dígitos)."""
    digits = re.sub(r"[^\d]", "", cnpj)
    if len(digits) < 8:
        return None
    return KNOWN_CUSTOMER_ROOTS.get(digits[:8])


def lookup_customer_name_by_text(text: str) -> Optional[str]:
    """
    Busca o nome do cliente pelo texto literal — usado quando o CNPJ do
    destinatário está mascarado no PDF (ex: "00.***.***/****-04") e portanto
    não há como identificar a entidade pela raiz do CNPJ.
    """
    for pattern in KNOWN_CUSTOMER_NAME_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return None

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InvoiceLayout(str, Enum):
    NFE = "nfe"          # Nota Fiscal Eletrônica (DANFE)
    NFSE = "nfse"        # Nota Fiscal de Serviços Eletrônica
    BOLETO = "boleto"    # Boleto bancário
    UTILITY = "utility"  # Conta de concessionária (energia, água, gás)
    GENERIC = "generic"  # Fatura genérica / outros formatos


class ParsingMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_FALLBACK = "llm_fallback"
    HYBRID = "hybrid"  # Determinístico + complementado pelo LLM


class Address(BaseModel):
    street: Optional[str] = None
    number: Optional[str] = None
    complement: Optional[str] = None
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "Brasil"


class Party(BaseModel):
    name: Optional[str] = None
    cnpj: Optional[str] = None
    cpf: Optional[str] = None
    ie: Optional[str] = None   # Inscrição Estadual
    im: Optional[str] = None   # Inscrição Municipal
    # Sempre um objeto Address (nunca None) — garante chaves estáveis no JSON
    # de saída, com valores null quando a informação não estiver disponível.
    address: Address = Field(default_factory=Address)
    email: Optional[str] = None
    phone: Optional[str] = None


class LineItem(BaseModel):
    code: Optional[str] = None
    description: str
    ncm: Optional[str] = None    # Código NCM (NF-e)
    cfop: Optional[str] = None   # CFOP (NF-e)
    unit: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    discount: Optional[float] = None
    tax_rate: Optional[float] = None
    total: Optional[float] = None


class Tax(BaseModel):
    name: str                    # ICMS, ISS, IPI, PIS, COFINS, etc.
    base: Optional[float] = None
    rate: Optional[float] = None
    amount: Optional[float] = None


class Invoice(BaseModel):
    """
    Layout padrão de saída — usado uniformemente por TODAS as faturas,
    independentemente do layout original do PDF (NF-e, NFS-e, boleto,
    concessionária, genérico) e do método de extração (determinístico ou
    fallback via LLM).

    Todas as chaves deste schema sempre existem no JSON de saída. Quando um
    valor não pôde ser identificado na fatura, a chave permanece presente
    com valor null — nunca é omitida. Isso garante que qualquer sistema
    consumidor possa depender de uma estrutura estável, sem precisar tratar
    chaves ausentes.
    """

    # Identificação
    invoice_number: Optional[str] = None
    invoice_layout: InvoiceLayout = InvoiceLayout.GENERIC
    access_key: Optional[str] = None   # Chave de acesso NF-e (44 dígitos)
    series: Optional[str] = None

    # Datas
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    service_period_start: Optional[date] = None
    service_period_end: Optional[date] = None

    # Partes — sempre um objeto Party (nunca None), mesmo princípio do Address acima.
    supplier: Party = Field(default_factory=Party)
    customer: Party = Field(default_factory=Party)

    # Itens e tributos
    line_items: List[LineItem] = Field(default_factory=list)
    taxes: List[Tax] = Field(default_factory=list)

    # Totais
    subtotal: Optional[float] = None
    discount: Optional[float] = None
    total_taxes: Optional[float] = None
    total: Optional[float] = None
    currency: str = "BRL"

    # Pagamento
    payment_method: Optional[str] = None
    payment_terms: Optional[str] = None
    bank_slip_barcode: Optional[str] = None  # Código de barras do boleto
    bank_slip_line: Optional[str] = None     # Linha digitável

    # Metadados de processamento
    source_file: str
    parsing_method: ParsingMethod = ParsingMethod.DETERMINISTIC

    # Score calculado imediatamente após a extração determinística — é este
    # valor que decide se o fallback via LLM é acionado (comparado contra
    # CONFIDENCE_THRESHOLD). Nunca é alterado depois, mesmo que o LLM
    # complemente o invoice — serve de registro de auditoria do "antes".
    confidence_score_initial: float = Field(default=0.0, ge=0.0, le=1.0)

    # Score final, recalculado após o enrich() do LLM fallback (quando
    # acionado). Para faturas resolvidas só com extração determinística,
    # este valor é idêntico a confidence_score_initial.
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    raw_text: Optional[str] = None
    notes: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

    @field_validator("access_key")
    @classmethod
    def validate_access_key(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            digits = "".join(c for c in v if c.isdigit())
            if len(digits) != 44:
                return None
            return digits
        return v

    @field_validator("bank_slip_barcode")
    @classmethod
    def validate_barcode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            digits = "".join(c for c in v if c.isdigit())
            if len(digits) not in (44, 47, 48):
                return None
            return digits
        return v

    def to_output_dict(self) -> dict:
        """
        Serializa para o dicionário de saída final, seguindo o layout padrão
        do schema. Todas as chaves do modelo são incluídas — campos sem valor
        aparecem como null — exceto `raw_text`, que é dado interno de
        depuração e nunca faz parte do schema de saída.
        """
        return self.model_dump(
            exclude={"raw_text"},
            exclude_none=False,
            mode="json",
        )

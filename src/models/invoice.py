from __future__ import annotations

from datetime import date
from enum import Enum
from typing import ClassVar, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Sentinela usada em campos numéricos para distinguir duas situações que
# `null` por si só não consegue diferenciar:
#   - valor null: o dado simplesmente não consta na fatura (ausência real)
#   - valor READ_ERROR_SENTINEL: HÁ um valor impresso na fatura para esse
#     campo, mas o parser determinístico não conseguiu interpretá-lo (ex:
#     formato numérico inesperado, rótulo capturou texto não numérico). É
#     um sinal de que o padrão de extração precisa ser refinado — diferente
#     de uma ausência legítima.
READ_ERROR_SENTINEL = "LEITURA_FALHOU"

# Tipo usado pelos campos numéricos sujeitos a essa distinção: número (lido
# com sucesso), None (ausente) ou READ_ERROR_SENTINEL (presente, mas ilegível).
NumericValue = Union[float, str, None]


class InvoiceLayout(str, Enum):
    NFE = "nfe"          # Nota Fiscal Eletrônica (DANFE)
    NFSE = "nfse"        # Nota Fiscal de Serviços Eletrônica
    BOLETO = "boleto"    # Boleto bancário
    UTILITY = "utility"  # Conta de concessionária (energia, água, gás)
    GENERIC = "generic"  # Fatura genérica / outros formatos


class TipoFaturaOperacional(str, Enum):
    """
    Tipo operacional da fatura de energia elétrica — derivado por inferência,
    não lido diretamente do documento.

    Importa para o dashboard porque uma UC em Mercado Livre gera DUAS faturas
    por competência (distribuidora + comercializadora). Somar ambas como
    "despesa" duplicaria o custo. O dashboard usa este campo para consolidar
    corretamente por UC/competência, sem dupla contagem nem subestimação.

      CATIVO             — Mercado cativo: fatura única, TE + TUSD juntos,
                           emitida pela concessionária/distribuidora local.
      DISTRIBUIDORA_MLE  — Mercado Livre, fatura da distribuidora: cobra apenas
                           uso da rede (TUSD); energia adquirida da comercializadora
                           aparece como abatimento/dedução ("Energia Terc.
                           Comercializada" ou "Energia ACL").
      COMERCIALIZADORA_MLE — Mercado Livre, fatura da comercializadora: cobra
                           apenas energia (TE) em MWh, sem TUSD nem demanda.
    """
    CATIVO               = "CATIVO"
    DISTRIBUIDORA_MLE    = "DISTRIBUIDORA_MLE"
    COMERCIALIZADORA_MLE = "COMERCIALIZADORA_MLE"


class ParsingMethod(str, Enum):
    SEMANTIC = "semantic"         # DoclingExtractor (estrutura nativa) como fonte principal, sem LLM
    DETERMINISTIC = "deterministic"  # Regex (DeterministicParser) como fonte principal, sem LLM
    HYBRID = "hybrid"             # Camadas determinísticas complementadas pelo LLM fallback


class Address(BaseModel):
    state: Optional[str] = None


class Party(BaseModel):
    name: Optional[str] = None
    cnpj: Optional[str] = None
    cpf: Optional[str] = None
    ie: Optional[str] = None   # Inscrição Estadual
    im: Optional[str] = None   # Inscrição Municipal
    # Sempre um objeto Address (nunca None) — garante chaves estáveis no JSON
    # de saída, com valores null quando a informação não estiver disponível.
    address: Address = Field(default_factory=Address)


class Tax(BaseModel):
    name: str                    # ICMS, ISS, IPI, PIS, COFINS, etc.
    base: NumericValue = None
    rate: NumericValue = None
    amount: NumericValue = None


class LineItem(BaseModel):
    code: Optional[str] = None
    description: str
    ncm: Optional[str] = None    # Código NCM (NF-e)
    cfop: Optional[str] = None   # CFOP (NF-e)
    unit: Optional[str] = None
    quantity: NumericValue = None
    unit_price: NumericValue = None
    discount: NumericValue = None
    tax_rate: NumericValue = None
    total: NumericValue = None
    # Tributos embutidos neste item (ICMS, PIS, COFINS por linha quando
    # a fatura detalha impostos por componente de energia, ex: CEMIG NF-e).
    # Lista vazia = imposto não detalhado por item (vai em Invoice.taxes).
    taxes: List[Tax] = Field(default_factory=list)


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

    Campos numéricos (ver NumericValue) distinguem dois tipos de "sem valor":
      - null: o dado não consta na fatura (ausência confirmada).
      - "LEITURA_FALHOU" (READ_ERROR_SENTINEL): HÁ um valor impresso na
        fatura para esse campo, mas o parser não conseguiu interpretá-lo —
        sinaliza que o padrão de extração precisa de ajuste, sem se
        confundir com uma ausência real.

    Sobre consumer_unit vs client_number:
      São identificadores DIFERENTES. A Unidade Consumidora (UC) identifica
      o ponto de conexão/instalação física. O Número do Cliente identifica
      a conta/contrato junto à concessionária — um mesmo cliente pode ter
      várias UCs. Ambos são metadados de identificação, não itens de cobrança,
      por isso ficam no nível do Invoice e não em line_items.
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

    # Identificadores da instalação de energia (metadados, não itens de cobrança)
    consumer_unit: Optional[str] = None  # Unidade Consumidora (UC) / Nº da Instalação
    client_number: Optional[str] = None  # Número do Cliente junto à concessionária (≠ UC)

    # Tipo operacional da fatura — inferido por regras (não lido diretamente
    # do documento). None quando a fatura não é de energia ou não há sinais
    # suficientes para classificar com segurança.
    tipo_fatura_operacional: Optional[TipoFaturaOperacional] = None

    # Itens e tributos
    # line_items: TODOS os componentes de cobrança da fatura (produtos, serviços,
    # componentes de energia: TUSD HP/HFP, TE HP/HFP, consumo, demanda, energia
    # terceirizada, multas, etc.). Cada item pode ter tributos embutidos (item.taxes)
    # quando a fatura detalha impostos por componente de energia.
    line_items: List[LineItem] = Field(default_factory=list)
    # taxes: apenas tributos cobrados de forma independente, sem vínculo com
    # um item específico (ex: ICMS, PIS, COFINS em NF-e com tabela de cálculo
    # separada). Quando os tributos aparecem embutidos nos itens, ficam em
    # item.taxes e este campo fica vazio.
    taxes: List[Tax] = Field(default_factory=list)

    # Totais
    subtotal: NumericValue = None
    discount: NumericValue = None
    total_taxes: NumericValue = None
    total: NumericValue = None
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

    # Ordem canônica dos campos no JSON de saída.
    # Campos não listados (ex: adicionados no futuro) vão ao final.
    _OUTPUT_KEY_ORDER: ClassVar[list[str]] = [
        # Identificação e classificação
        "invoice_number", "invoice_layout", "tipo_fatura_operacional",
        "access_key", "series",
        # Datas
        "issue_date", "due_date", "service_period_start", "service_period_end",
        # Partes
        "supplier", "customer",
        # Identificadores de instalação de energia
        "consumer_unit", "client_number",
        # Itens e tributos
        "line_items", "taxes",
        # Totais e moeda
        "subtotal", "discount", "total_taxes", "total", "currency",
        # Pagamento
        "payment_method", "payment_terms", "bank_slip_barcode", "bank_slip_line",
        # Metadados de processamento
        "source_file", "parsing_method",
        "confidence_score_initial", "confidence_score",
        "notes", "errors",
    ]

    def to_output_dict(self) -> dict:
        """
        Serializa para o dicionário de saída final.

        Transformações aplicadas (sem remover dados):
          - Campos reordenados em grupos semânticos (identificação → datas →
            partes → instalação → itens → totais → pagamento → metadados).

        Todas as chaves do modelo estão presentes no JSON de saída — campos
        sem valor aparecem como null. Exceção: `raw_text` (dado interno de
        depuração, nunca faz parte do schema de saída).
        """
        d = self.model_dump(exclude={"raw_text"}, exclude_none=False, mode="json")

        # Reconstrói o dict na ordem canônica; campos desconhecidos vão ao final
        ordered: dict = {k: d[k] for k in self._OUTPUT_KEY_ORDER if k in d}
        ordered.update({k: v for k, v in d.items() if k not in ordered})
        return ordered

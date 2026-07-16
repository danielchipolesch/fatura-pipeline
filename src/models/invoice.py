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


class Tax(BaseModel):
    name: str                    # ICMS, ISS, IPI, PIS, COFINS, etc.
    base: NumericValue = None
    rate: NumericValue = None
    amount: NumericValue = None


class EnergyMetrics(BaseModel):
    """
    Métricas técnicas específicas de faturas de energia elétrica, relevantes
    para BI e gestão de mercado de energia (cativo/livre). Todo campo aqui é
    extraído diretamente do texto da fatura — NUNCA calculado a partir de
    outros campos. Quando o dado não está impresso no documento (ex: fator
    de potência, ausente em todos os layouts mapeados até o momento), o
    campo permanece null — isso é informação válida (ausência confirmada),
    não uma falha de extração.

    Demanda medida: faturas em tarifa Horosazonal Verde têm um único valor
    de demanda (sem distinção ponta/fora-ponta — convencionalmente chamado
    "fora ponta" pelo padrão ANEEL/CEMIG/Enel). Faturas em tarifa Azul
    trariam ambos os valores separadamente, mas nenhuma amostra analisada
    até agora usa essa modalidade.

    consumer_unit vs client_number: são identificadores DIFERENTES e não
    devem ser confundidos. A Unidade Consumidora (UC) identifica o ponto de
    conexão/instalação física (rótulos: "UC", "Nº DA INSTALAÇÃO",
    "UNID. CONSUMIDORA"). O Número do Cliente identifica a conta/contrato
    do cliente junto à concessionária (rótulo: "Nº DO CLIENTE") — um mesmo
    cliente pode ter várias UCs, e vice-versa em alguns arranjos.
    """
    consumer_unit: Optional[str] = None  # Unidade Consumidora (UC) / Nº da Instalação
    client_number: Optional[str] = None  # Número do Cliente junto à concessionária (≠ UC)
    consumption_peak_kwh: NumericValue = None        # Consumo ativo medido na ponta (kWh)
    consumption_offpeak_kwh: NumericValue = None      # Consumo ativo medido fora de ponta (kWh)
    measured_demand_peak_kw: NumericValue = None      # Demanda medida na ponta (kW) — tarifa Azul
    measured_demand_offpeak_kw: NumericValue = None   # Demanda medida fora de ponta / única (kW)
    power_factor_measured: NumericValue = None        # Fator de potência medido
    reactive_energy_excess_peak_kwh: NumericValue = None     # UFER ponta (kWh)
    reactive_energy_excess_offpeak_kwh: NumericValue = None  # UFER fora de ponta (kWh)
    demand_overage_value: NumericValue = None         # Valor (R$) de ultrapassagem de demanda contratada
    reactive_penalty_peak_value: NumericValue = None      # Valor (R$) da multa reativa (UFER) na ponta
    reactive_penalty_offpeak_value: NumericValue = None   # Valor (R$) da multa reativa (UFER) fora de ponta
    # Energia adquirida de comercializadora (ACL) — só presente em faturas
    # DISTRIBUIDORA_MLE; representa o crédito de energia injetada pela
    # comercializadora, listado como abatimento na fatura da distribuidora.
    third_party_energy_peak_kwh: NumericValue = None      # Energia Terc. Comercializad HP (kWh)
    third_party_energy_offpeak_kwh: NumericValue = None   # Energia Terc. Comercializad HFP (kWh)


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

    # Métricas de energia para BI — sempre um objeto (nunca None), mesmo
    # princípio do Address/Party acima.
    energy: EnergyMetrics = Field(default_factory=EnergyMetrics)

    # Tipo operacional da fatura — inferido por regras (não lido diretamente
    # do documento). None quando a fatura não é de energia ou não há sinais
    # suficientes para classificar com segurança.
    tipo_fatura_operacional: Optional[TipoFaturaOperacional] = None

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
        # Itens, tributos e métricas de energia
        "line_items", "taxes", "energy",
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
            partes → itens → totais → pagamento → metadados).

        Todas as chaves do modelo estão presentes no JSON de saída — campos
        sem valor aparecem como null. Exceção: `raw_text` (dado interno de
        depuração, nunca faz parte do schema de saída).
        """
        d = self.model_dump(exclude={"raw_text"}, exclude_none=False, mode="json")

        # Reconstrói o dict na ordem canônica; campos desconhecidos vão ao final
        ordered: dict = {k: d[k] for k in self._OUTPUT_KEY_ORDER if k in d}
        ordered.update({k: v for k, v in d.items() if k not in ordered})
        return ordered

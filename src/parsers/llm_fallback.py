"""
Fallback LLM para campos que o parser determinístico não conseguiu extrair.

Utiliza Ollama como servidor de LLM open source — sem dependência de APIs externas
nem contratos comerciais. Os modelos rodam localmente, na infraestrutura da instituição.

Dois modos de uso (configurável via LLM_ENRICH_MODE):
  - "always" (padrão): enrich_nulls() sempre roda após extração determinística,
    enviando o JSON parcial + texto bruto ao LLM para preencher apenas campos null.
  - "threshold": enrich() só é chamado quando confidence_score < CONFIDENCE_THRESHOLD
    (comportamento original — útil para documentos OCR ou como fallback leve).

Princípios gerais:
  - Nunca sobrescreve campos já preenchidos pelo parser determinístico
  - Processamento SÍNCRONO por arquivo (sem paralelismo — preserva CPU)
  - O modelo é baixado automaticamente (pull lazy) na primeira chamada
  - Temperatura 0: saída determinística e estruturada em JSON

Modelos recomendados (configurar LLM_MODEL no .env):
  qwen2.5:3b   — Apache 2.0, multilingual, ~2 GB, padrão
  llama3.2:3b  — Meta Community License, ~2 GB
  mistral:7b   — Apache 2.0, mais capaz mas mais lento em CPU
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date
from typing import Optional

import requests
from loguru import logger

from src.extractors.fields import parse_currency
from src.models.invoice import Invoice, LineItem, ParsingMethod

try:
    from src.parsers.docling_loader import DocumentContent
except ImportError:
    DocumentContent = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Você é um especialista em extração de dados de faturas e documentos fiscais brasileiros.
Extraia o MÁXIMO de campos possível do texto da fatura abaixo — preencha todos os
campos do schema que conseguir identificar no texto, não apenas os marcados como
prioritários. Quanto mais campos preenchidos corretamente, melhor.
Responda SOMENTE com JSON válido e nada mais. Sem explicações, sem markdown, sem texto extra.
Se um campo realmente não estiver presente no texto, omita-o do JSON (não invente valores).
Valores monetários: número decimal (ex: 1234.56 — sem R$ ou separadores de milhar).
Datas: formato YYYY-MM-DD.
"""

_USER_PROMPT_TEMPLATE = """\
Os seguintes campos são PRIORITÁRIOS (o parser determinístico não conseguiu extraí-los):
{fields}

Mas extraia TODOS os campos abaixo que você conseguir identificar no texto da fatura,
não se limite apenas aos prioritários:

- invoice_number: string (número da nota/fatura)
- issue_date: string YYYY-MM-DD (data de emissão)
- due_date: string YYYY-MM-DD (data de vencimento)
- supplier_name: string (nome/razão social do emitente/fornecedor)
- supplier_cnpj: string formato XX.XXX.XXX/XXXX-XX
- supplier_cpf: string
- supplier_state: string (sigla UF do emitente, ex: SP, MG, RJ)
- customer_name: string (nome/razão social do destinatário/cliente)
- customer_cnpj: string formato XX.XXX.XXX/XXXX-XX
- customer_cpf: string
- customer_state: string (sigla UF do destinatário, ex: SP, MG, RJ)
- subtotal: number
- total: number (valor total a pagar)
- payment_method: string (forma de pagamento, ex: boleto, PIX, débito automático)
- payment_terms: string (condições de pagamento)
- line_items: lista de objetos {{"description": string, "quantity": number, "unit": string, "unit_price": number, "total": number}}
  com os componentes da fatura de energia (TUSD, Tarifa de Energia, Demanda Ponta/Fora Ponta,
  ICMS, PIS, COFINS, etc.) ou itens/serviços de nota fiscal; até 10 itens, sem omitir nenhum
- consumer_unit: string (código da unidade consumidora / código de instalação / número do ponto de entrega)
- client_number: string (número do cliente junto à concessionária — diferente da UC)

{context_section}Texto da fatura (até 6000 caracteres):
---
{text}
---
"""

_SECTION_CONTEXT_TEMPLATE = """\
Contexto específico para os campos prioritários:
---
{section_text}
---

"""

# Prompts do modo enriquecimento (sempre roda após extração determinística)

_ENRICH_SYSTEM_PROMPT = """\
Você é um especialista em extração de dados de faturas e documentos fiscais brasileiros.
Receberá um JSON com dados já extraídos de uma fatura — campos com valor null não foram
encontrados pelo parser. Sua tarefa é analisar o texto da fatura e preencher APENAS os
campos null que você conseguir identificar com certeza no texto.
NÃO altere campos que já possuem valor preenchido.
Responda SOMENTE com JSON válido. Sem explicações, sem markdown, sem texto extra.
Se um campo realmente não estiver presente no texto, omita-o do JSON — não invente valores.
Valores monetários: número decimal (ex: 1234.56 — sem R$ nem separadores de milhar).
Datas: formato YYYY-MM-DD.
"""

_ENRICH_USER_TEMPLATE = """\
Dados já extraídos da fatura (null = não encontrado pelo parser):
{partial_json}

Preencha apenas os campos null acima que você conseguir identificar no texto abaixo.
Retorne somente os campos que você vai preencher — omita os que já têm valor ou que
não foram encontrados no texto.

Campos disponíveis para preenchimento:
- invoice_number: string
- issue_date: string YYYY-MM-DD
- due_date: string YYYY-MM-DD
- service_period_start: string YYYY-MM-DD
- service_period_end: string YYYY-MM-DD
- supplier_name: string
- supplier_cnpj: string formato XX.XXX.XXX/XXXX-XX
- supplier_state: sigla UF (ex: MG, SP, PE)
- customer_name: string
- customer_cnpj: string formato XX.XXX.XXX/XXXX-XX
- customer_state: sigla UF
- consumer_unit: string (código da UC / instalação)
- client_number: string (número do cliente na concessionária)
- total: number (valor total a pagar)
- subtotal: number
- payment_method: string
- line_items: lista de {{"description": string, "quantity": number, "unit": string, "unit_price": number, "total": number}}
  Se line_items já contém itens com sub-campos null, retorne a lista completa com os sub-campos preenchidos.
  Inclua apenas itens identificados no texto — não invente itens.

Texto da fatura (até 6000 caracteres):
---
{text}
---
"""

# ---------------------------------------------------------------------------
# Cliente Ollama
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 5    # segundos para estabelecer conexão
_INFER_TIMEOUT   = 180  # segundos para inferência (CPU sem GPU pode ser lento)
_PULL_TIMEOUT    = 600  # segundos para download do modelo


class LLMFallback:
    """
    Extrator via Ollama (open source) para campos não resolvidos
    pelo parser determinístico.

    A classe verifica conectividade no __init__ e faz pull lazy do
    modelo na primeira chamada a enrich() — sem bloquear o startup.
    """

    def __init__(self) -> None:
        self._base_url: str = os.getenv("LLM_BASE_URL", "http://ollama:11434").rstrip("/")
        self._model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self._enabled: bool = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
        self._model_ready: bool = False  # True após confirmar que o modelo está disponível

        if self._enabled:
            self._check_connectivity()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def enrich(self, invoice: Invoice, missing_fields: list[str], content=None) -> Invoice:
        """
        Tenta preencher os campos ausentes via Ollama e retorna o Invoice atualizado.
        O processamento é síncrono — bloqueia até receber a resposta da LLM.
        Nunca sobrescreve campos já preenchidos pelo parser determinístico.

        Quando `content` (DocumentContent) é fornecido:
          - usa content.markdown (mais rico — inclui tabelas formatadas) em vez de raw_text
          - injeta seções específicas do section_map para campos prioritários (section-scoped context)
        """
        if not self._enabled or not missing_fields:
            return invoice

        has_content = content is not None and DocumentContent is not None
        source_text = (
            content.markdown if has_content and content.markdown
            else (invoice.raw_text or "")
        )

        if not source_text:
            return invoice

        logger.info(
            f"LLM fallback acionado para '{invoice.source_file}' — "
            f"campos ausentes: {[f.split(' — ')[0] for f in missing_fields]}"
        )

        if not self._ensure_model():
            logger.warning("Modelo indisponível — LLM fallback ignorado.")
            return invoice

        context_section = _build_section_context(missing_fields, content) if has_content else ""

        prompt = _USER_PROMPT_TEMPLATE.format(
            fields="\n".join(f"- {f}" for f in missing_fields),
            context_section=context_section,
            text=source_text[:6000],
        )

        try:
            raw = self._infer(prompt)
            logger.debug(f"Resposta LLM ({len(raw)} chars): {raw[:200]}")
            extracted = _parse_json(raw)
            invoice = _merge(invoice, extracted)
            invoice.parsing_method = ParsingMethod.HYBRID

        except requests.exceptions.Timeout:
            msg = f"Timeout na inferência ({_INFER_TIMEOUT}s) — arquivo ignorado pelo LLM."
            logger.error(msg)
            invoice.errors.append(msg)
        except Exception as exc:
            logger.error(f"LLM fallback falhou: {exc}")
            invoice.errors.append(f"llm_fallback_error: {exc}")

        return invoice

    def enrich_nulls(self, invoice: Invoice, content=None) -> Invoice:
        """
        Modo enriquecimento: sempre acionado após extração determinística.

        Envia o JSON parcial atual (campos já preenchidos visíveis, nulls explícitos)
        e o texto bruto ao LLM, que preenche apenas os campos null identificados.
        Nunca sobrescreve campos já preenchidos pelo parser determinístico.
        Retorna o invoice inalterado se nenhum campo enriquecível estiver null.
        """
        if not self._enabled:
            return invoice

        null_fields = _get_enrichable_null_fields(invoice)
        if not null_fields:
            logger.debug(
                f"Enriquecimento LLM ignorado para '{invoice.source_file}' — "
                "nenhum campo null enriquecível"
            )
            return invoice

        has_content = content is not None and DocumentContent is not None
        source_text = (
            content.markdown if has_content and content.markdown
            else (invoice.raw_text or "")
        )
        if not source_text:
            return invoice

        logger.info(
            f"LLM enriquecimento para '{invoice.source_file}' — "
            f"{len(null_fields)} campo(s) null: {null_fields}"
        )

        if not self._ensure_model():
            logger.warning("Modelo indisponível — enriquecimento LLM ignorado.")
            return invoice

        partial = _build_partial_json(invoice)
        prompt = _ENRICH_USER_TEMPLATE.format(
            partial_json=json.dumps(partial, ensure_ascii=False, indent=2, default=str),
            text=source_text[:6000],
        )

        try:
            raw = self._infer(prompt, system=_ENRICH_SYSTEM_PROMPT, num_ctx=8192, num_predict=2048)
            logger.debug(f"Resposta LLM enriquecimento ({len(raw)} chars): {raw[:200]}")
            extracted = _parse_json(raw)
            invoice = _merge(invoice, extracted)
            invoice.parsing_method = ParsingMethod.HYBRID
        except requests.exceptions.Timeout:
            msg = f"Timeout na inferência LLM ({_INFER_TIMEOUT}s) — enriquecimento ignorado."
            logger.error(msg)
            invoice.errors.append(msg)
        except Exception as exc:
            logger.error(f"LLM enriquecimento falhou: {exc}")
            invoice.errors.append(f"llm_enrich_error: {exc}")

        return invoice

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _check_connectivity(self) -> None:
        """Testa a conexão com o Ollama. Desabilita o fallback se inacessível."""
        try:
            resp = requests.get(
                f"{self._base_url}/api/tags",
                timeout=_CONNECT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(f"Ollama conectado: {self._base_url} | modelo alvo: {self._model}")
        except Exception as exc:
            logger.warning(
                f"Ollama não acessível em '{self._base_url}': {exc}. "
                "LLM fallback desabilitado — processamento continuará apenas com extração determinística."
            )
            self._enabled = False

    def _ensure_model(self) -> bool:
        """
        Verifica se o modelo está disponível no Ollama.
        Faz pull automático se não estiver — operação bloqueante, com log de progresso.
        """
        if self._model_ready:
            return True

        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=_CONNECT_TIMEOUT)
            resp.raise_for_status()
            installed = [m["name"] for m in resp.json().get("models", [])]
            model_base = self._model.split(":")[0]

            if not any(model_base in name for name in installed):
                logger.info(
                    f"Modelo '{self._model}' não encontrado localmente. "
                    f"Iniciando download (pode demorar alguns minutos em CPU)..."
                )
                self._pull_model()
            else:
                logger.debug(f"Modelo '{self._model}' já disponível no Ollama.")

            self._model_ready = True
            return True

        except Exception as exc:
            logger.error(f"Falha ao verificar/baixar modelo '{self._model}': {exc}")
            return False

    def _pull_model(self) -> None:
        """
        Faz pull do modelo via API de streaming do Ollama.
        Loga o progresso sem bloquear indefinidamente (timeout de 10 min).
        """
        with requests.post(
            f"{self._base_url}/api/pull",
            json={"name": self._model},
            stream=True,
            timeout=_PULL_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    completed = data.get("completed")
                    total = data.get("total")
                    if completed and total:
                        pct = completed / total * 100
                        logger.debug(f"  Download: {status} {pct:.1f}%")
                    elif status:
                        logger.debug(f"  {status}")
                except json.JSONDecodeError:
                    pass

        logger.info(f"Modelo '{self._model}' baixado com sucesso.")

    def _infer(
        self,
        user_prompt: str,
        system: str = _SYSTEM_PROMPT,
        num_ctx: int = 4096,
        num_predict: int = 1024,
    ) -> str:
        """
        Chama a API de chat do Ollama de forma síncrona e retorna o texto da resposta.
        Usa format='json' para forçar saída JSON válida.
        temperature=0 garante saída determinística e estruturada.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "format": "json",   # Força o modelo a retornar JSON válido
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=_INFER_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Funções auxiliares (puras — sem estado)
# ---------------------------------------------------------------------------

_SUPPLIER_FIELDS = {"supplier_name", "supplier_cnpj", "supplier_cpf", "supplier_state"}
_CUSTOMER_FIELDS = {"customer_name", "customer_cnpj", "customer_cpf", "customer_state"}
_ENERGY_FIELDS   = {"consumer_unit", "client_number", "line_items"}

_SUPPLIER_SECTION_KEYS = {"emitente", "fornecedor", "remetente", "vendedor", "prestador"}
_CUSTOMER_SECTION_KEYS = {"destinatario", "destinatário", "cliente", "tomador", "comprador"}
_ENERGY_SECTION_KEYS   = {
    "itens", "fatura", "cobranca", "cobrança", "consumo", "energia",
    "discriminacao", "discriminação", "composicao", "composição",
    "unidade consumidora", "instalacao", "instalação",
}


def _build_section_context(missing_fields: list[str], content) -> str:
    """
    Constrói um bloco de contexto focado nas seções relevantes para os campos ausentes.
    Usa content.section_map (dict[str, str]) para mapear nomes de seção ao seu texto.
    Retorna string pronta para inserção no prompt (vazia se nada relevante encontrado).
    """
    if content is None or not getattr(content, "section_map", None):
        return ""

    missing_set = {f.split(" — ")[0].strip() for f in missing_fields}
    section_map: dict = content.section_map

    sections_needed: list[str] = []

    need_supplier = bool(missing_set & _SUPPLIER_FIELDS)
    need_customer = bool(missing_set & _CUSTOMER_FIELDS)
    need_energy   = bool(missing_set & _ENERGY_FIELDS)

    for section_name, section_text in section_map.items():
        key = section_name.lower().strip()
        key = unicodedata.normalize("NFD", key)
        key = "".join(c for c in key if unicodedata.category(c) != "Mn")

        if need_supplier and any(k in key for k in _SUPPLIER_SECTION_KEYS):
            sections_needed.append(f"[{section_name}]\n{section_text[:800]}")
        elif need_customer and any(k in key for k in _CUSTOMER_SECTION_KEYS):
            sections_needed.append(f"[{section_name}]\n{section_text[:800]}")
        elif need_energy and any(k in key for k in _ENERGY_SECTION_KEYS):
            sections_needed.append(f"[{section_name}]\n{section_text[:1200]}")

    if not sections_needed:
        return ""

    combined = "\n\n".join(sections_needed)
    return _SECTION_CONTEXT_TEMPLATE.format(section_text=combined)

def _get_enrichable_null_fields(invoice: Invoice) -> list[str]:
    """
    Retorna os nomes dos campos enriquecíveis (passíveis de preenchimento pelo LLM)
    que estão atualmente null no invoice. Campos de metadados são excluídos.
    """
    null: list[str] = []
    if not invoice.invoice_number:
        null.append("invoice_number")
    if not invoice.issue_date:
        null.append("issue_date")
    if not invoice.due_date:
        null.append("due_date")
    if not invoice.service_period_start:
        null.append("service_period_start")
    if not invoice.service_period_end:
        null.append("service_period_end")
    if not invoice.supplier.name:
        null.append("supplier_name")
    if not invoice.supplier.cnpj:
        null.append("supplier_cnpj")
    if not invoice.supplier.address.state:
        null.append("supplier_state")
    if not invoice.customer.name:
        null.append("customer_name")
    if not invoice.customer.cnpj:
        null.append("customer_cnpj")
    if not invoice.customer.address.state:
        null.append("customer_state")
    if not invoice.consumer_unit:
        null.append("consumer_unit")
    if not invoice.client_number:
        null.append("client_number")
    if invoice.total is None:
        null.append("total")
    if invoice.subtotal is None:
        null.append("subtotal")
    # line_items: inclui se lista vazia OU se algum item ainda tem quantity e total null
    # (itens com dados parciais precisam de enriquecimento de sub-campos)
    if not invoice.line_items or any(
        item.quantity is None and item.total is None
        for item in invoice.line_items
    ):
        null.append("line_items")
    return null


def _build_partial_json(invoice: Invoice) -> dict:
    """
    Constrói um dict com os campos enriquecíveis para enviar ao LLM no modo enriquecimento.
    Campos com valor são incluídos como contexto; campos null aparecem explicitamente como null.
    """
    return {
        "invoice_number": invoice.invoice_number,
        "issue_date": str(invoice.issue_date) if invoice.issue_date else None,
        "due_date": str(invoice.due_date) if invoice.due_date else None,
        "service_period_start": str(invoice.service_period_start) if invoice.service_period_start else None,
        "service_period_end": str(invoice.service_period_end) if invoice.service_period_end else None,
        "supplier_name": invoice.supplier.name,
        "supplier_cnpj": invoice.supplier.cnpj,
        "supplier_state": invoice.supplier.address.state,
        "customer_name": invoice.customer.name,
        "customer_cnpj": invoice.customer.cnpj,
        "customer_state": invoice.customer.address.state,
        "consumer_unit": invoice.consumer_unit,
        "client_number": invoice.client_number,
        "total": invoice.total,
        "subtotal": invoice.subtotal,
        "line_items": [
            {
                "description": item.description,
                "unit": item.unit,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "total": item.total,
            }
            for item in invoice.line_items
        ] or None,
    }


def _parse_json(text: str) -> dict:
    """Extrai JSON da resposta, tolerante a markdown code fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    return float(v) if isinstance(v, (int, float)) else parse_currency(str(v))


def _merge(invoice: Invoice, data: dict) -> Invoice:
    """
    Preenche o máximo possível de campos do invoice com dados extraídos pelo LLM.
    Nunca sobrescreve campos já preenchidos pelo parser determinístico — apenas
    complementa o que estiver faltando, maximizando a quantidade de chave-valor
    presentes no resultado final.
    """
    if not invoice.invoice_number and data.get("invoice_number"):
        invoice.invoice_number = str(data["invoice_number"])

    if not invoice.issue_date and data.get("issue_date"):
        try:
            invoice.issue_date = date.fromisoformat(str(data["issue_date"]))
        except ValueError:
            pass

    if not invoice.due_date and data.get("due_date"):
        try:
            invoice.due_date = date.fromisoformat(str(data["due_date"]))
        except ValueError:
            pass

    if not invoice.total and data.get("total") is not None:
        invoice.total = _as_float(data["total"])

    if not invoice.subtotal and data.get("subtotal") is not None:
        invoice.subtotal = _as_float(data["subtotal"])

    if not invoice.payment_method and data.get("payment_method"):
        invoice.payment_method = str(data["payment_method"])

    if not invoice.payment_terms and data.get("payment_terms"):
        invoice.payment_terms = str(data["payment_terms"])

    # Fornecedor (invoice.supplier é sempre um Party — nunca None)
    if not invoice.supplier.name and data.get("supplier_name"):
        invoice.supplier.name = str(data["supplier_name"])
    if not invoice.supplier.cnpj and data.get("supplier_cnpj"):
        invoice.supplier.cnpj = str(data["supplier_cnpj"])
    if not invoice.supplier.cpf and data.get("supplier_cpf"):
        invoice.supplier.cpf = str(data["supplier_cpf"])
    if not invoice.supplier.address.state and data.get("supplier_state"):
        invoice.supplier.address.state = str(data["supplier_state"])

    # Cliente (invoice.customer é sempre um Party — nunca None)
    if not invoice.customer.name and data.get("customer_name"):
        invoice.customer.name = str(data["customer_name"])
    if not invoice.customer.cnpj and data.get("customer_cnpj"):
        invoice.customer.cnpj = str(data["customer_cnpj"])
    if not invoice.customer.cpf and data.get("customer_cpf"):
        invoice.customer.cpf = str(data["customer_cpf"])
    if not invoice.customer.address.state and data.get("customer_state"):
        invoice.customer.address.state = str(data["customer_state"])

    # Itens de fatura: três estratégias em ordem de preferência
    llm_items_raw = data.get("line_items") if isinstance(data.get("line_items"), list) else []

    if invoice.line_items and llm_items_raw:
        all_empty = all(
            item.quantity is None and item.unit_price is None and item.total is None
            for item in invoice.line_items
        )
        if all_empty:
            # Estratégia A: todos os itens existentes são OCR vazio — substituição completa
            invoice.line_items = []
        else:
            # Estratégia B: itens existentes têm dados parciais — merge por descrição.
            # Preenche sub-campos null (unit, quantity, unit_price, total) nos itens
            # existentes cujas descrições coincidam com os itens retornados pelo LLM.
            llm_by_desc = {
                str(r.get("description", "")).strip().lower(): r
                for r in llm_items_raw
                if isinstance(r, dict) and r.get("description")
            }
            for item in invoice.line_items:
                match = llm_by_desc.get(item.description.strip().lower())
                if not match:
                    continue
                if item.unit is None and match.get("unit"):
                    item.unit = str(match["unit"])[:20]
                if item.quantity is None and match.get("quantity") is not None:
                    item.quantity = _as_float(match["quantity"])
                if item.unit_price is None and match.get("unit_price") is not None:
                    item.unit_price = _as_float(match["unit_price"])
                if item.total is None and match.get("total") is not None:
                    item.total = _as_float(match["total"])

    # Estratégia C: lista vazia (após descarte de lixo ou sem itens) — popula do LLM
    if not invoice.line_items and llm_items_raw:
        for raw_item in llm_items_raw[:10]:
            if not isinstance(raw_item, dict) or not raw_item.get("description"):
                continue
            invoice.line_items.append(
                LineItem(
                    description=str(raw_item["description"])[:300],
                    unit=str(raw_item["unit"])[:20] if raw_item.get("unit") else None,
                    quantity=_as_float(raw_item.get("quantity")),
                    unit_price=_as_float(raw_item.get("unit_price")),
                    total=_as_float(raw_item.get("total")),
                )
            )

    # Identificadores de instalação de energia
    if not invoice.consumer_unit and data.get("consumer_unit"):
        invoice.consumer_unit = str(data["consumer_unit"])
    if not invoice.client_number and data.get("client_number"):
        invoice.client_number = str(data["client_number"])

    return invoice

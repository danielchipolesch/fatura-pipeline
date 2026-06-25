"""
Fallback LLM para campos que o parser determinístico não conseguiu extrair.

Utiliza Ollama como servidor de LLM open source — sem dependência de APIs externas
nem contratos comerciais. Os modelos rodam localmente, na infraestrutura da instituição.

Princípios de uso:
  - Acionado SOMENTE quando confidence_score < CONFIDENCE_THRESHOLD
  - Processamento SÍNCRONO por arquivo (sem paralelismo — preserva CPU)
  - O modelo é baixado automaticamente (pull lazy) na primeira chamada
  - Temperatura 0: saída determinística e estruturada em JSON
  - Nunca deve ser a via primária de extração

Modelos recomendados (configurar LLM_MODEL no .env):
  qwen2.5:3b   — Apache 2.0, multilingual, ~2 GB, padrão
  llama3.2:3b  — Meta Community License, ~2 GB
  mistral:7b   — Apache 2.0, mais capaz mas mais lento em CPU
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Optional

import requests
from loguru import logger

from src.extractors.fields import parse_currency
from src.models.invoice import Invoice, LineItem, ParsingMethod

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
- supplier_city: string (cidade do emitente)
- supplier_state: string (sigla UF do emitente)
- customer_name: string (nome/razão social do destinatário/cliente)
- customer_cnpj: string formato XX.XXX.XXX/XXXX-XX
- customer_cpf: string
- customer_city: string (cidade do destinatário)
- customer_state: string (sigla UF do destinatário)
- subtotal: number
- total: number (valor total a pagar)
- payment_method: string (forma de pagamento, ex: boleto, PIX, débito automático)
- payment_terms: string (condições de pagamento)
- line_items: lista de objetos {{"description": string, "quantity": number, "unit_price": number, "total": number}}
  com os principais itens/serviços da fatura (no máximo 5 itens, os mais relevantes)

Texto da fatura (primeiros 4000 caracteres):
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

    def enrich(self, invoice: Invoice, missing_fields: list[str]) -> Invoice:
        """
        Tenta preencher os campos ausentes via Ollama e retorna o Invoice atualizado.
        O processamento é síncrono — bloqueia até receber a resposta da LLM.
        Nunca sobrescreve campos já preenchidos pelo parser determinístico.
        """
        if not self._enabled or not missing_fields or not invoice.raw_text:
            return invoice

        logger.info(
            f"LLM fallback acionado para '{invoice.source_file}' — "
            f"campos ausentes: {[f.split(' — ')[0] for f in missing_fields]}"
        )

        if not self._ensure_model():
            logger.warning("Modelo indisponível — LLM fallback ignorado.")
            return invoice

        prompt = _USER_PROMPT_TEMPLATE.format(
            fields="\n".join(f"- {f}" for f in missing_fields),
            text=invoice.raw_text[:4000],
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

    def _infer(self, user_prompt: str) -> str:
        """
        Chama a API de chat do Ollama de forma síncrona e retorna o texto da resposta.
        Usa format='json' para forçar saída JSON válida.
        temperature=0 garante saída determinística e estruturada.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "format": "json",   # Força o modelo a retornar JSON válido
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024,
                "num_ctx": 4096,
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
    if not invoice.supplier.address.city and data.get("supplier_city"):
        invoice.supplier.address.city = str(data["supplier_city"])
    if not invoice.supplier.address.state and data.get("supplier_state"):
        invoice.supplier.address.state = str(data["supplier_state"])

    # Cliente (invoice.customer é sempre um Party — nunca None)
    if not invoice.customer.name and data.get("customer_name"):
        invoice.customer.name = str(data["customer_name"])
    if not invoice.customer.cnpj and data.get("customer_cnpj"):
        invoice.customer.cnpj = str(data["customer_cnpj"])
    if not invoice.customer.cpf and data.get("customer_cpf"):
        invoice.customer.cpf = str(data["customer_cpf"])
    if not invoice.customer.address.city and data.get("customer_city"):
        invoice.customer.address.city = str(data["customer_city"])
    if not invoice.customer.address.state and data.get("customer_state"):
        invoice.customer.address.state = str(data["customer_state"])

    # Itens de fatura — só complementa se o parser determinístico não achou nenhum
    if not invoice.line_items and isinstance(data.get("line_items"), list):
        for raw_item in data["line_items"][:10]:
            if not isinstance(raw_item, dict) or not raw_item.get("description"):
                continue
            invoice.line_items.append(
                LineItem(
                    description=str(raw_item["description"])[:300],
                    quantity=_as_float(raw_item.get("quantity")),
                    unit_price=_as_float(raw_item.get("unit_price")),
                    total=_as_float(raw_item.get("total")),
                )
            )

    return invoice

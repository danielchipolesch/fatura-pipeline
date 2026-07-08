# fatura-pipeline

Pipeline de extração estruturada de faturas e notas fiscais em PDF, com foco em documentos fiscais brasileiros (NFe/DANFE, NFSe, boletos e contas de energia elétrica — mercado cativo e livre).

---

## Visão Geral

O pipeline converte PDFs de documentos fiscais em objetos JSON padronizados, prontos para ingestão em sistemas de BI, ERP ou data warehouse. A extração é feita em três camadas progressivas, preferindo sempre a interpretação semântica e degradando graciosamente para regex e LLM quando necessário.

**Tipos de documento suportados**

| Layout | Exemplos |
|--------|----------|
| NFe / DANFE | RGE Sul, CEMIG Distribuição |
| NFSe | Notas de serviço com ISS |
| Boleto | Light, Enel |
| Conta de Energia (Cativo) | CEMIG, Enel SP/CE, Amazonas Energia |
| Conta de Energia (Mercado Livre) | Comerc, distribuidoras com energia ACL |
| Genérico | Qualquer documento com estrutura detectável |

---

## Arquitetura

### Pipeline de três camadas

```
PDF
 │
 ▼
┌─────────────────────────────────────────┐
│  Camada 1 — DoclingExtractor (Semântica) │
│  Docling: seções, pares KV, índice       │
│  espacial, tabelas, OCR automático       │
└──────────────────┬──────────────────────┘
                   │ campos ausentes
                   ▼
┌─────────────────────────────────────────┐
│  Camada 2 — DeterministicParser (Regex)  │
│  Padrões indexados por layout,           │
│  score de confiança por cobertura        │
└──────────────────┬──────────────────────┘
                   │ confiança < threshold
                   ▼
┌─────────────────────────────────────────┐
│  Camada 3 — LLMFallback (Ollama)         │
│  qwen2.5:3b (Apache 2.0), ativado        │
│  apenas quando necessário                │
└─────────────────────────────────────────┘
                   │
                   ▼
            invoice.json
```

**DoclingExtractor** extrai via estrutura nativa do Docling (seções, pares chave-valor, bounding boxes e tabelas). Para impostos, aplica detecção matemática de triplas (base × alíquota/100 ≈ valor), resistindo a layouts onde os valores aparecem antes do label no texto linearizado (DANFE NF3E, por exemplo).

**DeterministicParser** cobre o que o Docling não encontrou com expressões regulares por layout, incluindo identificadores brasileiros (CNPJ, CPF, CEP, chave de acesso), datas em formatos variados e métricas de energia.

**LLMFallback** processa o documento completo via Ollama quando o score de confiança fica abaixo do limiar configurado, ou quando OCR é detectado (documentos digitalizados têm confiança forçada a 0).

### Classificação de layout

```
NFe (DANFE, CFOP) > NFSe (Tomador, ISS) > Boleto > Energia (kWh, tarifa) > Genérico
```

### Tipo de fatura de energia

| Valor | Critério |
|-------|----------|
| `COMERCIALIZADORA_MLE` | Unidade em MWh, CNPJ de comercializadora, termos ACL/livre |
| `DISTRIBUIDORA_MLE` | "Energia Terc. Comercializada", "Energia ACL", TUSD ponta/fora-ponta separados |
| `CATIVO` | Layout energia sem sinais de mercado livre |
| `null` | Não é fatura de energia |

---

## Estrutura do Repositório

```
fatura-pipeline/
├── src/
│   ├── main.py                  # Entrypoint + CLI (batch / watch / arquivo único)
│   ├── pipeline.py              # Orquestrador das três camadas
│   ├── models/
│   │   └── invoice.py           # Modelos Pydantic: Invoice, Tax, LineItem, EnergyMetrics…
│   ├── extractors/
│   │   ├── docling_extractor.py # Camada 1 — extração semântica
│   │   ├── fields.py            # Funções puras de extração de campos (regex)
│   │   ├── layout.py            # Classificação de layout de documento
│   │   ├── tables.py            # Extração de itens de linha a partir de DataFrames
│   │   ├── taxes.py             # Extração de impostos (ICMS, PIS, COFINS, ISS…)
│   │   ├── tipo_fatura.py       # Inferência do tipo operacional de fatura de energia
│   │   └── known_entities.py    # Tabela CNPJ: distribuidoras, comercializadoras, clientes
│   ├── parsers/
│   │   ├── docling_loader.py    # PDF → DocumentContent (OCR automático incluso)
│   │   ├── deterministic.py     # Camada 2 — parsing por regex e layout
│   │   └── llm_fallback.py      # Camada 3 — enriquecimento via Ollama
│   └── utils/
│       └── helpers.py           # SHA-256, serialização JSON, registro de idempotência
├── scripts/                     # Scripts utilitários de debug
├── input/                       # PDFs de entrada (gitignore, montado no Docker)
├── output/                      # JSONs gerados + .processed_registry.json
├── logs/                        # pipeline.log (rotativo)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Início Rápido

### Pré-requisitos

- Docker e Docker Compose

### Configuração

```bash
cp .env.example .env
# Edite .env conforme necessário (ver variáveis abaixo)
```

### Execução com Docker

```bash
# Build e inicialização
docker compose up --build

# Coloque PDFs em input/ e o pipeline processa automaticamente
# Os JSONs gerados ficam em output/
```

Para processar apenas um arquivo:

```bash
docker compose run --rm pipeline python -m src.main minha_fatura.pdf
```

---

## Uso da CLI

```bash
# Modo batch: processa todos os PDFs em INPUT_DIR
python -m src.main

# Arquivo específico
python -m src.main caminho/para/fatura.pdf

# Modo watch: monitora INPUT_DIR a cada POLL_INTERVAL_SECONDS
python -m src.main --watch
```

O pipeline é idempotente: arquivos já processados (SHA-256 registrado em `.processed_registry.json`) são ignorados em execuções subsequentes.

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `INPUT_DIR` | `/app/input` | Diretório de entrada dos PDFs |
| `OUTPUT_DIR` | `/app/output` | Diretório de saída dos JSONs |
| `LOGS_DIR` | `/app/logs` | Diretório de logs |
| `LOG_LEVEL` | `INFO` | Nível de log (DEBUG, INFO, WARNING, ERROR) |
| `WATCH_MODE` | `false` | Ativar modo de monitoramento contínuo |
| `POLL_INTERVAL_SECONDS` | `30` | Intervalo do modo watch |
| `CONFIDENCE_THRESHOLD` | `0.60` | Score mínimo antes de acionar o LLM |
| `LLM_FALLBACK_ENABLED` | `false` | Habilitar fallback via Ollama |
| `LLM_MODEL` | `qwen2.5:3b` | Modelo Ollama a usar |
| `LLM_BASE_URL` | `http://ollama:11434` | URL do serviço Ollama |
| `DOCLING_OCR_MODE` | `auto` | Modo OCR: `auto` / `always` / `never` |
| `DOCLING_OCR_BACKEND` | `auto` | Backend OCR: `auto` / `rapidocr` / `tesseract` |
| `DOCLING_TABLE_STRUCTURE` | `true` | Habilitar análise de estrutura de tabelas |
| `DOCLING_BATCH_CONCURRENCY` | `2` | Paralelismo de conversão (ajustar para CPU disponível) |

---

## Schema de Saída

Cada PDF gera um arquivo `<nome>.json` com o seguinte esquema (todos os campos opcionais são `null` quando ausentes):

```jsonc
{
  // Identificação
  "invoice_number": "141846706",
  "invoice_layout": "NFe",           // NFe | NFSe | Boleto | Utility | Generic
  "access_key": "31240...44 dígitos",
  "series": "1",

  // Datas (ISO 8601)
  "issue_date": "2024-03-01",
  "due_date": "2024-03-15",
  "service_period_start": "2024-02-01",
  "service_period_end": "2024-02-29",

  // Partes
  "supplier": { "name": "RGE SUL", "cnpj": "...", "address": { ... } },
  "customer": { "name": "...",     "cnpj": "...", "address": { ... } },

  // Itens e impostos
  "line_items": [ { "description": "ENERGIA ATIVA", "quantity": 1234.5, "unit": "kWh", "unit_price": 0.85, "total": 1049.33 } ],
  "taxes": [
    { "name": "ICMS",  "base": 223361.36, "rate": 17.0,  "amount": 37971.43 },
    { "name": "PIS",   "base": 187297.43, "rate": 1.04,  "amount": 1947.91  },
    { "name": "COFINS","base": 187297.43, "rate": 4.79,  "amount": 8971.55  }
  ],

  // Métricas de energia (null para documentos não-energia)
  "energy_metrics": {
    "consumer_unit": "7007856",
    "measured_demand_kw": 120.0,
    "consumption_kwh": 1234.5,
    "power_factor": null,
    "reactive_excess_kvarh": null
  },

  // Tipo operacional de energia
  "tipo_fatura_operacional": "CATIVO",  // CATIVO | DISTRIBUIDORA_MLE | COMERCIALIZADORA_MLE | null

  // Totais (BRL)
  "currency": "BRL",
  "subtotal": null,
  "discount": null,
  "total_taxes": null,
  "total": 48890.89,

  // Pagamento
  "payment_method": null,
  "bank_slip_barcode": "...",
  "bank_slip_line": "...",

  // Metadados de processamento
  "source_file": "141846706.pdf",
  "parsing_method": "SEMANTIC",        // SEMANTIC | DETERMINISTIC | HYBRID
  "confidence_score_initial": 0.85,
  "confidence_score": 0.92,
  "notes": [],
  "errors": []
}
```

**Convenção para valores numéricos**

| Valor | Significado |
|-------|-------------|
| `float` | Campo extraído com sucesso |
| `null` | Campo ausente no documento |
| `"LEITURA_FALHOU"` | Campo presente mas ilegível — indica necessidade de refinamento do extrator |

---

## Fornecedores Conhecidos

O pipeline possui lookup direto por CNPJ para os seguintes emissores:

| Empresa | Tipo |
|---------|------|
| CEMIG Geração e Transmissão | Distribuidora |
| CEMIG Distribuição | Distribuidora |
| Enel Distribuição SP | Distribuidora |
| Enel Distribuição CE | Distribuidora |
| Amazonas Energia | Distribuidora |
| Light | Distribuidora |
| RGE Sul | Distribuidora |
| Comerc Energia | Comercializadora |

Novos fornecedores podem ser adicionados em [`src/extractors/known_entities.py`](src/extractors/known_entities.py).

---

## OCR

O modo `auto` (padrão) converte o PDF via caminho rápido primeiro. Se o texto extraído for inferior a 120 caracteres por página, o pipeline ativa OCR automaticamente.

**Backends disponíveis**

| Backend | Características |
|---------|----------------|
| `rapidocr` (padrão) | ONNX, sem PyTorch, leve, suporte a CPU |
| `tesseract` | Fallback clássico, requer instalação do sistema |

---

## Desenvolvimento Local

```bash
pip install -r requirements.txt

# Processar um arquivo localmente (sem Docker)
INPUT_DIR=./input OUTPUT_DIR=./output python -m src.main fatura.pdf
```

Para ativar o fallback LLM localmente, suba o Ollama separadamente:

```bash
ollama serve
ollama pull qwen2.5:3b

LLM_FALLBACK_ENABLED=true LLM_BASE_URL=http://localhost:11434 python -m src.main fatura.pdf
```

---

## Dependências Principais

| Pacote | Papel |
|--------|-------|
| `docling` | Extração semântica de PDFs (seções, KV, tabelas, OCR) |
| `pydantic` | Validação e serialização do schema de saída |
| `rapidocr-onnxruntime` | OCR leve sem PyTorch |
| `loguru` | Logging estruturado com rotação automática |
| `python-dateutil` | Parsing flexível de datas |
| `requests` | Comunicação com Ollama |
| `pandas` | Manipulação de tabelas extraídas pelo Docling |

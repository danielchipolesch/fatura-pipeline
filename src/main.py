"""
Entry point do fatura-pipeline.

Modos de uso:
  python -m src.main                    # batch: processa todos os PDFs em /app/input
  python -m src.main arquivo.pdf        # processa um único arquivo
  python -m src.main --watch            # modo watch: polling contínuo em /app/input

Saída:
  /app/output/<nome_sem_ext>.json       # JSON padronizado por arquivo
  /app/logs/pipeline.log                # log rotativo
  /app/output/.processed_registry.json # controle de idempotência
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carrega .env antes de qualquer import que leia os env vars
load_dotenv()

from src.pipeline import FaturaPipeline
from src.utils.helpers import (
    load_processed_registry,
    save_processed_registry,
    sha256_file,
    write_json,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

_INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input"))
_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
_LOGS_DIR = Path(os.getenv("LOGS_DIR", "/app/logs"))
_POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
_WATCH_MODE = os.getenv("WATCH_MODE", "false").lower() == "true"
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_REGISTRY_FILE = _OUTPUT_DIR / ".processed_registry.json"


def _setup_logging() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=_LOG_LEVEL, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(
        _LOGS_DIR / "pipeline.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


# ---------------------------------------------------------------------------
# Processamento de um arquivo
# ---------------------------------------------------------------------------

def _process_file(pdf_path: Path, pipeline: FaturaPipeline, registry: dict) -> None:
    file_hash = sha256_file(pdf_path)
    if file_hash in registry:
        logger.debug(f"Já processado (hash idêntico): {pdf_path.name}")
        return

    output_path = _OUTPUT_DIR / (pdf_path.stem + ".json")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        invoice = pipeline.process(pdf_path)
        write_json(invoice.to_output_dict(), output_path)
        registry[file_hash] = output_path.name
        logger.success(f"Salvo: {output_path.name}")

    except Exception as exc:
        logger.error(f"Erro ao processar '{pdf_path.name}': {exc}")
        # Salva um JSON de erro para rastreabilidade
        error_payload = {
            "source_file": pdf_path.name,
            "error": str(exc),
            "status": "failed",
        }
        write_json(error_payload, _OUTPUT_DIR / (pdf_path.stem + ".error.json"))


# ---------------------------------------------------------------------------
# Batch e Watch
# ---------------------------------------------------------------------------

def _run_batch(pipeline: FaturaPipeline, registry: dict) -> None:
    """Processa todos os PDFs encontrados em input/ (modo batch — execução única)."""
    pdfs = sorted(_INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        logger.info(f"Nenhum PDF encontrado em {_INPUT_DIR}")
        return

    logger.info(f"{len(pdfs)} PDF(s) encontrado(s) para processar")
    for pdf in pdfs:
        _process_file(pdf, pipeline, registry)
        save_processed_registry(registry, _REGISTRY_FILE)


def _poll_once(pipeline: FaturaPipeline, registry: dict, known_files: dict[str, tuple[float, int]]) -> None:
    """
    Um ciclo de polling do modo watch.

    `known_files` rastreia (mtime, size) dos arquivos já vistos nesta execução —
    permite pular silenciosamente arquivos inalterados sem recalcular hash nem
    gerar log a cada ciclo. Só processa e loga arquivos novos ou modificados,
    para que o aviso "N PDF(s) encontrado(s)" apareça apenas quando há algo
    de fato novo a processar, não em todo polling.
    """
    pdfs = sorted(_INPUT_DIR.glob("*.pdf"))
    new_or_changed: list[Path] = []

    for pdf in pdfs:
        stat = pdf.stat()
        fingerprint = (stat.st_mtime, stat.st_size)
        if known_files.get(pdf.name) == fingerprint:
            continue  # já visto nesta versão exata — não precisa nem hashear
        new_or_changed.append(pdf)
        known_files[pdf.name] = fingerprint

    if not new_or_changed:
        return  # nada novo neste ciclo — silencioso, sem log

    logger.info(f"{len(new_or_changed)} PDF(s) novo(s)/modificado(s) encontrado(s) para processar")
    for pdf in new_or_changed:
        _process_file(pdf, pipeline, registry)
        save_processed_registry(registry, _REGISTRY_FILE)


def _run_watch(pipeline: FaturaPipeline, registry: dict) -> None:
    logger.info(f"Modo watch ativo — polling a cada {_POLL_INTERVAL}s em {_INPUT_DIR}")
    known_files: dict[str, tuple[float, int]] = {}
    while True:
        _poll_once(pipeline, registry, known_files)
        time.sleep(_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger.info("fatura-pipeline iniciado")

    args = sys.argv[1:]
    watch_mode = _WATCH_MODE or "--watch" in args
    single_file: Path | None = None

    for arg in args:
        if not arg.startswith("--"):
            p = Path(arg)
            if p.suffix.lower() == ".pdf":
                single_file = p if p.is_absolute() else _INPUT_DIR / p
                break

    registry = load_processed_registry(_REGISTRY_FILE)
    pipeline = FaturaPipeline()

    if single_file:
        if not single_file.exists():
            logger.error(f"Arquivo não encontrado: {single_file}")
            sys.exit(1)
        _process_file(single_file, pipeline, registry)
        save_processed_registry(registry, _REGISTRY_FILE)

    elif watch_mode:
        _run_watch(pipeline, registry)

    else:
        _run_batch(pipeline, registry)
        save_processed_registry(registry, _REGISTRY_FILE)

    logger.info("fatura-pipeline finalizado")


if __name__ == "__main__":
    main()

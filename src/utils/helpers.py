"""Utilitários gerais do pipeline."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Retorna hash SHA-256 do arquivo para controle de idempotência."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, cls=_JSONEncoder),
        encoding="utf-8",
    )


def load_processed_registry(registry_path: Path) -> dict[str, str]:
    """
    Carrega o registro de arquivos já processados.
    Formato: {sha256: output_filename}
    """
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_processed_registry(registry: dict[str, str], registry_path: Path) -> None:
    write_json(registry, registry_path)

import sys
sys.path.insert(0, '/app')
from src.parsers.docling_loader import DoclingLoader
from pathlib import Path

loader = DoclingLoader()
# Um arquivo de cada fornecedor/layout conhecido
samples = [
    '634039.pdf',          # Enel SP - bare hex-hash
    'SVP 1004635.pdf',     # Enel SP - rich hex-hash
    'SVP 13926.pdf',       # Enel SP - access-key variant
    'SVP 854684.pdf',      # Amazonas Energia
    '645731.pdf',          # CEMIG Distribuição
]
with open('/app/output/_text_full.txt', 'w', encoding='utf-8') as out:
    for fname in samples:
        f = Path('/app/input') / fname
        content = loader.load(f)
        out.write(f"\n{'='*80}\n=== {f.name} | {content.page_count}pag | ocr={content.ocr_used} | {len(content.full_text)} chars ===\n{'='*80}\n")
        out.write(content.full_text)
        out.write("\n")
        print(f"OK: {f.name} ({len(content.full_text)} chars)", flush=True)
print("DONE")

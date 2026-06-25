import sys
sys.path.insert(0, '/app')
from src.parsers.docling_loader import DoclingLoader
from pathlib import Path

loader = DoclingLoader()
files = sorted(Path('/app/input').glob('*.pdf'))
with open('/app/output/_text_dump2.txt', 'w', encoding='utf-8') as out:
    for f in files:
        try:
            content = loader.load(f)
            out.write(f"\n{'='*80}\n=== {f.name} | {content.page_count}pag | ocr={content.ocr_used} ===\n{'='*80}\n")
            out.write(content.full_text[:2800])
            out.write("\n")
        except Exception as e:
            out.write(f"\n=== {f.name}: ERRO {e} ===\n")
        print(f"OK: {f.name}", flush=True)
print("DONE")

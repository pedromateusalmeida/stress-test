"""Extrai todas as imagens PNG dos notebooks executados para consulta e relatório."""

from __future__ import annotations

import base64
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks"
SAIDA = ROOT / "reports" / "figuras_relatorio"
SAIDA.mkdir(parents=True, exist_ok=True)
total = 0
for path in sorted(BASE.glob("[0-9][0-9]_*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    for indice_celula, celula in enumerate(notebook.cells):
        imagens = [saida.data["image/png"] for saida in celula.get("outputs", [])
                   if saida.get("data", {}).get("image/png")]
        for numero, valor in enumerate(imagens, 1):
            destino = SAIDA / f"{path.name[:2]}_cel{indice_celula:02d}_fig{numero}.png"
            destino.write_bytes(base64.b64decode(valor))
            total += 1
            print(destino.relative_to(ROOT))
print(f"Imagens extraídas: {total}")

"""Auditoria pós-execução dos notebooks e CSVs deste projeto.

Uso: python scripts/auditar_variante.py
Esta auditoria verifica rastreabilidade e integridade técnica; não certifica
validade causal dos cenários ou aprovação do modelo para produção.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "models"
arquivos = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
numeros = {int(p.name[:2]) for p in arquivos}
assert set(range(18)).issubset(numeros), f"Notebooks 00–17 incompletos: {sorted(numeros)}"

linhas = []
problemas = []
for path in arquivos:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    fontes = "\n".join("".join(c["source"]) for c in code)
    for i, cell in enumerate(code):
        compile("".join(cell["source"]), f"{path.name}:célula {i}", "exec")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                problemas.append(f"{path.name}: saída de erro na célula {i}")
    if re.search(r"[A-Za-z]:\\Users\\", fontes):
        problemas.append(f"{path.name}: caminho pessoal absoluto em código")
    executadas = sum(c.get("execution_count") is not None for c in code)
    if executadas != len(code):
        problemas.append(f"{path.name}: {executadas}/{len(code)} células de código executadas")
    linhas.append((path.name, len(code), executadas))

manifesto = json.loads((MODELS / "manifesto_modelo.json").read_text(encoding="utf-8"))
assert manifesto["baseline_holdout"]["auc_roc"] > 0
assert abs(manifesto["baseline_holdout"]["brier"] - 0.1440383796505263) < 1e-7
assert abs(manifesto["baseline_holdout"]["pr_auc_trap"] - 0.39599053922779864) < 1e-7
assert abs(manifesto["baseline_holdout"]["ks_modelo"] - 0.5877142857142856) < 1e-7

resultados = []
for path in sorted(TABLES.glob("*.csv")):
    if path.stat().st_size == 0:
        problemas.append(f"{path.name}: CSV vazio")
        continue
    frame = pd.read_csv(path, nrows=5)
    cols = {c.lower() for c in frame.columns}
    resultados.append((path.name, sum(1 for _ in open(path, encoding="utf-8-sig")) - 1,
                       ", ".join(c for c in frame.columns if
                                 any(k in c.lower() for k in ("brier", "auc", "ks_modelo")))))
    if path.name.startswith("resultados_") and ("brier" in cols or "auc" in cols or "auc_roc" in cols):
        has_pr = any("prauc" in c or "pr_auc" in c for c in cols)
        has_ks = any("ks_modelo" in c or "ks_credito" in c for c in cols)
        if not has_pr or not has_ks:
            problemas.append(f"{path.name}: colunas centrais PR-AUC/KS ausentes ou não padronizadas")

print("NOTEBOOKS")
for row in linhas:
    print(" | ".join(map(str, row)))
print("\nCSVS")
for row in resultados:
    print(" | ".join(map(str, row)))
print("\nPROBLEMAS")
for problema in problemas:
    print("-", problema)
if problemas:
    raise SystemExit(f"Auditoria encontrou {len(problemas)} problema(s)")
print("Auditoria técnica concluída sem problemas detectados.")

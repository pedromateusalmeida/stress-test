"""Execute os notebooks numerados e salve as saídas nos próprios .ipynb.

Da raiz do projeto:
    python scripts/executar_notebooks.py             # 01 a 17; modelo congelado
    python scripts/executar_notebooks.py 04 05       # intervalo inclusivo
    python scripts/executar_notebooks.py 00 00 --retrain-model

O notebook 00 reconstrói o Booster e requer autorização explícita nesta CLI,
pois altera models/model_class_retreinado.json e seu manifesto. Os demais
notebooks usam o artefato congelado sem retreiná-lo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inicio", type=int, nargs="?", default=1)
    parser.add_argument("fim", type=int, nargs="?", default=17)
    parser.add_argument("--retrain-model", action="store_true",
                        help="permite executar 00 e substituir o modelo e manifesto congelados")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parents[1]
    pasta = raiz / "notebooks"
    if args.inicio < 0 or args.fim > 17 or args.inicio > args.fim:
        parser.error("Use um intervalo inclusivo entre 00 e 17.")
    if args.inicio == 0 and not args.retrain_model:
        parser.error("00 só executa com --retrain-model; ele substitui o Booster.")
    if args.retrain_model and args.inicio != 0:
        parser.error("--retrain-model deve ser usado com intervalo iniciado em 00.")
    if args.retrain_model:
        os.environ["CREDIT_STRESS_RETRAIN_MODEL"] = "1"
    arquivos = sorted(
        p for p in pasta.glob("[0-9][0-9]_*.ipynb")
        if args.inicio <= int(p.name[:2]) <= args.fim
    )
    if len(arquivos) != args.fim - args.inicio + 1:
        parser.error("Há notebooks ausentes no intervalo solicitado.")
    for path in arquivos:
        print(f"INICIO {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        try:
            client = NotebookClient(
                nb, timeout=7200, kernel_name="python3",
                resources={"metadata": {"path": str(raiz)}},
                allow_errors=False,
            )
            client.execute()
        except Exception:
            print(f"FALHA {path.name}; notebook original preservado", flush=True)
            raise
        nbformat.write(nb, path)
        print(f"FIM {path.name}", flush=True)


if __name__ == "__main__":
    main()

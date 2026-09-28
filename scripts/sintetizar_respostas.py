"""Resume superfícies observadas sem impor limite ou declarar aprovação.

Execute após os notebooks 04 e 05: python scripts/sintetizar_respostas.py
As quatro colunas de extremos por métrica podem vir de cenários diferentes.
"""

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1] / "results" / "tables"
UNI = pd.read_csv(BASE / "resultados_univariado_extensivo.csv")
MULTI = pd.read_csv(BASE / "resultados_multivariado_populacional.csv")
M = {"brier": "max", "auc": "min", "prauc_trap": "min", "ks_modelo": "min"}


def resumo(frame: pd.DataFrame, grupo: str, psi: str, extra: str) -> pd.DataFrame:
    linhas = []
    for nome, dados in frame.groupby(grupo, sort=False):
        row = {grupo: nome, "n_cenarios": len(dados), "psi_max_grade": dados[psi].max()}
        for metrica, sentido in M.items():
            coluna = f"delta_{metrica}"
            idx = dados[coluna].idxmax() if sentido == "max" else dados[coluna].idxmin()
            ponto = frame.loc[idx]
            row[f"pior_delta_{metrica}"] = ponto[coluna]
            row[f"psi_no_pior_{metrica}"] = ponto[psi]
            row[f"mecanismo_no_pior_{metrica}"] = ponto[extra]
            row[f"ess_frac_no_pior_{metrica}"] = ponto.get("ess_frac", ponto.get("ess_ratio"))
        linhas.append(row)
    return pd.DataFrame(linhas)


univariado = resumo(UNI, "variavel", "psi_variavel", "trajetoria")
multivariado = resumo(MULTI, "familia", "psi_conjunto_16_clusters", "variaveis")
univariado.to_csv(BASE / "sintese_resposta_univariada.csv", index=False, encoding="utf-8-sig")
multivariado.to_csv(BASE / "sintese_resposta_multivariada.csv", index=False, encoding="utf-8-sig")
print("Resumo univariado: cada extremo é o pior observado de sua própria métrica.")
print(univariado[["variavel", "n_cenarios", "psi_max_grade",
                  "pior_delta_brier", "pior_delta_auc",
                  "pior_delta_prauc_trap", "pior_delta_ks_modelo"]].to_string(index=False))
print("\nResumo multivariado: famílias são geradores, não probabilidades futuras.")
print(multivariado[["familia", "n_cenarios", "psi_max_grade",
                   "pior_delta_brier", "pior_delta_auc",
                   "pior_delta_prauc_trap", "pior_delta_ks_modelo"]].to_string(index=False))

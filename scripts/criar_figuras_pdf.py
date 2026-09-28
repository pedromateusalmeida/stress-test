"""Figuras de leitura em página A4 a partir dos CSVs detalhados executados."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "tables"
OUT = ROOT / "reports" / "figuras_relatorio"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 160, "savefig.dpi": 220})


def salvar(fig, nome):
    fig.savefig(OUT / nome, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / nome)


u = pd.read_csv(BASE / "resultados_univariado_extensivo.csv")
selecionadas = {
    "RevolvingUtilizationOfUnsecuredLines": ("Utilização", "#146C94"),
    "MonthlyIncome": ("Renda", "#C85235"),
}
metricas = [("brier", "Δ Brier"), ("auc", "Δ AUC ROC"),
            ("prauc_trap", "Δ PR-AUC trap."), ("ks_modelo", "Δ KS do modelo")]
fig, axs = plt.subplots(2, 2, figsize=(6.75, 5.2), constrained_layout=True)
for ax, (metrica, label) in zip(axs.flat, metricas):
    for variavel, (curto, cor) in selecionadas.items():
        sub = u[u.variavel == variavel]
        ax.scatter(sub.psi_variavel, sub[f"delta_{metrica}"], s=7,
                   alpha=.48, c=cor, label=curto, rasterized=True)
    ax.axhline(0, color="#222", lw=.8, ls=":")
    ax.set(xlabel="PSI da entrada", ylabel=label)
    ax.set_xscale("symlog", linthresh=.001)
    ax.set_xlim(left=0)
    ax.grid(alpha=.15)
axs[0, 0].legend(loc="best")
fig.suptitle("Resposta univariada: dois exemplos da grade completa", fontsize=11)
salvar(fig, "relatorio_respostas_univariadas.png")


m = pd.read_csv(BASE / "resultados_multivariado_populacional.csv")
m["faixa"] = pd.qcut(m.psi_conjunto_16_clusters, q=10, labels=False, duplicates="drop")
fig, axs = plt.subplots(2, 2, figsize=(6.75, 5.2), constrained_layout=True)
for ax, (metrica, label) in zip(axs.flat, metricas):
    linhas = []
    exp_col = f"delta_{metrica}_exploracao"
    conf_col = f"delta_{metrica}_confirmacao"
    for _, bloco in m.groupby("faixa"):
        ponto = bloco.loc[bloco[exp_col].idxmax() if metrica == "brier"
                          else bloco[exp_col].idxmin()]
        linhas.append((bloco.psi_conjunto_16_clusters.median(),
                       ponto[exp_col], ponto[conf_col]))
    matriz = np.array(linhas)
    ax.plot(matriz[:, 0], matriz[:, 1], "o-", color="#146C94", lw=1.5,
            ms=3.7, label="Exploração")
    ax.plot(matriz[:, 0], matriz[:, 2], "s--", color="#C85235", lw=1.3,
            ms=3.4, label="Mesmo cenário: confirmação")
    ax.axhline(0, color="#222", lw=.8, ls=":")
    ax.set(xlabel="PSI conjunto mediano da faixa", ylabel=label)
    ax.grid(alpha=.15)
axs[0, 0].legend(loc="best", frameon=True)
fig.suptitle("Extremos observados por faixa de drift conjunto", fontsize=11)
salvar(fig, "relatorio_envelopes_multivariados.png")


v = pd.read_csv(BASE / "resultados_percentuais_valores_univariado.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.75, 3.2), constrained_layout=True)
for variavel, (curto, cor) in selecionadas.items():
    sub = v[v.variavel == variavel].sort_values("mudanca_valor_solicitada_pct")
    x = sub.mudanca_valor_solicitada_pct
    axs[0].plot(x, sub.psi_variavel, "o-", ms=3.3, lw=1.6, color=cor, label=curto)
    axs[1].plot(x, 100 * sub.troca_decisao, "o-", ms=3.3, lw=1.6, color=cor, label=curto)
for ax in axs:
    ax.axvline(0, color="#222", lw=.8, ls=":")
    ax.set(xlabel="Mudança solicitada no valor (%)")
    ax.grid(alpha=.15)
axs[0].set(ylabel="PSI da entrada", title="Drift realizado")
axs[1].set(ylabel="Decisões que cruzaram 0,50 (%)", title="Resposta da decisão")
axs[0].legend(loc="upper right")
fig.suptitle("Choques de valor: −50% a +50%, passo de 5%", fontsize=11)
salvar(fig, "relatorio_percentuais_valores.png")


pares = pd.read_csv(BASE / "resultados_percentuais_mix_pares.csv")
fig, axs = plt.subplots(2, 2, figsize=(6.75, 5.2), constrained_layout=True)
ultimo = None
for ax, (metrica, label) in zip(axs.flat, metricas):
    coluna = "delta_auc_roc" if metrica == "auc" else f"delta_{metrica}"
    ultimo = ax.scatter(pares.psi_conjunto_2x2, pares[coluna],
                        c=pares.intensidade_pct, cmap="viridis", vmin=5, vmax=50,
                        s=8, alpha=.58, rasterized=True)
    ax.axhline(0, color="#222", lw=.8, ls=":")
    ax.set(xlabel="PSI conjunto do par", ylabel=label)
    ax.grid(alpha=.15)
fig.colorbar(ultimo, ax=axs.ravel().tolist(), shrink=.72,
             label="Magnitude pedida em cada variável (%)")
fig.suptitle("Pares de entradas: drift conjunto e desempenho", fontsize=11)
salvar(fig, "relatorio_respostas_pares.png")

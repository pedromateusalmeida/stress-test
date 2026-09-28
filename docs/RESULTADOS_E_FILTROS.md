# Resultados, filtros e leitura dos gráficos

Este projeto guarda os **pontos completos** dos experimentos em `results/tables/`. Os gráficos do relatório selecionam poucos exemplos para caber na página. A análise de todas as dez entradas e de todos os 45 pares está nos CSVs do notebook `09` e no atlas visual do notebook `17`.

## 1. Onde está cada tipo de resultado?

| Arquivo em `results/tables/` | Conteúdo |
|---|---|
| `resultados_percentuais_mix_univariado.csv` | 200 cenários: dez entradas × vinte percentuais assinados; métricas ponderadas, PSI e ESS. |
| `resultados_percentuais_valores_univariado.csv` | 200 cenários: dez entradas × vinte percentuais assinados; PSI, mudança efetiva, score e inversão de decisão. |
| `resultados_percentuais_mix_pares.csv` | 1.800 cenários: 45 pares × quatro sinais × dez magnitudes; métricas ponderadas e PSI marginal/conjunto. |
| `resultados_percentuais_valores_pares.csv` | 1.800 cenários de valor de pares; PSI, score e decisões sem rótulo contrafactual. |
| `resultados_transicoes_atraso.csv` | 60 transições discretas adicionais para contagens de atraso. |
| `17_indice_variaveis.csv` / `17_indice_pares.csv` | Índices do atlas: código F01–F10, métricas resumidas e caminho do PNG para as dez entradas e os 45 pares. |
| `resultados_univariado_extensivo.csv` | 2.688 cenários de reponderação por variável, trajetória e intensidade. |
| `resultados_multivariado_populacional.csv` | 3.072 cenários de mix em quatro famílias; métricas na amostra toda, exploração e confirmação. |
| `sintese_resposta_univariada.csv` / `sintese_resposta_multivariada.csv` | Extremos observados por variável/família e por métrica; cada extremo pode vir de cenário diferente. |
| `painel_drift_descritivo.csv` / `painel_resposta_psi_sem_limite.csv` | Drift observado sem rótulo e curvas históricas usadas no painel descritivo. |
| Demais `resultados_*.csv`, `12_*.csv`, `13_*.csv`, `16_*.csv` | Ensaios especializados de qualidade, cobertura, ruptura, operação, calibração e segmentos. |

Os índices `17_*` têm uma coluna `figura` com caminho relativo à raiz do projeto. Exemplo: filtre `variavel=MonthlyIncome` e abra o PNG indicado; para pares, filtre pelas colunas `variavel_a` e `variavel_b`. O [mapa das metodologias](METODOLOGIA.md) liga cada arquivo ao notebook e informa se existe `Y` válido.

## 2. Conferir cobertura integral

Na raiz do repositório, depois da execução do `09`, este exemplo confirma as contagens:

```python
from pathlib import Path
import pandas as pd

base = Path("results/tables")
mix_1 = pd.read_csv(base / "resultados_percentuais_mix_univariado.csv")
valor_1 = pd.read_csv(base / "resultados_percentuais_valores_univariado.csv")
mix_2 = pd.read_csv(base / "resultados_percentuais_mix_pares.csv")
valor_2 = pd.read_csv(base / "resultados_percentuais_valores_pares.csv")

print(len(mix_1), mix_1["variavel"].nunique())       # 200, 10
print(len(valor_1), valor_1["variavel"].nunique())   # 200, 10
print(len(mix_2), len(mix_2[["variavel_a", "variavel_b"]].drop_duplicates()))     # 1800, 45
print(len(valor_2), len(valor_2[["variavel_a", "variavel_b"]].drop_duplicates())) # 1800, 45
print(sorted(mix_2["direcao"].unique()))             # ++, +-, -+, --
```

Essas contagens foram verificadas nos CSVs desta versão. Se sua reexecução gerar menos linhas, leia as saídas de erro e regras de viabilidade do gerador antes de comparar com o relatório. Os vinte níveis assinados univariados são `−50, −45, ..., −5, +5, ..., +45, +50` por cento; o zero é a referência. Nos pares há dez magnitudes positivas combinadas com os quatro sinais.

## 3. Ler uma entrada em todos os níveis

Exemplo com renda mensal, sem depender da figura selecionada no relatório:

```python
var = "MonthlyIncome"

colunas_mix = [
    "mudanca_relativa_pct", "psi_variavel", "n_efetivo",
    "prevalencia", "brier", "auc_roc", "prauc_trap", "ks_modelo",
    "delta_brier", "delta_auc_roc", "delta_prauc_trap", "delta_ks_modelo",
]
trilha_mix = (
    mix_1.loc[mix_1["variavel"].eq(var), colunas_mix]
    .sort_values("mudanca_relativa_pct")
)
print(trilha_mix.to_string(index=False))

colunas_valor = [
    "mudanca_valor_solicitada_pct", "mudanca_media_realizada_pct",
    "psi_variavel", "psi_score", "troca_decisao",
    "fora_suporte_treino_frac",
]
trilha_valor = (
    valor_1.loc[valor_1["variavel"].eq(var), colunas_valor]
    .sort_values("mudanca_valor_solicitada_pct")
)
print(trilha_valor.to_string(index=False))
```

Na **trilha de mix**, o mesmo cliente mantém `(X,Y)` e muda seu peso; Brier, AUC ROC, PR-AUC e KS do modelo são métricas ponderadas sob hipótese de estabilidade do risco condicional. Na **trilha de valor**, a renda do mesmo cliente é alterada; `troca_decisao` é a fração que atravessou o corte 0,50. Não existe um novo desfecho observado para calcular AUC/Brier do “novo mundo”.

Para repetir o exemplo em qualquer entrada, troque o nome de `var` por uma das dez colunas do [manifesto](../models/manifesto_modelo.json).

## 4. Ler um par e as quatro direções

```python
a = "MonthlyIncome"
b = "DebtRatio"

par = mix_2.loc[
    (mix_2["variavel_a"].eq(a) & mix_2["variavel_b"].eq(b))
    | (mix_2["variavel_a"].eq(b) & mix_2["variavel_b"].eq(a))
].copy()

print(
    par[[
        "variavel_a", "variavel_b", "direcao", "intensidade_pct",
        "psi_conjunto_2x2", "psi_a", "psi_b", "n_efetivo",
        "delta_brier", "delta_auc_roc", "delta_prauc_trap",
        "delta_ks_modelo",
    ]]
    .sort_values(["direcao", "intensidade_pct"])
    .to_string(index=False)
)
```

`direcao` dá os sinais na ordem `variavel_a` e `variavel_b` **armazenada no CSV**. Por exemplo, `+-` aumenta a participação do grupo alto de A e reduz a de B. `psi_conjunto_2x2` usa quatro células do cruzamento dos grupos; não é a soma de `psi_a` e `psi_b`. Em `resultados_percentuais_valores_pares.csv`, a mesma direção se aplica a valores das duas entradas e as saídas são score/drift/decisão, sem métrica supervisionada contrafactual.

## 5. O que significam as colunas frequentes?

| Coluna ou padrão | Interpretação |
|---|---|
| `mudanca_*_solicitada_pct`, `intensidade_pct` | Controle pedido ao gerador. Não é PSI nem necessariamente mudança efetiva. |
| `mudanca_*_realizada_pct`, `n_alterados` | Alteração após arredondamento, zeros e regras de domínio. |
| `psi_variavel`, `psi_a`, `psi_b` | Drift marginal da entrada com bins fixos da referência. |
| `psi_conjunto_2x2`, `psi_conjunto_16_clusters` | Drift de grupos conjuntos, com agrupamento declarado no notebook. A granularidade é diferente do PSI marginal. |
| `psi_score` | Drift da distribuição de scores. Não mede erro. |
| `ks_modelo` versus `ks_variavel`/`ks_drift` | O primeiro exige `Y` e mede separação das classes; os demais medem drift de entrada. |
| `auc` ou `auc_roc` | AUC ROC. Alguns notebooks históricos usam o nome curto `auc`. |
| `prauc_trap` ou `pr_auc_trap` | PR-AUC trapezoidal. `ap` é medida separada. |
| `delta_*` | Métrica do cenário menos a métrica basal correspondente. |
| `n_efetivo`, `ess`, `ess_ratio`, `ess_frac` | Tamanho efetivo ou sua fração após pesos; ver [Dicionário](DICIONARIO.md). |
| `troca_decisao` | Fração de linhas cuja decisão no corte 0,50 mudou. |
| `fora_suporte_treino_frac` | Fração fora do mínimo–máximo empírico do treino; indicador simples de extrapolação, não prova de invalidade. |

Os nomes diferem levemente entre famílias de notebooks porque os geradores foram desenvolvidos em etapas. **Não concatene CSVs por posição de coluna.** Leia a legenda do notebook e padronize nomes, população e mecanismo antes de combinar resultados.

## 6. Como ler uma figura de resposta

1. Confira **quem** foi alterado: entrada, par, coorte, score ou rótulo hipotético.
2. Leia o eixo horizontal: percentual solicitado, PSI realizado ou outra medida. São grandezas diferentes.
3. Leia o eixo vertical: métrica absoluta, `Δ` frente ao basal, score ou fração de decisões invertidas.
4. Leia cor, linha e símbolo: podem representar variável, direção, família ou exploração/confirmação.
5. Em gráficos `PSI → ΔM`, cada ponto é um **cenário**, não um instante no tempo.
6. Para Brier, subir é pior; para AUC ROC, PR-AUC e KS do modelo, descer é pior. Para cenário de valor sem novo `Y`, não interprete score/inversão como erro observado.
7. Confira prevalência e ESS em reponderação. PSI semelhante pode acompanhar respostas distintas, e um extremo com ESS muito baixo requer cautela.

A figura do relatório que mostra **utilização** e **renda** é um recorte ilustrativo. A presença dessas duas linhas no PDF não significa que só essas variáveis foram testadas. Abra o `09`, o `17` ou os CSVs para ver todas.

## 7. Um número de “tolerância” ainda exigiria uma decisão externa

Esta suíte mostra curvas de impacto até `±50%` na grade percentual e outros limites próprios de cada gerador. Ela não declara “o modelo tolera PSI 0,1” ou “falha a partir de 15%”. Para transformar a curva em gatilho de produção, a organização precisa fixar qual deterioração de **cada métrica** importa para seu caso, suporte mínimo, incerteza, tamanho/frequência de janela e ação. Uma coorte temporal rotulada é necessária para confirmar a relação observada fora desta amostra.


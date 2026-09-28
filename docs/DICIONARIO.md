# Dicionário, siglas e fórmulas

Este dicionário acompanha os notebooks e o [relatório](../reports/RELATORIO_DIDATICO.md). Sempre que um número for apresentado, leia também **população, mecanismo, unidade, referência e disponibilidade de rótulo**.

## 1. Objetos do estudo

| Termo | Significado aqui |
|---|---|
| Observação ou linha `i` | Um cliente/registro do arquivo usado naquele experimento. |
| `X` | Conjunto de dez entradas do modelo. `X_j` denota a variável `j`. |
| `Y` ou `y_i` | Alvo binário observado: 1 para evento conforme `SeriousDlqin2yrs`, 0 para não evento. Não está observado em `cs-test.csv`. |
| `p_i` ou score | Saída contínua do Booster para a linha `i`, entre 0 e 1. Não foi demonstrado que seja uma probabilidade de inadimplência calibrada. |
| Referência ou basal | Distribuição e métricas do conjunto antes do choque, com modelo e regras congelados. |
| Cenário | Regra de alteração dos dados, com mecanismo, população, direção e intensidade declarados. |
| Intensidade solicitada | Valor de controle do gerador, por exemplo `+10%`. |
| Alteração realizada | Movimento que realmente ocorreu depois de zeros, limites, arredondamento e renormalização. |
| Holdout | Parte rotulada de `cs-training.csv` separada antes do ajuste das árvores. É fora da amostra, não fora do tempo. |
| Modelo congelado | Booster salvo cujo hash e parâmetros estão no manifesto. Experimentos `01–17` mantêm esse artefato fixo. |
| Corte `c` | Regra de decisão `p_i ≥ c`; neste projeto `c=0,50`. Não é limiar de aprovação de teste de estresse. |

## 2. Quatro métricas centrais de desempenho

As métricas desta seção exigem um `Y` observado e justificável para o cenário. Com reponderação, os rótulos são históricos e a conclusão depende de estabilidade condicional e suporte. Em `cs-test.csv` e nos choques de valor econômico real, **não** há desempenho supervisionado observado.

### Brier score

```text
Brier = (1/n) Σ_i (y_i − p_i)²
Brier ponderado = Σ_i w_i (y_i − p_i)² / Σ_i w_i
```

- `n`: número de linhas; `i`: índice de uma linha.
- `y_i`: evento observado, 0 ou 1; `p_i`: score produzido pelo modelo.
- `w_i ≥ 0`: peso da linha no cenário de composição. Sem choque de mix, os pesos são iguais.
- `Σ`: soma sobre as linhas.

**Leitura:** menor Brier significa menor erro quadrático médio. Para um caso com `y=1` e `p=0,70`, a contribuição é `(1−0,70)²=0,09`. O Brier mistura calibração e discriminação; um valor isolado não comprova calibração. O Booster foi treinado com peso positivo alto, por isso compare também com preditor constante na prevalência e examine curvas de calibração.

### AUC ROC

Em cada limiar `t`, o score gera uma matriz de confusão:

```text
TPR(t) = TP / (TP + FN)          FPR(t) = FP / (FP + TN)
AUC ROC = área sob TPR versus FPR ao variar t
```

- `TP`: eventos classificados como positivos; `FN`: eventos classificados como negativos.
- `FP`: não eventos classificados como positivos; `TN`: não eventos classificados como negativos.
- `TPR`: taxa de verdadeiros positivos, também chamada recall; `FPR`: taxa de falsos positivos.
- `t`: limiar percorrido em toda a faixa de scores, **não apenas 0,50**.

**Leitura:** AUC ROC maior indica melhor ordenação global de eventos acima de não eventos. Ela também pode ser lida como a probabilidade de um evento ter score maior que um não evento sorteados, atribuindo meio crédito a empates. Não informa sozinha calibração.

### PR-AUC trapezoidal e AP

```text
Precisão(t) = TP / (TP + FP)     Recall(t) = TPR(t)
PR-AUC trapezoidal = Σ_k (R_k − R_{k−1}) (P_k + P_{k−1}) / 2
AP = Σ_k (R_k − R_{k−1}) P_k
```

- `R_k`: recall no ponto `k` da curva ordenada por recall crescente.
- `P_k`: precisão no mesmo ponto.
- `k`: índice dos pontos da curva; `Σ_k`: soma dos trechos.
- **PR-AUC trapezoidal** interpola linearmente entre pontos. A **AP** usa os degraus de precisão. São medidas diferentes e aparecem com nomes separados nos CSVs.

**Leitura:** maior é melhor dentro da população definida, mas PR-AUC e precisão dependem da **prevalência**. Um choque de mix pode aumentar PR-AUC bruta enquanto AUC e Brier pioram, simplesmente porque a composição de classes mudou. Compare prevalência e outras métricas. No código, a PR-AUC central é `auc(recall, precision)` após `precision_recall_curve`; `average_precision_score` calcula AP adicional.

### KS do modelo

```text
KS_modelo = max_t [TPR(t) − FPR(t)]
```

- `t`: todos os limiares relevantes do score.
- `TPR` e `FPR`: taxas definidas acima.
- `max_t`: maior separação vertical da curva ROC em relação à diagonal.

**Leitura:** maior KS do modelo indica maior separação, naquele conjunto, entre scores de eventos e não eventos. Ele usa `Y`. Não confunda com o **KS de drift** abaixo, que não usa `Y`.

## 3. Medidas de drift e resposta de score

### PSI — Population Stability Index

Congele `B` bins usando a referência. Para cada bin `b`:

```text
PSI = Σ_b (q_b − r_b) ln[(q_b + ε)/(r_b + ε)]
```

- `r_b`: proporção da referência no bin `b`.
- `q_b`: proporção do cenário no mesmo bin.
- `B`: número de bins; `Σ_b`: soma sobre todos os bins.
- `ln`: logaritmo natural; `ε`: pequena suavização para evitar logaritmo de zero.
- A categoria de ausentes, se existente, precisa de regra estável.

**Leitura:** PSI é **adimensional**; `PSI=0,10` não quer dizer mudança de 10%. Zero significa proporções iguais **nesses bins**, podendo esconder movimentos dentro de cada faixa. Compare PSIs apenas quando referência, bins, ausentes e suavização foram definidos do mesmo modo. Este projeto não adota cortes universais de PSI.

### KS de drift

```text
KS_drift = sup_x |F_cenário(x) − F_referência(x)|
```

- `F`: função de distribuição acumulada da variável.
- `x`: valor possível da variável; `sup_x`: maior distância em todos os valores.
- Não precisa de `Y`.

**Leitura:** compara duas distribuições de **uma entrada**. `KS_drift` e `KS_modelo` respondem perguntas distintas; um pode crescer sem que o outro piore.

### Distância Wasserstein de ordem 1

```text
W1 = ∫ |F_cenário(x) − F_referência(x)| dx
```

- `∫`: integral sobre os valores da variável.
- `dx`: unidade da entrada; por isso W1 de renda e W1 de idade não são diretamente comparáveis.
- Alguns notebooks dividem W1 por escala da referência, declarada no caderno, para comparar magnitudes.

### Drift de score e inversão de decisão

```text
Δscore_médio = média(p_cenário) − média(p_basal)
inversão = média[ 1{(p_cenário ≥ 0,50) ≠ (p_basal ≥ 0,50)} ]
```

- `1{condição}`: vale 1 quando a condição é verdadeira, 0 quando falsa.
- `≠`: a decisão mudou para aquele registro.
- `média`: fração das linhas após somar o indicador e dividir pelo total.

**Leitura:** inversão de 2% significa que 2% das decisões atravessaram 0,50. Não diz se essas novas decisões estão certas sem rótulo pertinente. PSI do score mede mudança na distribuição de scores, não erro do modelo.

## 4. Pesos, suporte e incerteza

| Conceito | Definição e interpretação |
|---|---|
| Peso `w_i` | Massa atribuída à linha histórica `i`. Alterar pesos preserva cada combinação `(X_i,Y_i)`, mas muda a frequência de perfis. |
| Prevalência ponderada | `Σ_i w_i y_i / Σ_i w_i`. Deve acompanhar a leitura de PR-AUC e precisão. |
| ESS — tamanho efetivo | `(Σ_i w_i)² / Σ_i w_i²`. Com pesos iguais, ESS é `n`; com massa concentrada, cai. É medida de suporte informativo aproximado, não número literal de pessoas novas. |
| Suporte/overlap | Perfis da população alvo precisam estar representados nos dados históricos para que reponderação seja interpretável. Pesos extremos podem revelar falta de sobreposição. |
| Bootstrap pareado | Reamostra os **mesmos índices** para o cenário basal e o estressado; a distribuição de `ΔM` estima incerteza da diferença, condicional à amostra e ao mecanismo. |
| Exploração/confirmação | Uma parte escolhe cenários extremos; outra avalia os mesmos cenários. Uma segunda parte aleatória do mesmo holdout não substitui validação temporal. |
| Seleção múltipla | O pior entre milhares de cenários tende a ser extremo também por ruído; confirme fora da busca. |

## 5. Tipos de mudança de distribuição

| Termo | Definição |
|---|---|
| Drift marginal/univariado | Mudança na distribuição de uma entrada considerada isoladamente. |
| Drift conjunto/multivariado | Mudança da distribuição de combinações de entradas, inclusive associação entre elas. Marginais iguais não garantem conjunto igual. |
| Covariate shift | Mudança em `P(X)` sob hipótese adicional `P_alvo(Y|X) = P_referência(Y|X)`. Essa igualdade não é demonstrada pelo PSI. |
| Ruptura condicional | Mudança em `P(Y|X)`. Pode ocorrer com `P(X)` idêntica e PSI de entradas zero. |
| Contrafactual | Valor/desfecho que não se observou para o mesmo cliente sob outra condição. Um novo valor de renda não traz automaticamente um novo `Y`. |
| Erro de medição | Valor de entrada registrado ou entregue ao modelo incorretamente, com cliente e evento reais mantidos. |
| Relação estatística preservada | Gerador tenta manter dependências observadas entre entradas. Isso não prova causa/efeito econômico. |
| Extrapolação | Cenário usa valores ou combinações pouco representados no treino; a predição existe, mas sua confiabilidade não é estabelecida. |

## 6. Porcentagens: pedido, participação e realizado

Exemplo: se o grupo de renda alta representa `q=0,20` da referência e se pede aumento **relativo** de participação de 10%, o alvo é `q'=q×(1+0,10)=0,22`. Isso é ganho de **2 pontos percentuais** na participação, não 10 pontos. A viabilidade do grupo, as linhas disponíveis e a normalização podem fazer a participação realizada diferir do alvo.

Já em um choque de **valor** de +10%, uma renda registrada de 1.000 vira 1.100 antes de regras de domínio. Uma contagem zero continua zero sob multiplicação; uma contagem um multiplicada por 1,05 pode continuar um depois de arredondamento. Assim, `+5%` solicitado pode produzir `0%` realizado em muitas linhas. O PSI é calculado **depois** dessas operações. Sempre leia, nessa ordem: mecanismo, percentual pedido, percentual/fração realizada, PSI, score/decisão e métricas justificáveis.

## 7. Siglas

| Sigla | Nome por extenso | Uso neste projeto |
|---|---|---|
| AP | *Average precision* | Integração por degraus da curva precisão–recall; adicional. |
| AUC ROC | *Area under the receiver operating characteristic curve* | Área sob TPR versus FPR; métrica central. |
| BSS | *Brier skill score* | Comparação do Brier com referência; contexto, não métrica central. |
| CPU / GPU | *Central / graphics processing unit* | Hardware de execução; treino configurado com CUDA. |
| DRO | *Distributionally robust optimization* | Busca de mix adversarial na família de grupos definida. |
| ESS | *Effective sample size* | Suporte informativo aproximado após reponderação. |
| FN / FP | *False negative / false positive* | Totais de erros da decisão binária. |
| FPR / TPR | *False positive rate / true positive rate* | Eixos da ROC; TPR é recall. |
| KS | Kolmogorov–Smirnov | Pode ser **KS do modelo** ou **KS de drift**; confira o sufixo. |
| OOS / OoT | *Out of sample / out of time* | Holdout aqui é OOS, não OoT. |
| PD | *Probability of default* | Exige horizonte, população e calibração; score bruto não foi validado como PD. |
| PR-AUC | *Area under the precision–recall curve* | Aqui é a versão trapezoidal, métrica central. |
| PSI | *Population stability index* | Drift em proporções de bins fixos. |
| SLA | *Service level agreement* | Acordo de serviço; latência local não define SLA de produção. |
| W1 | Wasserstein de ordem 1 | Distância entre distribuições numéricas. |

Para detalhes estatísticos e referências usadas, leia as seções 4, 5 e 9 do [relatório didático](../reports/RELATORIO_DIDATICO.md).


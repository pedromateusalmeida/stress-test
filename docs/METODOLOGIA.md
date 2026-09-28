# Metodologia e cobertura dos testes

Este documento explica **qual pergunta cada teste responde, como o cenário é construído e o que pode ser concluído**. O [relatório](../reports/RELATORIO_DIDATICO.md) contém fórmulas, literatura, números e gráficos comentados. O [guia de execução](GUIA_DE_EXECUCAO.md) contém os comandos.

## 1. A pergunta correta antes de calcular PSI

“Uma variável mudou 10%” ainda não define um teste. Pode significar:

1. **Mix:** a participação de clientes de determinado perfil aumentou 10%, mas cada cliente histórico preserva seus valores e seu desfecho. Reponderamos linhas completas `(X,Y)`.
2. **Valor:** o valor da variável de um mesmo cliente foi multiplicado por 1,10. Seu desfecho nesse novo mundo não foi observado.
3. **Medição:** o valor econômico e o desfecho não mudaram, mas o campo registrado e entregue ao modelo contém erro.
4. **Relação entre variáveis:** uma mudança em renda coocorre com outras mudanças segundo uma regra estatística ou um mix de perfis históricos.
5. **Risco condicional:** a chance de evento muda para clientes com as mesmas entradas; pode haver perda de desempenho com PSI de entrada igual a zero.

Esses mecanismos não podem ser reunidos numa única “tolerância de PSI”. O procedimento do projeto é:

```text
mecanismo e população → intensidade solicitada → dados efetivamente alterados
→ drift efetivo → score/decisão e, quando justificável, métricas com Y
→ suporte/incerteza → interpretação do cenário
```

O PSI é uma medida de distância entre proporções em bins fixos. Ele não é porcentagem de alteração nem porcentagem de queda de AUC. O **KS de drift** também compara distribuições de uma entrada; o **KS do modelo** compara distribuições do score entre as duas classes. Veja [Dicionário](DICIONARIO.md).

## 2. Dados, modelo e avaliação

- **Entradas:** dez colunas, sempre na ordem registrada no [manifesto](../models/manifesto_modelo.json).
- **Alvo:** `SeriousDlqin2yrs`, observado em `cs-training.csv` e ausente em `cs-test.csv`.
- **Divisão:** 15% do treino original para holdout; do restante, 17,65% para validação, ambos com semente 33 e sem estratificação. O holdout de 22.500 linhas é fora da amostra, **não fora do tempo**.
- **Booster avaliado:** artefato congelado em `models/`, retreinado com `scale_pos_weight=13.882494684620836`. O objeto `model_class` da sessão original não foi fornecido serializado.
- **Corte de decisão:** 0,50; serve para taxa de decisão, recall e precisão. AUC, PR-AUC e KS do modelo varrem limiares. O Brier usa o score contínuo.
- **Métricas centrais supervisionadas:** Brier (menor é melhor), AUC ROC, PR-AUC **trapezoidal** e KS do modelo (maiores são melhores). Recall e precisão são contexto e **não param testes**.
- **Política de tolerância:** nenhum notebook define número de PSI, queda de AUC ou perda de recall que aprove/reprove o modelo. Uma decisão dessas requer o uso de negócio, custos, incerteza e validação temporal.

### As dez entradas avaliadas

| Coluna | Leitura prática | Cuidado de cenário |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Utilização rotativa de crédito sem garantia | Distribuição assimétrica e extremos; percentual pode cruzar bins de forma desigual. |
| `age` | Idade | Não permitir valor negativo; choques altos podem sair do suporte do treino. |
| `NumberOfTime30-59DaysPastDueNotWorse` | Contagem de atrasos de 30–59 dias | Inteira, muitos zeros; pequenos percentuais podem arredondar para o mesmo valor. |
| `DebtRatio` | Razão de dívida informada na base | Não presumir limite superior de 1; conferir unidades e caudas. |
| `MonthlyIncome` | Renda mensal | Valores ausentes/zero e assimetria mudam intensidade realizada. |
| `NumberOfOpenCreditLinesAndLoans` | Número de linhas e empréstimos abertos | Contagem não negativa, sujeita a degraus. |
| `NumberOfTimes90DaysLate` | Contagem de atrasos de 90 dias ou mais | Inteira, muitos zeros; transições explícitas complementam percentuais. |
| `NumberRealEstateLoansOrLines` | Número de linhas/empréstimos imobiliários | Contagem não negativa. |
| `NumberOfTime60-89DaysPastDueNotWorse` | Contagem de atrasos de 60–89 dias | Inteira, muitos zeros. |
| `NumberOfDependents` | Número de dependentes | Contagem e ausências devem ser tratadas separadamente. |

Os nomes das colunas não garantem uma interpretação econômica completa. Ao repetir em outra carteira, valide definições, unidades, política de ausentes e restrições de domínio junto à fonte dos dados.

## 3. Quando se pode usar o rótulo histórico?

| Cenário | O `Y` histórico serve para Brier/AUC/PR-AUC/KS? | Condição ou limite |
|---|---|---|
| Reponderação de clientes observados | **Sim, condicionalmente** | Os perfis permanecem inteiros; exige suporte e estabilidade aproximada de `P(Y|X)`. A métrica é ponderada, não desempenho observado no futuro. |
| Falha de registro/ingestão da entrada | **Sim, condicionalmente** | O mesmo cliente e desfecho permanecem, apenas o valor entregue ao modelo foi corrompido. |
| Mudança econômica real do valor de uma entrada | **Não, em geral** | O novo `Y` do mesmo cliente é contrafactual. Relate score, decisão e drift. |
| Simulação de `Y` com `X` fixo | **Somente dentro da hipótese** | Demonstra possibilidade lógica, não estima o que ocorrerá em produção. |
| `cs-test.csv` | **Não** | Há entradas, mas não há rótulos observados. |

Nos resultados, `ΔM=M_cenário−M_basal`. `ΔBrier>0` indica piora do erro; `ΔAUC<0`, `ΔPR-AUC<0` ou `ΔKS_modelo<0` indicam piora nas respectivas métricas. Uma métrica pode melhorar enquanto outra piora, especialmente quando a prevalência muda. Nunca substitua esses quatro números por uma conclusão automática “passou/falhou”.

## 4. Cobertura da grade percentual: todas as variáveis e pares

O notebook [09_percentuais_guiados.ipynb](../notebooks/09_percentuais_guiados.ipynb) é a grade sistemática principal. Ele produz:

| Bloco | Cobertura | Rótulo e resposta |
|---|---:|---|
| Mix univariado | 10 entradas × 20 níveis assinados = **200** cenários | Reponderação de `(X,Y)`; PSI, ESS, prevalência e quatro métricas. |
| Valor univariado | 10 entradas × 20 níveis assinados = **200** | Alteração de `X`; PSI, score, mudança efetiva e decisões. |
| Mix de pares | 45 pares × 4 direções × 10 magnitudes = **1.800** | Reponderação de perfis conjuntos; PSI marginal e conjunto, ESS e quatro métricas. |
| Valores de pares | 45 pares × 4 direções × 10 magnitudes = **1.800** | Alteração conjunta de `X`; PSI, score e decisões. |
| Transições de atraso | **60** cenários adicionais | Mudanças discretas para contagens em que multiplicar por 1,05 pode não alterar nada. |

As magnitudes são 5%, 10%, ..., 50%. O univariado usa direções `−` e `+`. Em pares, `++`, `+-`, `-+` e `--` significam os sinais das variáveis A e B; cada combinação é avaliada em todas as dez magnitudes. **Zero é o basal**, usado como referência, e não é contado entre os 4.060 cenários. Os CSVs completos estão em `results/tables/`. Nos cenários de **valor em pares**, o notebook 09 usa uma amostra piloto de 5.000 linhas por custo computacional; o mix rotulado usa o holdout de 22.500 linhas. Compare as duas famílias como mecanismos diferentes, não como métricas da mesma população.

No mix, “+10%” se refere à **participação do grupo alto** definida no notebook, sujeita a viabilidade e normalização; não eleva cada valor individual em 10%. Em valores, “+10%” multiplica o valor registrado, mas arredondamentos, zeros e limites podem reduzir a mudança efetiva. Em ambos, o PSI é calculado depois do choque. Não equipare `10%` solicitado a `PSI=0,10`.

O notebook [17_atlas_todas_variaveis_e_pares.ipynb](../notebooks/17_atlas_todas_variaveis_e_pares.ipynb) organiza a leitura visual de todas as entradas e pares a partir desses CSVs. As figuras do relatório usam poucos exemplos para serem legíveis em página; **isso não restringe o número de cenários executados**.

## 5. Mapa dos notebooks

| Nº | Método e pergunta | Mudança / `Y` | Como replicar em outra base |
|---|---|---|---|
| 00 | **Preparação:** qual Booster, split, baseline e hash estão sendo avaliados? | Retreina e substitui artefato; procedimento separado. | Congele modelo, semente, ordem e tipos de entrada, política de ausentes e referência. |
| 01 | **Drift univariado de mix:** ao favorecer valores altos/baixos de uma entrada, como mudam as métricas? | Pesos por entrada em linhas `(X,Y)). | Defina perfil de pesos sem olhar `Y`, bins fixos, grade, ESS e bootstrap pareado. |
| 02 | **Mix multivariado legível:** e se um perfil conjunto ganhar participação? | Peso de coorte definida por duas ou mais entradas. | Especifique coortes com regras de `X`, normalize pesos e examine marginais e conjunto. |
| 03 | **Sensibilidade sintética dirigida:** como scores mudam ao alterar valor e propagar parte do choque? | `X` de `cs-test`; sem `Y`. | Escolha escala, intensidade, grau de propagação e regras de domínio. |
| 04 | **Univariado extensivo:** quais trajetórias de pesos geram respostas diferentes na mesma variável? | 2.688 cenários reponderados. | Varra postos, centro, caudas, extremos e ausência; confira rotinas ponderadas contra scikit-learn. |
| 05 | **Multivariado populacional extensivo:** quais interações/famílias dão respostas mais severas dentro da busca? | 3.072 cenários de mix. | Declare famílias/direções e separe exploração da confirmação; mostre ESS. |
| 06 | **Sintético multivariado:** como localização, caudas e dependência mudam scores em novas combinações? | Gerador estatístico de `X`; sem `Y`. | Valide suporte, restrições, correlação e intensidade realizada; evite leitura causal. |
| 07 | **Importance weighting observado:** como aproximar o mix de `cs-test`? | Razão de densidades aplicada ao holdout rotulado. | Ajuste classificador de domínio, examine sobreposição e truncamento de pesos. |
| 08 | **DRO por grupos:** qual a resposta a um mix adversarial dentro dos grupos escolhidos? | Pesos de grupos de `X`, otimizados para Brier. | Defina grupos antes de selecionar extremos e confirme em outra metade. |
| 09 | **Grade de percentuais:** o que `±5%` a `±50%` vira em PSI, métricas e decisões? | Todos os 10 univariados e 45 pares; mix e valores separados. | Replique a grade assinada, restrições e filtros de saída; compare pedido versus realizado. |
| 10 | **Painel descritivo:** como comparar drift observado e resposta histórica sem inventar alerta? | `cs-test` sem `Y`; usa saídas 04/05. | Congele bins, janela e referência; mostre ruído amostral sem corte automático. |
| 11 | **Média, dispersão e forma:** como a renda alterada se propaga estatisticamente às demais entradas? | `X` sintético; sem novo `Y`. | Separe localização, escala, cauda e grau de propagação; compare correlações. |
| 12 | **Qualidade de medição:** qual o impacto de falhas de registro/ingestão? | Mesmo `Y` se apenas entrada registrada muda. | Modele mecanismos reais de erro, fração atingida e persistência do desfecho. |
| 13 | **Cobertura/extrapolação:** o modelo é frágil fora dos perfis densos do treino? | Distância, clusters e pesos em linhas históricas. | Ajuste representação só no treino; relate `n`, positivos, ESS e incerteza por janela. |
| 14 | **Ruptura condicional:** pode haver deterioração com PSI de entrada zero? | `X` fixo e `Y` hipotético. | Declare segmento/regra de troca; não apresente rótulos simulados como observação. |
| 15 | **Carga e contrato:** como inferência e esquema reagem a lotes e campos problemáticos? | Teste operacional em `cs-test`; sem `Y`. | Separe preparação e inferência, repita medições, valide tipos/ordem/ausência. |
| 16 | **Calibração, segmentos e corte:** onde o score e decisões próximas de 0,50 são frágeis? | Erro de registro em bandas de score com `Y` histórico. | Fixe bandas sem consultar `Y`, mostre tamanho e eventos por segmento. |
| 17 | **Atlas de cobertura:** como visualizar cada variável e par da grade completa? | Lê os CSVs do 09; não cria novo `Y`. | Gere painéis por entrada e mapas dos 45 pares com legenda de mecanismo. |

Os cadernos `03`, `11` e `16` usam variáveis ou segmentos **dirigidos** para demonstrar mecanismos particulares. A cobertura exaustiva das dez entradas e 45 pares está no `09` e no atlas `17`. “Todo o modelo” aqui significa todas as **entradas incluídas** nessas grades, não todas as combinações possíveis de distribuições ou futuros econômicos.

## 6. Passos para transferir a metodologia a outro contexto

1. **Defina o evento, horizonte e uso do score.** Estabeleça se deseja ordenação, probabilidade calibrada ou decisão, e fixe o corte antes de examinar o holdout.
2. **Separe uma avaliação rotulada.** Use holdout fora do treino e, se houver datas, reserve também um conjunto temporal final. Não selecione cenários e declare confirmação no mesmo subconjunto.
3. **Congele o artefato e o contrato.** Salve hash do modelo, ordem/tipos das entradas, imputação, versões, bins e referência. Uma nova versão do Booster exige nova linha de base.
4. **Classifique o mecanismo.** Mix, valor real, erro de medição, dependência e mudança de `P(Y|X)` pedem suposições diferentes. Escreva a regra de cenário antes do resultado.
5. **Construa uma grade gradual com sinais.** Inclua zero, direções opostas e intensidades coerentes com a variável; para contagens/zeros, adicione transições discretas. Registre o solicitado e o realizado.
6. **Meça drift com bins fixos.** Calcule PSI e outra medida conforme a variável; observe score e relações entre entradas. Um único PSI marginal não descreve todo cenário multivariado.
7. **Use `Y` somente quando válido.** Em mix, pondere rótulos históricos sob hipótese de estabilidade e verifique ESS/suporte; em valor contrafactual sem novo desfecho, não declare desempenho.
8. **Compare quatro métricas e contexto.** Brier, AUC ROC, PR-AUC trapezoidal e KS do modelo, com prevalência, ESS, recall/precisão diagnósticos e incerteza. PR-AUC depende da prevalência.
9. **Preserve todos os pontos.** Mostre curvas intensidade→PSI, PSI→`ΔM`, direções, dispersão e exemplos de métricas discordantes. Não escolha apenas o pior ponto.
10. **Decida ações fora da simulação.** Para produção, a instituição precisa escolher uma política de impacto, janela, incerteza, investigação e dados temporais rotulados. Este projeto entrega evidência para essa decisão, não um limite universal.

## 7. Verificações e limites

A suíte confronta métricas ponderadas vetorizadas com scikit-learn, exige retorno à linha de base com pesos uniformes, distingue KS do modelo de KS de drift e preserva todos os cenários em CSV. O holdout foi reutilizado em muitos testes; o cenário mais extremo entre milhares pode refletir seleção. Sob pesos concentrados, ESS baixo enfraquece a evidência. Geradores sintéticos preservam algumas associações estatísticas, mas não identificam causalidade nem fornecem rótulos futuros. Nenhum desses testes é validação temporal ou autorização de uso em produção.



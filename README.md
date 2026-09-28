# Testes de estresse e drift de um modelo de crédito

Projeto didático e reproduzível para investigar **quanto as entradas de um classificador podem mudar e como o modelo responde**. O caso usa dez variáveis de crédito, um Booster XGBoost congelado e dois arquivos: `cs-training.csv`, com rótulos, e `cs-test.csv`, sem rótulos observados.

O estudo não declara um valor universal de PSI “seguro”. Ele registra, para cada mecanismo e intensidade testados, o **drift efetivamente produzido** e a mudança nas métricas que podem ser calculadas. Assim, outra equipe pode estudar as curvas e definir uma política de monitoramento adequada ao seu uso do modelo.

## Comece por aqui

1. Leia [como instalar e executar](docs/GUIA_DE_EXECUCAO.md). Coloque os dois CSVs originais em `data/raw/`; eles são locais e não devem ser publicados automaticamente.
2. Consulte [o mapa das metodologias](docs/METODOLOGIA.md) para escolher a pergunta certa: mudança do mix de clientes, alteração de valores, erro de medição, ruptura do risco ou carga operacional.
3. Execute os notebooks `01` a `17` com o **modelo congelado** de `models/`. O notebook `00` retreina o modelo e é um procedimento separado; veja o guia antes de usá-lo.
4. Leia [o relatório completo](reports/RELATORIO_DIDATICO.md), os [conceitos e siglas](docs/DICIONARIO.md) e [como localizar e filtrar os resultados](docs/RESULTADOS_E_FILTROS.md).

**Cobertura das entradas:** o notebook `09` executa a grade de −50% a +50% em passos de 5% para **todas as dez variáveis** isoladamente e para **todos os 45 pares**, com as quatro combinações de sinais nos pares. O notebook `17` reúne visualizações por variável e por par. Algumas figuras do relatório mostram duas entradas apenas para facilitar a leitura; os CSVs e os notebooks guardam a cobertura integral. Outros notebooks estudam mecanismos específicos ou exemplos dirigidos, identificados no [mapa](docs/METODOLOGIA.md).

## O que cada resultado significa

- **Métricas centrais com rótulo válido:** Brier, AUC ROC, PR-AUC trapezoidal e KS **do modelo**. Recall e precisão no corte 0,50 são referências descritivas. AP aparece separadamente; não é a PR-AUC trapezoidal.
- **Drift das entradas:** PSI, KS **de drift**, Wasserstein e mudanças de distribuição/associação conforme o cenário. PSI não é porcentagem e não indica sozinho perda de desempenho.
- **Cenários de composição:** reponderam linhas históricas completas `(X,Y)`; permitem calcular métricas supervisionadas **sob a hipótese** de que o risco condicional permanece estável e os perfis têm suporte nos dados.
- **Cenários de valor econômico hipotético:** mudam `X` do mesmo cliente sem observar seu novo `Y`. Neles se medem drift, score e decisões, não AUC/Brier de um mundo contrafactual.
- **Erros de registro:** podem usar o `Y` histórico quando apenas o dado entregue ao modelo foi corrompido e o cliente/desfecho real permaneceu igual.
- **`cs-test.csv`:** não possui rótulos observados. É usado para drift, scores e testes operacionais, nunca para inventar desempenho supervisionado.

O modelo salvo em `models/model_class_retreinado.json` é um **retreino auditável** com `scale_pos_weight=13.882494684620836` e corte de decisão `0,50`. Ele **não é** uma cópia serializada do objeto original `model_class` do notebook recebido. O [manifesto](models/manifesto_modelo.json) registra parâmetros, divisão, hashes e política de predição. O holdout é aleatório e rotulado, mas **não é uma validação temporal**.

O parâmetro `n_estimators=800` era um limite de treino; a parada antecipada deixou 103 rodadas armazenadas. As previsões deste projeto usam as 103 rodadas, conforme o manifesto.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `data/raw/` | Lugar dos CSVs locais. Consulte o guia antes de executar; dados brutos não são resultado gerado. |
| `models/` | Booster congelado e manifesto de proveniência. |
| `notebooks/` | Experimentos numerados `00` a `17`, com explicações e saídas executadas. |
| `scripts/` | Executor, síntese, auditoria e manutenção dos artefatos publicados. |
| `results/tables/` | CSVs com cenários completos, resumos e painéis. |
| `results/figures/` | PNGs produzidos por notebooks específicos. |
| `reports/` | Relatório didático em Markdown e PDF, com figuras selecionadas. |
| `docs/` | Início rápido, metodologia, dicionário e instruções para ler os resultados. |
| `reference/` | Notebook de modelagem e cópias históricas exploratórias, separados da suíte executável; revise outputs/caminhos antes de publicar. |

## Linha de base e limites

No holdout de 22.500 observações, a linha de base registrada no manifesto é: AUC ROC **0,866840**, PR-AUC trapezoidal **0,395991**, KS do modelo **0,587714** e Brier **0,144038**. O Brier do preditor constante na prevalência é menor; portanto, os scores do modelo ponderado **não devem ser tratados como probabilidades calibradas de inadimplência** sem trabalho adicional. Os números deste repositório descrevem este conjunto de dados, esta divisão e este Booster.

As simulações não substituem uma coorte temporal com desfechos maduros, uma justificativa econômica dos cenários ou uma decisão de risco sobre qual deterioração exige ação. O relatório explica essas limitações e mostra exemplos numéricos sem chamar extremos observados de “tolerância aprovada”.

## Navegação

- [Guia de execução](docs/GUIA_DE_EXECUCAO.md)
- [Metodologias e cobertura](docs/METODOLOGIA.md)
- [Dicionário e fórmulas](docs/DICIONARIO.md)
- [Resultados, filtros e leitura dos gráficos](docs/RESULTADOS_E_FILTROS.md)
- [Relatório completo — Markdown](reports/RELATORIO_DIDATICO.md)
- [Relatório para leitura — PDF](reports/RELATORIO_DIDATICO.pdf)



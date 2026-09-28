# Guia de execução

Este guia começa **sem depender de nenhuma conversa anterior**. O objetivo é executar a suíte com o Booster já salvo, reproduzir as tabelas e entender quando o resultado contém desempenho supervisionado ou apenas sensibilidade de score.

## 1. Arquivos necessários

Depois de clonar ou copiar este projeto, a estrutura mínima deve conter:

```text
credit-drift-stress-tests/
├── data/raw/
│   ├── cs-training.csv       ← copie localmente
│   └── cs-test.csv           ← copie localmente
├── models/
│   ├── model_class_retreinado.json
│   └── manifesto_modelo.json
├── notebooks/
├── scripts/
└── requirements.txt
```

Os nomes dos CSVs correspondem aos disponibilizados na competição [Give Me Some Credit, página oficial de dados](https://www.kaggle.com/competitions/GiveMeSomeCredit/data). O acesso pode exigir conta e aceitação das regras da competição; a página informa licença sujeita a essas regras. Baixe os arquivos por essa página se ainda não os tiver, copie-os para `data/raw/` e compare seus hashes com o manifesto. Os CSVs desta análise foram fornecidos localmente ao projeto.

`cs-training.csv` contém o alvo `SeriousDlqin2yrs`; `cs-test.csv` contém essa coluna sem rótulos observados. Não preencha a coluna vazia com rótulos aleatórios. O projeto mantém `data/raw/` fora do versionamento por padrão. Se você trocar os dados, considere isso um **novo experimento**: o manifesto, as contagens, os bins, o modelo e as tabelas deixam de representar a nova base.

O manifesto registra os hashes SHA-256 esperados e a ordem das dez entradas. Ao receber os arquivos de outra pessoa, confira se são a mesma versão antes de comparar números. O projeto não pressupõe acesso aos caminhos pessoais de quem criou a análise.

## 2. Ambiente Python

A execução registrada usou Python 3.12 e as versões fixadas em `requirements.txt`. A partir da **raiz do repositório**, crie um ambiente isolado:

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se `python3.12` não estiver instalado, use uma versão compatível com todas as dependências fixadas e registre a diferença. Para abrir os notebooks interativamente, execute `jupyter lab` na raiz e selecione o kernel desse ambiente.

## 3. Executar a análise padrão com o modelo congelado

Confirme que os dois CSVs estão em `data/raw/` e que o Booster e manifesto estão em `models/`. Depois execute:

```bash
python -u scripts/executar_notebooks.py 01 17
python scripts/sintetizar_respostas.py
python scripts/auditar_variante.py
```

Os dois números do executor são **início e fim inclusivos**. A execução começa em `01` porque `00` é um procedimento de **retreino**; com o modelo congelado, o hash e a linha de base do manifesto permanecem os do estudo. O notebook `17` é o atlas visual de todas as entradas e pares. A sequência pode levar tempo; cada notebook salva suas saídas no próprio arquivo e os CSVs/PNGs em `results/`.

Se você quiser executar apenas um caderno:

```bash
python -u scripts/executar_notebooks.py 09 09
```

Verifique antes as dependências de resultados: `10` usa tabelas de `04` e `05`; `17` usa os quatro CSVs percentuais de `09`. Rodar `01–17` em ordem cobre essas dependências. A auditoria técnica confere notebooks, métricas e tabelas; **não** valida a plausibilidade econômica nem aprova o modelo para produção.

## 4. Onde encontrar cada resultado

| Pergunta | Abra primeiro | Tabelas principais |
|---|---|---|
| Qual é o modelo e a linha de base? | `models/manifesto_modelo.json`, notebook `00` | Manifesto e relatório |
| Como cada entrada afeta o desempenho sob mudança de mix? | `01`, `04`, `09` | `resultados_univariado_extensivo.csv`, `resultados_percentuais_mix_univariado.csv` |
| Quanto um aumento/redução de valor muda PSI, score e decisões? | `09`, atlas `17` | `resultados_percentuais_valores_univariado.csv` |
| E se duas entradas mudarem juntas? | `02`, `05`, `09`, `17` | `resultados_percentuais_mix_pares.csv`, `resultados_percentuais_valores_pares.csv` |
| E se as dependências forem propagadas? | `03`, `06`, `11` | `resultados_sinteticos.csv`, `resultados_media_dispersao_forma.csv` |
| Como se comporta a população sem rótulos? | `07`, `10`, `15` | `resultados_drift_observado.csv`, `painel_drift_descritivo.csv` |
| Que outras fragilidades existem? | `08`, `12–16` | CSVs correspondentes em `results/tables/` |

A lista completa e um exemplo de filtro por variável estão em [Resultados e filtros](RESULTADOS_E_FILTROS.md). As imagens do relatório são **exemplos selecionados** para caber na página; o atlas e os CSVs conservam todas as dez entradas e os 45 pares.

## 5. Retreino opcional: notebook 00

O notebook `00_preparacao_modelo.ipynb` relê a base, repete o split, treina um **novo** Booster, salva o modelo e reescreve o manifesto. Faça isso apenas quando quiser estudar um novo artefato ou reproduzir o treino em seu ambiente:

```bash
python -u scripts/executar_notebooks.py 00 00 --retrain-model
python -u scripts/executar_notebooks.py 01 17
python scripts/sintetizar_respostas.py
python scripts/auditar_variante.py
```

**Antes de executar `00`, copie `models/` para um local de arquivo se precisar preservar o Booster fornecido.** Mesmo com os mesmos parâmetros e semente, versões, hardware e detalhes do treino podem mudar o hash e as predições. A configuração recebida inclui `device=cuda`, `tree_method=gpu_hist` e `scale_pos_weight=13.882494684620836`; confira a compatibilidade do XGBoost e da GPU no seu ambiente. O `scale_pos_weight` não é um limiar de decisão. O limiar de decisão fixo para os diagnósticos é **0,50**.

Se um novo modelo for salvo, todos os resultados e o relatório numérico anteriores passam a se referir ao modelo antigo até serem regenerados. A versão para GitHub apresenta os resultados do Booster cujo SHA-256 aparece no manifesto.

## 6. Ler os números sem misturar cenários

1. **Identifique o mecanismo.** Mix de clientes altera pesos de linhas `(X,Y)`; choque de valor altera `X`; erro de medição altera apenas a entrada registrada; troca de rótulo no notebook `14` é hipótese ilustrativa.
2. **Leia a intensidade realizada.** `+5%` solicitado pode gerar PSI quase zero ou grande, conforme distribuição, bins, zeros e arredondamento. Percentual e PSI têm unidades diferentes.
3. **Confirme se há `Y` válido.** Brier, AUC ROC, PR-AUC e KS do modelo só são interpretáveis com desfechos observados ou sob hipótese explícita. Em `cs-test` e choques econômicos contrafactuais, olhe score, decisões e drift.
4. **Compare com o basal e com o suporte.** Para Brier, `Δ>0` é piora; para AUC, PR-AUC e KS do modelo, `Δ<0` é piora. Em cenários com pesos, leia também ESS e prevalência.
5. **Leia a curva inteira.** Um extremo observado em milhares de cenários não é um limiar de tolerância. O estudo não usa recall/precisão nem PSI como regra de parada.

O [dicionário](DICIONARIO.md) define cada sigla e fórmula; a [metodologia](METODOLOGIA.md) explica as hipóteses de cada teste.

## 7. Problemas comuns

| Sintoma | O que verificar |
|---|---|
| `FileNotFoundError` para um CSV | Nome exato, pasta `data/raw/` e diretório de execução na raiz do projeto. |
| Hash/contagem diferente do manifesto | Arquivos de outra versão ou alterados. Registre uma nova proveniência antes de comparar resultados. |
| Notebook `10` ou `17` sem CSV de entrada | Execute `04`/`05` ou `09`, respectivamente, ou use `01–17` na ordem. |
| Erro de kernel ou biblioteca | Ative o ambiente e instale `requirements.txt`; confira `python --version` e o kernel do Jupyter. |
| Erro de CUDA no `00` | O retreino usa parâmetros de GPU. Use o Booster congelado para executar `01–17`, ou adapte e documente um novo experimento de treino. |
| AUC/Brier ausentes em cenário de valor | Comportamento esperado: o novo desfecho do cliente não foi observado. |
| Diferença entre AP e PR-AUC | São integrações distintas da curva precisão–recall; veja o dicionário. |




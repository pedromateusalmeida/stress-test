# Notebooks

Os cadernos 01–17 usam o Booster congelado em `models/`. O notebook 00 retreina e é opcional. Execute da raiz do repositório com `python -u scripts/executar_notebooks.py 01 17`; consulte o [guia](../docs/GUIA_DE_EXECUCAO.md) para preparar dados e ambiente.

| Bloco | Cadernos | Pergunta |
|---|---|---|
| Modelo e baseline | 00 | Proveniência, split, treino e avaliação inicial. |
| Mix de clientes | 01, 02, 04, 05, 07, 08, 13 | Como métricas respondem a reponderação de perfis históricos? |
| Valores e dependência | 03, 06, 11 | Como entradas, score e decisão mudam sem novo rótulo observado? |
| Grade cruzada | 09 | Como todas as dez entradas e os 45 pares respondem a mix e valor em passos de 5%? |
| Observação e operação | 10, 15 | O que medir no arquivo sem rótulos e no contrato de inferência? |
| Falhas e limites | 12, 14, 16 | Medição, risco condicional, calibração, segmentos e corte. |
| Atlas completo | 17 | Como visualizar **cada uma das dez entradas e 45 pares** do 09? |

O notebook 09 contém os quatro blocos sistemáticos: mix e valor univariados (200 cenários cada), mix e valor de pares (1.800 cenários cada), além de 60 transições discretas. Alguns outros cadernos usam exemplos dirigidos para estudar um mecanismo. O [mapa metodológico](../docs/METODOLOGIA.md) explica o escopo e a validade de `Y` em cada um.


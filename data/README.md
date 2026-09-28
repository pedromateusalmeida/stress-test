# Dados de entrada

Copie `cs-training.csv` e `cs-test.csv` para `data/raw/` antes de executar os notebooks. Essa pasta é ignorada pelo Git; as cópias locais podem permanecer aqui sem entrar em um commit. Os arquivos originais fora deste projeto também permanecem preservados.

- `cs-training.csv`: base rotulada usada para treino, validação e holdout.
- `cs-test.csv`: base sem desfechos observados, usada para drift, score e operação. A coluna de alvo vazia não deve receber rótulos fictícios.

Os nomes correspondem aos arquivos da competição [Give Me Some Credit, página oficial de dados](https://www.kaggle.com/competitions/GiveMeSomeCredit/data), cujo acesso está sujeito às regras da competição. Obtenha os dados por essa fonte se não tiver as cópias locais e confirme os hashes antes de reproduzir os resultados. Os SHA-256 dos dois arquivos usados no estudo estão em [models/manifesto_modelo.json](../models/manifesto_modelo.json). Dados de outra versão exigem nova proveniência, novo modelo e reexecução antes de comparar os números do relatório. Veja o [guia](../docs/GUIA_DE_EXECUCAO.md).


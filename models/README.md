# Modelo e proveniência

`model_class_retreinado.json` é o Booster XGBoost **congelado** para os notebooks 01–17. `manifesto_modelo.json` registra hash SHA-256 do Booster, ordem das dez entradas, origem dos CSVs, parâmetros, split, corte 0,50 e métricas basais.

Este Booster foi **retreinado** com `scale_pos_weight=13.882494684620836`; o objeto original `model_class` da sessão de notebook não foi recebido serializado. O score bruto não foi demonstrado como PD calibrada.

O notebook 00 é opcional e **substitui** modelo/manifesto se for executado. Preserve uma cópia deste diretório antes de retreinar e reexecute todos os testes para obter resultados referentes ao novo hash. O [guia](../docs/GUIA_DE_EXECUCAO.md) descreve o procedimento.

Deep Flight Predictor✈️
Este repositório contém o desenvolvimento de um agente aprendiz baseado numa Rede Neuronal Multicamada (Multilayer Perceptron) para a previsão de atrasos em voos.
O projeto foi desenvolvido na unidade curricular de Inteligência Artificial da Licenciatura em Engenharia Informática. O objetivo principal é implementar um modelo de classificação binária capaz de prever se um voo sofrerá atraso (Classe 1) ou não (Classe 0) com base em atributos operacionais simulados.
Arquitetura da Rede
Camada de Entrada: Neurónios correspondentes ao vetor de atributos concatenado.
Camada Escondida: 8 neurónios (configuração base).
Camada de Saída: 2 neurónios (representando as duas classes via one-hot).
Experiências Realizadas
O projeto inclui uma análise experimental detalhada variando:
Número de neurónios na camada escondida (4 a 20).
Número de épocas de treino (10 a 200).
Taxa de aprendizagem (learning rate).
Impacto da normalização e seleção de atributos.
As métricas avaliadas incluem Exatidão, Precisão, Recall (Cobertura) e F1-Score.
Autor: David Power
Instituição: Politécnico de Castelo Branco - Escola Superior de Tecnologia